# marble/encoders/YuE/model.py
from typing import Dict
from pathlib import Path

import torch
from transformers.modeling_outputs import BaseModelOutput

from marble.core.base_encoder import BaseEncoder
from marble.core.base_transform import BaseAudioTransform


class YuEEncoder(BaseEncoder):
    """
    A wrapper around YuE's Stage-1 LLaMA model for extracting audio representations.

    YuE is an autoregressive causal LM (LLaMA2-based). This wrapper tokenizes audio
    via X-Codec, maps tokens into YuE's vocabulary, and extracts hidden states from
    the LLaMA transformer layers for use as audio embeddings.
    """
    NAME = "YuE"
    HUGGINGFACE_MODEL_NAME = "m-a-p/YuE-s1-7B-anneal-en-cot"
    SAMPLING_RATE = 16000  # X-Codec operates at 16kHz
    TOKEN_RATE = 50  # X-Codec frame rate: 16000 / hop_length(320) = 50 Hz
    NUM_FEATURES = 4096  # default hidden size (varies by model size: 4096 7B, 1024 0.5B)
    N_TRANSFORMER_LAYERS = 32  # default (varies by model size: 32 7B, 24 0.5B)

    # YuE vocabulary mapping for X-Codec codebook-0
    # xcodec config: codebook_size=1024, num_codebooks=12, global_offset=45334
    XCODEC_GLOBAL_OFFSET = 45334
    SOA_TOKEN_ID = 32001   # <SOA>
    EOA_TOKEN_ID = 32002   # <EOA>
    XCODEC_SEP_TOKEN_ID = 32016  # <xcodec>

    MODEL_CONFIGS = {
        "7B": "m-a-p/YuE-s1-7B-anneal-en-cot",
        "0.5B": "m-a-p/YuE-s1-0.5B",
    }

    def __init__(
        self,
        pre_trained_folder: str = None,
        model_size: str = "7B",
        train_mode: str = "freeze",
    ) -> None:
        """
        Initialize YuE encoder wrapper.

        Args:
            pre_trained_folder: Path or HF identifier of the pretrained model
            model_size: Size variant of YuE Stage-1 model
                - "7B"   (4096 hidden, 32 layers, ~7B params) - m-a-p/YuE-s1-7B-anneal-en-cot
                - "0.5B" (1024 hidden, 24 layers, ~0.5B params) - m-a-p/YuE-s1-0.5B
            train_mode: "freeze" to freeze parameters, "full" for fine-tuning
        """
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoConfig

        repo = pre_trained_folder or self.MODEL_CONFIGS.get(model_size, model_size)
        self.model_size = model_size
        self.sample_rate = self.SAMPLING_RATE

        print(f"Loading YuE Stage-1 model from {repo}")
        # YuE Stage-1 structure:
        # - LlamaForCausalLM (standard HuggingFace LLaMA2)
        # - Vocabulary includes text tokens (0-31999), special tokens (32000-32021),
        #   and X-Codec tokens (45334-57621, 12 codebooks x 1024)
        # - For representation extraction, we feed codebook-0 tokens only
        #   in unconditional single-track mode (no text/lyrics conditioning)

        config = AutoConfig.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.bfloat16
        )

        self.NUM_FEATURES = config.hidden_size
        self.N_TRANSFORMER_LAYERS = config.num_hidden_layers

        # Load X-Codec for audio tokenization (codebook-0 at 0.5 kbps)
        self._load_xcodec()

        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            for param in self.codec_model.parameters():
                param.requires_grad = False
            self.codec_model.eval()
        elif train_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    def _load_xcodec(self):
        """Load X-Codec (SoundStream) model for audio-to-token conversion."""
        from omegaconf import OmegaConf
        from huggingface_hub import hf_hub_download
        from marble.encoders.Xcodec.models.soundstream_hubert_new import SoundStream

        hf_repo = "m-a-p/xcodec"
        cache_dir = Path.home() / ".cache" / "xcodec"
        cache_dir.mkdir(parents=True, exist_ok=True)

        config_path = cache_dir / "config.yaml"
        ckpt_filename = "ckpt_00360000.pth"
        ckpt_path = cache_dir / ckpt_filename

        if not config_path.is_file():
            print(f"Downloading X-Codec config from '{hf_repo}'...")
            hf_hub_download(
                repo_id=hf_repo, filename="config.yaml",
                local_dir=cache_dir, local_dir_use_symlinks=False,
            )
        if not ckpt_path.is_file():
            print(f"Downloading X-Codec checkpoint from '{hf_repo}'...")
            hf_hub_download(
                repo_id=hf_repo, filename=ckpt_filename,
                local_dir=cache_dir, local_dir_use_symlinks=False,
            )

        # SoundStream.__init__ loads HuBERT from this local path
        hubert_dir = cache_dir / "semantic_ckpts" / "hf_1_325000"
        if not (hubert_dir / "pytorch_model.bin").is_file():
            print(f"Downloading HuBERT semantic checkpoint from '{hf_repo}'...")
            for fn in ["config.json", "preprocessor_config.json", "pytorch_model.bin"]:
                hf_hub_download(
                    repo_id=hf_repo,
                    filename=f"semantic_ckpts/hf_1_325000/{fn}",
                    local_dir=cache_dir, local_dir_use_symlinks=False,
                )

        config = OmegaConf.load(config_path)
        self.codec_model = SoundStream(**config.generator.config)

        params = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.codec_model.load_state_dict(params["codec_model"])

        for param in self.codec_model.parameters():
            param.requires_grad = False
        self.codec_model.eval()

    @torch.no_grad()
    def _audio_to_token_ids(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform [B, 1, T] to YuE vocabulary token IDs [B, T_frames + 3]."""
        codec_device = next(self.codec_model.parameters()).device
        codec_dtype = next(self.codec_model.parameters()).dtype
        audio = audio.to(device=codec_device, dtype=codec_dtype)

        # X-Codec encode: [B, 1, T] -> codes [n_q=1, B, T_frames]
        codes = self.codec_model.encode(audio, target_bw=0.5, mode="indices")

        # Codebook-0 tokens offset to YuE vocabulary range [45334, 46357]
        token_ids = codes[0].long() + self.XCODEC_GLOBAL_OFFSET  # [B, T_frames]

        # Wrap: [SOA, xcodec_sep, ...audio_tokens..., EOA]
        B = token_ids.shape[0]
        device = token_ids.device
        soa = torch.full((B, 1), self.SOA_TOKEN_ID, dtype=torch.long, device=device)
        sep = torch.full((B, 1), self.XCODEC_SEP_TOKEN_ID, dtype=torch.long, device=device)
        eoa = torch.full((B, 1), self.EOA_TOKEN_ID, dtype=torch.long, device=device)

        return torch.cat([soa, sep, token_ids, eoa], dim=1)

    def forward(
        self,
        input_values: torch.Tensor,
        output_hidden_states: bool = True,
        **kwargs,
    ):
        """
        Forward pass through YuE to extract representations.

        Args:
            input_values: Audio waveform tensor, shape (batch_size, num_samples)
                          Values should be in range [-1, 1] at 16kHz
            output_hidden_states: If True, return all layer hidden states
            **kwargs: Additional arguments passed to LLaMA model

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: (batch_size, seq_len, hidden_size)
                - hidden_states: tuple of (batch_size, seq_len, hidden_size) for each layer
        """
        model_parameters = next(self.model.parameters())
        device = model_parameters.device
        model_dtype = model_parameters.dtype
        input_values = input_values.to(device=device, dtype=model_dtype)

        # X-Codec expects shape (batch, channels, length)
        if input_values.ndim == 1:
            input_values = input_values.unsqueeze(0).unsqueeze(0)
        elif input_values.ndim == 2:
            input_values = input_values.unsqueeze(1)

        # Tokenize audio via X-Codec -> YuE vocabulary IDs
        input_ids = self._audio_to_token_ids(input_values)  # [B, T_frames + 3]
        input_ids = input_ids.to(device=device)

        # Forward through LLaMA
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Strip special tokens (2 prefix: SOA + xcodec_sep, 1 suffix: EOA)
        if output_hidden_states and outputs.hidden_states is not None:
            hidden_states = tuple(hs[:, 2:-1, :] for hs in outputs.hidden_states)
        else:
            hidden_states = None

        last_hidden = hidden_states[-1] if hidden_states else outputs.logits

        return BaseModelOutput(
            last_hidden_state=last_hidden,
            hidden_states=hidden_states,
        )


class YuEFeatureExtractor(BaseAudioTransform):
    """Audio preprocessing for YuE: ensures mono waveform at 16kHz."""
    NAME = "YuE"
    SAMPLING_RATE = 16000

    def __init__(
        self,
        pre_trained_folder: str = None,
        model_size: str = "7B",
        squeeze: bool = True,
    ) -> None:
        """
        Initialize YuE feature extractor.

        Args:
            pre_trained_folder: Unused, kept for interface consistency
            model_size: Unused, kept for interface consistency
            squeeze: If True, squeeze output to 1D (remove channel dimension)
        """
        super().__init__()
        self.sample_rate = self.SAMPLING_RATE
        self.squeeze = squeeze

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Preprocess audio for YuE (16kHz mono)."""
        x = sample["input_features"]
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1), \
            f"Input must be mono (1D or [1, T]), got shape {x.shape}"

        if x.ndim == 2:
            x = x.squeeze(0)

        if self.squeeze:
            x = x.squeeze()

        sample["input_features"] = x
        return sample

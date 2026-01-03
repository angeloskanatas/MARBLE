# marble/encoders/MusicGen/model.py
from typing import Dict
import torch
from transformers import AutoConfig
from marble.core.base_encoder import BaseEncoder
from marble.core.base_transform import BaseAudioTransform

# TODO: add stereo support?


class MusicGenEncoder(BaseEncoder):
    """
    A wrapper around MusicGen's decoder for extracting audio representations.
    
    MusicGen is an autoregressive decoder model. This wrapper extracts hidden states
    from the decoder's transformer layers for use as audio embeddings.
    """
    NAME = "MusicGen"
    HUGGINGFACE_MODEL_NAME = "facebook/musicgen-small"  # options: small, medium, large, (stereo-small, stereo-medium, stereo-large)
    SAMPLING_RATE = 32000
    TOKEN_RATE = 50  # EnCodec tokens sampled at 50 Hz
    NUM_FEATURES = 1024  # default hidden size (varies by model size: 1024 small, 1536 medium, 2048 large)
    N_TRANSFORMER_LAYERS = 24  # default (varies by model size: 24 small, 48 medium/large)

    def __init__(
        self,
        pre_trained_folder: str = None,
        model_size: str = "small",
        train_mode: str = "freeze",
        random_init: bool = False,
    ) -> None:
        """
        Initialize MusicGen encoder wrapper.
        
        Args:
            pre_trained_folder: Path or HF identifier of the pretrained model
            model_size: Size variant of MusicGen model
            train_mode: "freeze" to freeze parameters, "full" for fine-tuning
            random_init: If True, initialize decoder with random weights
        """
        super().__init__()
        from transformers import MusicgenForConditionalGeneration
        
        repo = pre_trained_folder or f"facebook/musicgen-{model_size}"
        self.model_size = model_size
        self.sample_rate = self.SAMPLING_RATE
        
        print(f"Loading MusicGen model from {repo}")
        # MusicGen structure (from config.json):
        # - audio_encoder: EnCodec model (facebook/encodec_32khz)
        #   - codebook_size: 2048 (vocab_size per codebook) - consistent across all model sizes
        #   - num_codebooks: 4 for mono, 8 for stereo (4 per channel, interleaved)
        #   - sampling_rate: 32000 Hz
        #   - audio_channels: 1 (mono) or 2 (stereo)
        # - decoder: Autoregressive transformer decoder
        #   - num_codebooks: 4 (mono) or 8 (stereo)
        #   - vocab_size: 2048 (matches codebook_size)
        #   - hidden_size: varies by model (1024 small, 1536 medium, 2048 large)
        #   - num_hidden_layers: varies by model (24 small, 48 medium/large)

        # For representation extraction, we use the decoder's hidden states
        if random_init:  # random initialization
            config = AutoConfig.from_pretrained(repo)
            self.full_model = MusicgenForConditionalGeneration(config)
            pretrained_model = MusicgenForConditionalGeneration.from_pretrained(repo)
            self.full_model.audio_encoder = pretrained_model.audio_encoder
        else:  # pretrained model
            self.full_model = MusicgenForConditionalGeneration.from_pretrained(repo)

        self.model = self.full_model.decoder
        self.audio_encoder = self.full_model.audio_encoder
        
        decoder_config = self.full_model.config.decoder
        self.NUM_FEATURES = getattr(decoder_config, 'hidden_size', 1024)
        self.N_TRANSFORMER_LAYERS = getattr(decoder_config, 'num_hidden_layers', 24)
        
        num_codebooks = getattr(decoder_config, 'num_codebooks', 4)
        vocab_size = getattr(decoder_config, 'vocab_size', 2048)
        audio_encoder_config = self.full_model.config.audio_encoder
        audio_channels = getattr(audio_encoder_config, 'audio_channels', 1)
        is_stereo = 'stereo' in model_size.lower() or audio_channels == 2
        
        if is_stereo and num_codebooks != 8:
            print(f"Warning: Stereo model expected 8 codebooks, got {num_codebooks}")
        elif not is_stereo and num_codebooks != 4:
            print(f"Warning: Mono model expected 4 codebooks, got {num_codebooks}")
        
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.is_stereo = is_stereo
        
        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            self.audio_encoder.eval()
        elif train_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    def forward(
        self,
        input_values: torch.Tensor,
        output_hidden_states: bool = True,
        **kwargs
    ):
        """
        Forward pass through MusicGen decoder to extract representations.
        
        Args:
            input_values: Audio waveform tensor, shape (batch_size, num_samples)
                          Values should be in range [-1, 1] at 32kHz
            output_hidden_states: If True, return all layer hidden states
            **kwargs: Additional arguments passed to decoder
        
        Returns:
            BaseModelOutput containing:
                - last_hidden_state: (batch_size, seq_len, hidden_size)
                - hidden_states: tuple of (batch_size, seq_len, hidden_size) for each layer
        """
        model_parameters = next(self.model.parameters())
        device = model_parameters.device
        model_dtype = model_parameters.dtype
        input_values = input_values.to(device=device, dtype=model_dtype)
        
        # EnCodec expects shape (batch, channels, length)
        if input_values.ndim == 1:
            # (length,) -> (1, 1, length)
            input_values = input_values.unsqueeze(0).unsqueeze(0)
        elif input_values.ndim == 2:
            # (batch, length) -> (batch, 1, length) for mono
            input_values = input_values.unsqueeze(1)
        # If already 3D (batch, channels, length), use as-is
        
        # MusicGen uses EnCodec to tokenize audio into discrete tokens
        # Then the decoder processes these tokens to generate representations
        
        # Step 1: Tokenize audio using EnCodec (audio_encoder)
        # EnCodec.encode() returns tuple: (codes_tensor, [None])
        # codes_tensor shape: (batch, 1, num_codebooks=4, seq_len) for mono
        encode_result = self.audio_encoder.encode(input_values, return_dict=False)
        if isinstance(encode_result, tuple) and len(encode_result) > 0:
            audio_codes = encode_result[0]
        elif isinstance(encode_result, torch.Tensor):
            audio_codes = encode_result
        else:
            raise ValueError(f"Unexpected encode() return type: {type(encode_result)}")
        
        if not isinstance(audio_codes, torch.Tensor):
            raise ValueError(f"audio_codes is not a tensor: {type(audio_codes)}")
        
        # Step 2: Reshape to (batch, num_codebooks, seq_len)
        # audio_codes shape from EnCodec:
        #   - Mono: (batch, 1, num_codebooks=4, seq_len)
        #   - Stereo: (batch, 1, num_codebooks=8, seq_len) with interleaved codebooks
        if audio_codes.ndim == 4:
            input_ids = audio_codes.squeeze(1)  # (batch, num_codebooks, seq_len)
        elif audio_codes.ndim == 3:
            # already (batch, num_codebooks, seq_len)
            input_ids = audio_codes
        elif audio_codes.ndim == 2:
            # (batch, seq_len) -> (batch, 1, seq_len) for single codebook
            input_ids = audio_codes.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected audio_codes tensor shape: {audio_codes.shape}")
        
        input_ids = input_ids.to(device=device, dtype=torch.long)
        
        # Step 3: Pass input_ids to MusicGen decoder
        # MusicGen's decoder (MusicgenForCausalLM) accepts input_ids with shape
        # (batch, num_codebooks, seq_len) where num_codebooks=4 for mono, 8 for stereo
        # 
        # How the decoder processes multiple codebooks:
        # 1. Decoder has get_input_embeddings() that returns ModuleList with num_codebooks embedding layers
        # 2. Each codebook's tokens (vocab_size=2048) are embedded separately using its corresponding embedding layer
        # 3. The embeddings from all codebooks are summed to create combined representation
        # 4. This combined representation is passed through transformer layers
        # 5. We extract hidden_states from all transformer layers for representation extraction
        #
        # For stereo models: codebooks are interleaved [1_L, 1_R, 2_L, 2_R, 3_L, 3_R, 4_L, 4_R]
        # The decoder processes all 8 codebooks together.
        outputs = self.model(
            input_ids=input_ids,  # shape: (batch, num_codebooks, seq_len)
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )
        
        return outputs


class MusicGenFeatureExtractor(BaseAudioTransform):
    """
    Audio preprocessing for MusicGen encoder using HuggingFace's MusicgenProcessor.
    
    Uses the official HuggingFace processor with EncodecFeatureExtractor:
    - Resamples audio to 32kHz (MusicGen's expected sample rate)
    - Handles normalization automatically
    - Returns preprocessed waveform ready for encoder
    """
    NAME = "MusicGen"
    HUGGINGFACE_MODEL_NAME = "facebook/musicgen-small"
    SAMPLING_RATE = 32000
    
    def __init__(
        self,
        pre_trained_folder: str = None,
        model_size: str = "small",
        squeeze: bool = True,
    ) -> None:
        """
        Initialize MusicGen feature extractor using HuggingFace processor.
        
        Args:
            pre_trained_folder: Path or HF identifier of the pretrained model
            model_size: Size variant of MusicGen model (small, medium, large, etc.)
            squeeze: If True, squeeze output to 1D (remove batch dimension)
        """
        super().__init__()
        from transformers import MusicgenProcessor
        
        repo = pre_trained_folder or f"facebook/musicgen-{model_size}"
        self.processor = MusicgenProcessor.from_pretrained(repo)
        self.feature_extractor = self.processor.feature_extractor
        self.sample_rate = self.feature_extractor.sampling_rate
        self.squeeze = squeeze

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Preprocess audio for MusicGen using HuggingFace's MusicgenProcessor."""
        x = sample["input_features"]
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1), \
            f"Input must be a 1D tensor (batch_size=1), got shape {x.shape}"  # FIXME: stereo support?
        
        if x.ndim == 2:
            x = x.squeeze(0)
        
        sr = sample.get("sampling_rate", self.sample_rate)
        audio_array = x.detach().cpu().numpy()
        inputs = self.processor(
            audio=audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            padding=False,
        )
        
        x = inputs.input_values
        if self.squeeze:
            x = x.squeeze()
        
        sample["input_features"] = x
        return sample

# marble/encoders/CLAP/model.py
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import sys
from pathlib import Path

from marble.core.base_encoder import BaseEncoder
from marble.core.base_transform import BaseAudioTransform


class CLAPEncoder(BaseEncoder):
    """
    CLAP (Contrastive Language-Audio Pretraining) encoder wrapper.
    
    Uses HTS-AT (Hierarchical Token-Semantic Audio Transformer) as the audio encoder.
    """
    NAME = "CLAP"
    SAMPLING_RATE = 48000  # CLAP uses 48kHz by default
    N_TRANSFORMER_LAYERS = 18  # number of SwinTransformerBlocks (for base model)

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        train_mode: str = "freeze",
        precomputed_lms: bool = True,
        spec_augment: bool = False,
        model_name: str = "base",  # "tiny", "base", or "large"
        enable_fusion: bool = False,
        fusion_type: str = 'None',
        **kwargs
    ) -> None:
        """
        Initialize CLAP encoder.
        
        Args:
            checkpoint_path: Path to CLAP checkpoint (.pt file)
            train_mode: "freeze" to freeze parameters, "full" for fine-tuning
            precomputed_lms: If True, expects precomputed mel-spectrograms
            spec_augment: If True, apply SpecAugment during training
            model_name: HTSAT model size ("tiny", "base", "large")
            enable_fusion: Whether fusion is enabled
            fusion_type: Fusion type if enabled
        """
        super().__init__()
        
        # make sure to git clone the CLAP repo (git clone https://github.com/LAION-AI/CLAP.git)
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "CLAP" / "src"))
        from laion_clap.clap_module.htsat import create_htsat_model
        from laion_clap.clap_module.model import CLAPAudioCfp
        
        self.sample_rate = self.SAMPLING_RATE
        self.precomputed_lms = precomputed_lms
        self.model_name = model_name
        
        audio_cfg = CLAPAudioCfp(
            model_type="HTSAT",
            model_name=model_name,
            sample_rate=self.SAMPLING_RATE,
            audio_length=1024,
            window_size=1024,
            hop_size=1024,
            fmin=50,
            fmax=14000,
            class_num=527,
            mel_bins=64,
            clip_samples=480000,
        )
        
        self.model = create_htsat_model(
            audio_cfg,
            enable_fusion=enable_fusion,
            fusion_type=fusion_type
        )
        
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        elif train_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load HTS-AT encoder weights from CLAP checkpoint.
        
        Args:
            checkpoint_path: Path to CLAP checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        if state_dict and next(iter(state_dict.keys())).startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        htsat_weights = {}
        prefix = "audio_branch."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                htsat_weights[new_key] = v
        
        if not htsat_weights:
            raise ValueError(
                f"No HTS-AT encoder weights found in checkpoint. "
                f"Expected keys starting with 'audio_branch.', "
                f"found keys: {list(state_dict.keys())[:5]}..."
            )
        
        missing, unexpected = self.model.load_state_dict(htsat_weights, strict=False)
        if missing:
            print(f"Warning: Missing keys when loading checkpoint: {len(missing)} keys")
        if unexpected:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected)} keys")

    def forward(
        self,
        x: torch.Tensor,
        output_hidden_states: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through HTS-AT encoder.
        
        Args:
            x: Mel-spectrogram tensor, shape (batch, 1, time_steps, mel_bins=64)
            output_hidden_states: If True, return all intermediate representations
            **kwargs: Additional arguments
        
        Returns:
            Tuple of hidden states (22 total: PatchEmbed + 18 blocks + 3 PatchMergings).
            Each tensor has shape (batch, seq_len, hidden_dim) where seq_len and hidden_dim vary by stage.
        """
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(device=device, dtype=model_dtype)
        
        if x.ndim == 2:
            # (T, F) -> (1, 1, T, F)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            # (B, T, F) -> (B, 1, T, F)
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(f"Unexpected channel dimension: {x.shape}, expected (B, 1, T, F)")
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}, expected (B, 1, T, F=64)")
        
        x = x.transpose(1, 3)  # (B, F, T, C) -> (B, C, T, F)
        x = self.model.bn0(x)
        x = x.transpose(1, 3)
        
        if self.model.training and hasattr(self.model, 'spec_augmenter') and self.model.spec_augmenter is not None:
            x = self.model.spec_augmenter(x)
        
        x = self.model.reshape_wav2img(x)
        
        if not output_hidden_states:
            output_dict = self.model.forward_features(x)
            return (output_dict["embedding"],)
        
        hidden_states = self.model.forward_features_with_hidden_states(x)
        return hidden_states


class CLAPFeatureExtractor(BaseAudioTransform):
    """Audio preprocessing for CLAP encoder."""
    NAME = "CLAP"
    SAMPLING_RATE = 48000
    
    def __init__(
        self,
        precomputed_lms: bool = False,
        squeeze: bool = True,
    ) -> None:
        """Initialize CLAP feature extractor.
        
        Args:
            precomputed_lms: If True, expects precomputed mel-spectrograms
            squeeze: Unused, kept for API compatibility
        """
        super().__init__()
        self.precomputed_lms = precomputed_lms
        self.squeeze = squeeze
        
        if not precomputed_lms:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "CLAP" / "src"))
            from torchlibrosa.stft import Spectrogram, LogmelFilterBank
            
            self.spectrogram_extractor = Spectrogram(
                n_fft=1024,
                hop_length=1024,  # CLAP uses 1024 hop
                win_length=1024,
                window='hann',
                center=True,
                pad_mode='reflect',
                freeze_parameters=True
            )
            self.spectrogram_extractor.double()
            
            self.logmel_extractor = LogmelFilterBank(
                sr=self.SAMPLING_RATE,
                n_fft=1024,
                n_mels=64,
                fmin=50,
                fmax=14000,
                ref=1.0,
                amin=1e-10,
                top_db=None,
                freeze_parameters=True
            )
            self.logmel_extractor.double()
        else:
            self.spectrogram_extractor = None
            self.logmel_extractor = None
        
        self.sample_rate = self.SAMPLING_RATE

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Preprocess audio for CLAP encoder.
        
        Args:
            sample: Dict with "input_features" (waveform) and "sampling_rate"
        
        Returns:
            Dict with "input_features" as mel-spectrogram tensor
        """
        x = sample["input_features"]
        assert isinstance(x, torch.Tensor)
        
        if self.precomputed_lms:
            if x.ndim == 2:
                x = x.unsqueeze(1)
            if x.ndim == 3 and x.shape[1] != 1:
                x = x.unsqueeze(1)
            sample["input_features"] = x
            return sample
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            assert x.shape[0] == 1, f"Input must be mono (C=1), got shape {x.shape}"
            x = x.squeeze(0)
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Input must be 1D (T,) or 2D (C, T), got shape {x.shape}")
        
        dtype = x.dtype
        x = x.double()
        
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = x.type(dtype)
        
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.ndim == 3:
            if x.shape[0] != 1:
                x = x.unsqueeze(0)
        
        sample["input_features"] = x
        return sample


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wav = torch.randn(4, 48000 * 10)  # 10 seconds of audio at 48kHz
    
    feature_extractor = CLAPFeatureExtractor(precomputed_lms=False, squeeze=True)
    encoder = CLAPEncoder(
        checkpoint_path=None,
        train_mode="freeze",
        precomputed_lms=True,
    ).to(device).eval()
    
    batch_mel_specs = []
    for i in range(wav.shape[0]):
        sample = {"input_features": wav[i].cpu(), "sampling_rate": 48000}
        processed = feature_extractor(sample)
        mel_spec = processed["input_features"]
        if mel_spec.ndim == 4:
            mel_spec = mel_spec.squeeze(0)
        batch_mel_specs.append(mel_spec)
    
    mel_specs = torch.stack(batch_mel_specs, dim=0).to(device)
    
    with torch.no_grad():
        hidden_states = encoder(mel_specs, output_hidden_states=True)
    
    print(f"Total number of layers: {len(hidden_states)}")
    print(f"Output shape of each layer: {[layer.shape for layer in hidden_states]}")

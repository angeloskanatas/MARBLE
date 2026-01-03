# marble/encoders/SLAP/model.py
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import sys
from pathlib import Path

from marble.core.base_encoder import BaseEncoder
from marble.core.base_transform import BaseAudioTransform


class SLAPEncoder(BaseEncoder):
    """
    SLAP (Siamese Language-Audio Pretraining) encoder wrapper.
    
    Uses HTS-AT (Hierarchical Token-Semantic Audio Transformer) as the audio encoder.
    """
    NAME = "SLAP"
    SAMPLING_RATE = 44100
    N_TRANSFORMER_LAYERS = 18  # number of SwinTransformerBlocks

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        train_mode: str = "freeze",
        precomputed_lms: bool = True,
        spec_augment: bool = False,
        embed_dim: int = 128,
        depths: Tuple[int, ...] = (2, 2, 12, 2),
        num_heads: Tuple[int, ...] = (4, 8, 16, 32),
        patch_size: int = 16,
        patch_stride: Tuple[int, int] = (4, 4),
        **kwargs
    ) -> None:
        """
        Initialize SLAP encoder.
        
        Args:
            checkpoint_path: Path to SLAP Lightning checkpoint (.ckpt file)
            train_mode: "freeze" to freeze parameters, "full" for fine-tuning
            precomputed_lms: If True, expects precomputed mel-spectrograms
            spec_augment: If True, apply SpecAugment during training
            embed_dim: Embedding dimension (default: 128)
            depths: Number of blocks per stage (default: [2, 2, 12, 2])
            num_heads: Number of attention heads per stage (default: [4, 8, 16, 32])
            patch_size: Patch size for patch embedding (default: 16)
            patch_stride: Patch stride for patch embedding (default: (4, 4))
        """
        super().__init__()
        
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "SLAP" / "src"))  # make sure to clone the SLAP repo (git clone https://github.com/Pliploop/SLAP.git)
        from networks.audio.htsat import HTSATSwinTransformer
        
        self.sample_rate = self.SAMPLING_RATE
        self.precomputed_lms = precomputed_lms
        
        self.model = HTSATSwinTransformer(
            spec_size=256,
            patch_size=patch_size,
            patch_stride=patch_stride,
            mel_bins=64,
            sample_rate=self.SAMPLING_RATE,
            spec_window_size=1024,
            hop_size=480,
            fmin=50,
            fmax=14000,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=list(num_heads),
            window_size=8,
            precomputed_lms=precomputed_lms,
            spec_augment=spec_augment,
            checkpoint_path=None,
            **kwargs
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
        """Load HTS-AT encoder weights from SLAP Lightning checkpoint.
        
        Args:
            checkpoint_path: Path to SLAP Lightning checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        htsat_weights = {}
        prefix = "audio_encoder.encoder."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                htsat_weights[new_key] = v
        
        if not htsat_weights:
            raise ValueError(
                f"No HTS-AT encoder weights found in checkpoint. "
                f"Expected keys starting with 'audio_encoder.encoder.', "
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
        
        x = x.transpose(1, 3)
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


class SLAPFeatureExtractor(BaseAudioTransform):
    """Audio preprocessing for SLAP encoder."""
    NAME = "SLAP"
    SAMPLING_RATE = 44100
    
    def __init__(
        self,
        precomputed_lms: bool = False,
        squeeze: bool = True,
    ) -> None:
        """Initialize SLAP feature extractor.
        
        Args:
            precomputed_lms: If True, expects precomputed mel-spectrograms
            squeeze: Unused, kept for API compatibility
        """
        super().__init__()
        self.precomputed_lms = precomputed_lms
        self.squeeze = squeeze
        
        if not precomputed_lms:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "SLAP" / "src"))
            from torchlibrosa.stft import Spectrogram, LogmelFilterBank
            
            self.spectrogram_extractor = Spectrogram(
                n_fft=1024,
                hop_length=480,
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
        """Preprocess audio for SLAP encoder.
        
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
        
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4 and x.shape[1] != 1:
            x = x.unsqueeze(1)
        
        sample["input_features"] = x
        return sample


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wav = torch.randn(4, 44100 * 10)  # 10 seconds of audio at 44.1kHz
    
    feature_extractor = SLAPFeatureExtractor(precomputed_lms=False, squeeze=True)
    encoder = SLAPEncoder(
        checkpoint_path=None,
        train_mode="freeze",
        precomputed_lms=True,
    ).to(device).eval()
    
    batch_mel_specs = []
    for i in range(wav.shape[0]):
        sample = {"input_features": wav[i].cpu(), "sampling_rate": 44100}
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

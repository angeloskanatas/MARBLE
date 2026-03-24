"""
Myna-Base encoder wrapper for MARBLE extraction pipeline.

Myna is a contrastive self-supervised model using a SimpleViT (ViT-S/16)
backbone on mel-spectrograms with token masking as the sole augmentation.

Architecture: 12 transformer layers, dim=384, heads=6, mlp_dim=1536 (22M params).
Input: raw waveform at 16kHz -> 128-bin mel spectrogram -> 16x16 patches.

This wrapper extracts layerwise hidden states from the ViT backbone.
Returns 13 hidden states: patch embedding output + 12 transformer layer outputs.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from marble.core.base_encoder import BaseEncoder
from marble.core.base_transform import BaseAudioTransform

# Add myna repo to path so we can import its modules
MYNA_REPO = str(Path.home() / "projects" / "music-jepa" / "myna")
if MYNA_REPO not in sys.path:
    sys.path.insert(0, MYNA_REPO)


class MynaBaseEncoder(BaseEncoder):
    """
    MARBLE-compatible encoder wrapper for Myna-Base.

    Returns a tuple of 13 tensors (patch embedding + 12 transformer layers),
    each with shape [B, num_patches, 384].

    For audio longer than the chunk duration, splits into non-overlapping
    chunks and concatenates token sequences across chunks (preserving
    temporal structure for frame-level downstream tasks).
    """

    NAME = "Myna-Base"
    SAMPLING_RATE = 16000
    NUM_FEATURES = 384
    N_TRANSFORMER_LAYERS = 12

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        train_mode: str = "freeze",
        n_samples: int = 49000,  # ~3.06s at 16kHz; gives exactly 96 mel frames (zero alignment loss with patch_size=16)
    ) -> None:
        super().__init__()
        self.sample_rate = self.SAMPLING_RATE

        from nnAudio.features.mel import MelSpectrogram as nnMelSpectrogram
        from vit import SimpleViT

        # Mel spectrogram transform (same as Myna training)
        self.mel_spec = nnMelSpectrogram(sr=self.SAMPLING_RATE, n_mels=128, verbose=False)

        # Compute frame count for this chunk size, aligned to patch_size
        self.n_samples = n_samples
        with torch.no_grad():
            dummy_mel = self.mel_spec(torch.randn(1, 1, n_samples))
        self.n_frames = (dummy_mel.shape[-1] // 16) * 16

        # Compute token rate: patches along time axis per second
        patches_per_chunk = (128 // 16) * (self.n_frames // 16)
        self.TOKEN_RATE = patches_per_chunk / (n_samples / self.SAMPLING_RATE)

        # Build model
        self.model = SimpleViT(
            image_size=(128, self.n_frames),
            channels=1,
            patch_size=16,
            num_classes=50,  # placeholder, head is replaced
            dim=384,
            depth=12,
            heads=6,
            mlp_dim=1536,
        )

        # Load checkpoint (ignoring projector head from contrastive training)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            filtered = {
                k: v for k, v in checkpoint.items()
                if not k.startswith("linear_head")
            }
            self.model.load_state_dict(filtered, strict=False)
            print(f"==> Loaded Myna-Base checkpoint from {checkpoint_path} "
                  f"({len(filtered)}/{len(checkpoint)} keys)")

        # Replace classification head with identity (we want embeddings)
        self.model.linear_head = nn.Identity()

        # Freeze/train
        if train_mode == "freeze":
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        elif train_mode == "full":
            for param in self.parameters():
                param.requires_grad = True
            self.train()

    def _forward_chunk(self, mel_chunk: torch.Tensor):
        """Process a single mel-spectrogram chunk, returning layerwise hidden states.

        Args:
            mel_chunk: [B, 1, 128, n_frames] mel spectrogram.

        Returns:
            List of 13 tensors, each [B, num_patches, 384].
            Index 0 = patch embedding output, indices 1-12 = transformer layers.
        """
        # Patch embedding + positional encoding
        x = self.model.to_patch_embedding(mel_chunk)
        x = x + self.model.pos_embedding.to(x.device, dtype=x.dtype)

        hidden_states = [x]

        # Forward through transformer layers, collecting hidden states
        for attn, ff in self.model.transformer.layers:
            x = attn(x) + x
            x = ff(x) + x
            hidden_states.append(x)

        # Apply final LayerNorm to last hidden state (matches model's forward)
        hidden_states[-1] = self.model.transformer.norm(hidden_states[-1])

        return hidden_states

    def forward(
        self,
        x: torch.Tensor,
        *args,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: Waveform tensor [B, num_samples] or [B, 1, num_samples] at 16kHz.

        Returns:
            Tuple of 13 tensors (patch embedding + 12 layers).
            Each tensor shape: [B, num_patches, 384] for single chunk,
            or [B, n_chunks * num_patches, 384] for multi-chunk audio.
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        x = x.to(device=device, dtype=dtype)

        # Handle channel dimension: (B, 1, T) -> (B, T)
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Compute mel spectrogram: (B, T) -> (B, 128, T_mel)
        mel = self.mel_spec(x.unsqueeze(1).to(device))
        B = mel.shape[0]
        total_frames = mel.shape[-1]

        # Determine chunks
        n_chunks = max(1, total_frames // self.n_frames)

        # Short audio: pad mel to n_frames
        if total_frames < self.n_frames:
            padded = torch.zeros(B, 128, self.n_frames, device=device, dtype=dtype)
            padded[:, :, :total_frames] = mel
            mel_chunk = padded.unsqueeze(1)  # (B, 1, 128, n_frames)
            return tuple(self._forward_chunk(mel_chunk))

        # Single chunk: process directly
        if n_chunks == 1:
            mel_chunk = mel[:, :, :self.n_frames].unsqueeze(1)
            return tuple(self._forward_chunk(mel_chunk))

        # Multi-chunk: process each chunk, concatenate token sequences
        all_chunk_hs = []
        for i in range(n_chunks):
            start = i * self.n_frames
            end = start + self.n_frames
            chunk = mel[:, :, start:end].unsqueeze(1)  # (B, 1, 128, n_frames)
            hs = self._forward_chunk(chunk)
            all_chunk_hs.append(hs)

        # Concatenate tokens across chunks for each layer
        num_layers = len(all_chunk_hs[0])
        concatenated = []
        for layer_idx in range(num_layers):
            layer_chunks = [all_chunk_hs[c][layer_idx] for c in range(n_chunks)]
            concatenated.append(torch.cat(layer_chunks, dim=1))  # (B, n_chunks*patches, 384)

        return tuple(concatenated)


class MynaFeatureExtractor(BaseAudioTransform):
    """Feature extractor for Myna: ensures mono waveform at 16kHz.

    The mel spectrogram is computed inside the encoder (not here),
    matching the MT2 pattern where spectral transforms are model-internal.
    """

    NAME = "Myna-Base"
    SAMPLING_RATE = 16000

    def __init__(self, squeeze: bool = True, **kwargs) -> None:
        super().__init__()
        self.squeeze = squeeze
        self.sample_rate = self.SAMPLING_RATE

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            sample: Dict with "input_features" (waveform [C, T]) and "sampling_rate".

        Returns:
            sample dict with mono waveform as "input_features".
        """
        x = sample["input_features"]

        # Ensure mono
        if x.ndim == 2 and x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)

        # Squeeze channel dim if requested: (1, T) -> (T,)
        if self.squeeze and x.ndim == 2 and x.shape[0] == 1:
            x = x.squeeze(0)

        sample["input_features"] = x
        return sample


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test with default checkpoint path
    ckpt = str(Path.home() / "projects" / "music-jepa" / "myna" / "myna-base.pth")
    try:
        model = MynaBaseEncoder(checkpoint_path=ckpt)
    except FileNotFoundError:
        print(f"Checkpoint not found at {ckpt}, testing without pretrained weights")
        model = MynaBaseEncoder(checkpoint_path=None)

    model = model.to(device).eval()

    print(f"Model: {model.NAME}")
    print(f"Layers: {model.N_TRANSFORMER_LAYERS}")
    print(f"Features: {model.NUM_FEATURES}")
    print(f"Chunk frames: {model.n_frames}")
    print(f"Token rate: {model.TOKEN_RATE:.1f} patches/sec")

    # Test 3s input (single chunk)
    wav_3s = torch.randn(2, 16000 * 3)
    with torch.no_grad():
        output = model(wav_3s.to(device))
    print(f"\n3s input:")
    print(f"  Layers: {len(output)}, shape: {output[0].shape}")

    # Test 10s input (multi-chunk)
    wav_10s = torch.randn(2, 16000 * 10)
    with torch.no_grad():
        output_10s = model(wav_10s.to(device))
    print(f"\n10s input:")
    print(f"  Layers: {len(output_10s)}, shape: {output_10s[0].shape}")

    # Test short input (1s, needs padding)
    wav_1s = torch.randn(2, 16000)
    with torch.no_grad():
        output_1s = model(wav_1s.to(device))
    print(f"\n1s input (padded):")
    print(f"  Layers: {len(output_1s)}, shape: {output_1s[0].shape}")

"""
MT2 encoder wrapper for MARBLE extraction pipeline.

MT2 is a Vision Transformer with two class tokens:
  - cls_stone (position 0): equivariant token (tonal tasks, CPSD loss)
  - cls_contrastive (position 1): contrastive token (semantic tasks, NT-Xent loss)

This wrapper extracts layerwise hidden states from the ViT backbone.
The `pooling_mode` parameter controls which tokens are returned:

  - "full" (default): full sequence [B, 2 + seq_len, H] - both cls + seq tokens
  - "cls_equiv": equivariant cls token only [B, 1, H]
  - "cls_contrastive": contrastive cls token only [B, 1, H]
  - "cls_avg": average of both cls tokens [B, 1, H]
  - "seq_only": sequence tokens only [B, seq_len, H]

For >4s audio, the encoder chunks into 4s segments and:
  - Averages class tokens across chunks
  - Concatenates sequence tokens in temporal order
"""
import sys
import os
from pathlib import Path

import gin
import torch
import torchaudio
from torch import nn

from marble.core.base_encoder import BaseEncoder

# Add mt2 repo to path so we can import its modules
MT2_REPO = str(Path(__file__).resolve().parents[4] / "mt2")
if MT2_REPO not in sys.path:
    sys.path.insert(0, MT2_REPO)

VALID_POOLING_MODES = ("full", "cls_equiv", "cls_contrastive", "cls_avg", "seq_only")


class MT2_Encoder(BaseEncoder):
    """
    MARBLE-compatible encoder wrapper for MT2.

    Returns a tuple of (num_layers + 1) tensors whose shape depends on pooling_mode:
      - "full":            [B, 2 + seq_len, H]  (default — all tokens)
      - "cls_equiv":       [B, 1, H]            (equivariant cls only)
      - "cls_contrastive": [B, 1, H]            (contrastive cls only)
      - "cls_avg":         [B, 1, H]            (mean of both cls tokens)
      - "seq_only":        [B, seq_len, H]      (sequence tokens only)
    """

    NAME = "MT2"
    SAMPLING_RATE = 16000
    NUM_FEATURES = 192  # ViT embedding dimension
    N_TRANSFORMER_LAYERS = 12

    def __init__(
        self,
        pre_trained_folder: str = None,
        train_mode: str = "freeze",
        pooling_mode: str = "full",
    ) -> None:
        super().__init__()
        assert pooling_mode in VALID_POOLING_MODES, (
            f"pooling_mode must be one of {VALID_POOLING_MODES}, got '{pooling_mode}'"
        )
        self.pooling_mode = pooling_mode
        self.sample_rate = self.SAMPLING_RATE

        # Load MT2 with gin config
        gin_config = os.path.join(MT2_REPO, "config", "mt2.gin")
        gin.parse_config_file(gin_config, skip_unknown=True)

        from src.model import MT2 as MT2Model
        self.model = MT2Model(encoder_type="vit", device="cpu")

        # Compute actual token rate from model's CQT output
        self._seq_len = self.model.seq_len  # number of CQT time frames per 4s
        self.TOKEN_RATE = self._seq_len / self.model.duration  # frames per second

        # Load pretrained weights
        ckpt_path = pre_trained_folder or os.path.join(MT2_REPO, "model_state_dict.pt")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)

        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        elif train_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()

    def _forward_chunk(self, x):
        """Process a single 4-second chunk, returning layerwise hidden states.

        Args:
            x: [B, 64000] waveform at 16kHz (exactly 4 seconds).

        Returns:
            List of 13 tensors, each [B, 2 + seq_len, 192].
            Always returns full sequence; pooling is applied in _apply_pooling.
        """
        from einops import rearrange, repeat

        # CQT branch (equivariant)
        hcqt = self.model.hcqt(x)
        hcqt = rearrange(hcqt, "b 1 f t -> b t f")
        x_stone = self.model.norm_in_stone(hcqt)
        x_stone = self.model.act_in(self.model.linear_in_stone(x_stone))
        cls_stone = repeat(self.model.cls_stone, "e -> b 1 e", b=x_stone.shape[0])
        cls_stone = cls_stone + torch.mean(x_stone, dim=1, keepdim=True)

        # Mel branch (contrastive)
        mel = rearrange(self.model.spec(x.unsqueeze(-1)), "b 1 e t -> b t e")
        x_contrastive = self.model.norm_in_contrastive(mel)
        x_contrastive = self.model.act_in(
            self.model.linear_in_contrastive(x_contrastive)
        )
        cls_contrastive = repeat(
            self.model.cls_contrastive, "e -> b 1 e", b=x_contrastive.shape[0]
        )
        cls_contrastive = cls_contrastive + torch.mean(
            x_contrastive, dim=1, keepdim=True
        )

        # Fuse and prepend class tokens
        tokens = x_contrastive + x_stone
        tokens = torch.cat((cls_stone, cls_contrastive, tokens), dim=1)
        tokens = self.model.pos_emb(tokens)

        # Collect layerwise hidden states
        hidden_states = [tokens]
        for block in self.model.backbone.blocks:
            tokens = block(tokens)
            hidden_states.append(tokens)

        # Apply final layer norm to last layer
        hidden_states[-1] = self.model.backbone.norm(hidden_states[-1])

        return hidden_states

    def _apply_pooling(self, hidden_states):
        """Apply pooling_mode to select which tokens to return.

        Args:
            hidden_states: list of tensors, each [B, 2 + seq_len, H]

        Returns:
            Tuple of tensors with shape depending on pooling_mode.
        """
        if self.pooling_mode == "full":
            return tuple(hidden_states)

        result = []
        for h in hidden_states:
            if self.pooling_mode == "cls_equiv":
                result.append(h[:, 0:1, :])        # [B, 1, H]
            elif self.pooling_mode == "cls_contrastive":
                result.append(h[:, 1:2, :])        # [B, 1, H]
            elif self.pooling_mode == "cls_avg":
                avg = h[:, :2, :].mean(dim=1, keepdim=True)  # [B, 1, H]
                result.append(avg)
            elif self.pooling_mode == "seq_only":
                result.append(h[:, 2:, :])          # [B, seq_len, H]
        return tuple(result)

    def forward(
        self,
        x: torch.Tensor,
        *args,
        output_hidden_states: bool = True,
        **kwargs,
    ):
        """
        Args:
            x: Waveform tensor [B, num_samples] at 16kHz.

        Returns:
            Tuple of 13 tensors (input + 12 block outputs).
            Shape depends on pooling_mode (see class docstring).
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        x = x.to(device=device, dtype=dtype)

        # Squeeze channel dim: (B, 1, T) -> (B, T). Dataloader returns mono as (B, 1, T)
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        chunk_samples = self.model.sr * self.model.duration  # 16000 * 4 = 64000
        B = x.shape[0]

        # Short input: pad to 4s
        if x.shape[-1] < chunk_samples:
            padded = torch.zeros(B, chunk_samples, device=device, dtype=dtype)
            padded[:, : x.shape[-1]] = x
            return self._apply_pooling(self._forward_chunk(padded))

        # Exact 4s input: process directly
        if x.shape[-1] == chunk_samples:
            return self._apply_pooling(self._forward_chunk(x))

        # >4s input: chunk into non-overlapping 4s segments
        n_chunks = x.shape[-1] // chunk_samples

        # Process all chunks
        all_chunk_hs = []
        for i in range(n_chunks):
            chunk = x[:, i * chunk_samples : (i + 1) * chunk_samples]
            hs = self._forward_chunk(chunk)
            all_chunk_hs.append(hs)

        # Per layer: average cls tokens, concatenate seq tokens
        num_layers = len(all_chunk_hs[0])
        full_hidden_states = []
        for layer_idx in range(num_layers):
            cls_sum = None
            seq_parts = []
            for chunk_idx in range(n_chunks):
                layer_out = all_chunk_hs[chunk_idx][layer_idx]  # (B, 2+seq_len, H)
                cls = layer_out[:, :2, :]   # (B, 2, H)
                seq = layer_out[:, 2:, :]   # (B, seq_len, H)
                if cls_sum is None:
                    cls_sum = cls
                else:
                    cls_sum = cls_sum + cls
                seq_parts.append(seq)

            cls_avg = cls_sum / n_chunks              # (B, 2, H)
            seq_concat = torch.cat(seq_parts, dim=1)  # (B, n_chunks*seq_len, H)
            combined = torch.cat([cls_avg, seq_concat], dim=1)
            full_hidden_states.append(combined)

        return self._apply_pooling(full_hidden_states)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MT2_Encoder(pooling_mode="full")
    model = model.to(device).eval()

    print(f"Token rate: {model.TOKEN_RATE} fps")
    print(f"Seq len per 4s chunk: {model._seq_len}")

    # Test 4s input - all pooling modes
    wav_4s = torch.randn(2, 16000 * 4)
    with torch.no_grad():
        output = model(wav_4s.to(device))
    print(f"\n4s input, pooling_mode='full':")
    print(f"  Layers: {len(output)}, shape: {output[0].shape}")

    for mode in ("cls_equiv", "cls_contrastive", "cls_avg", "seq_only"):
        model_m = MT2_Encoder(pooling_mode=mode).to(device).eval()
        with torch.no_grad():
            out = model_m(wav_4s.to(device))
        print(f"  pooling_mode='{mode}': shape={out[0].shape}")

    # Test 10s input (chunked)
    wav_10s = torch.randn(2, 16000 * 10)
    with torch.no_grad():
        output_10s = model(wav_10s.to(device))
    print(f"\n10s input, pooling_mode='full':")
    print(f"  Layers: {len(output_10s)}, shape: {output_10s[0].shape}")

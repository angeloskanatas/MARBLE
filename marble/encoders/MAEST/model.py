# marble/encoders/MAEST/model.py
from pathlib import Path
import sys
from typing import Optional, Tuple

import torch

from marble.core.base_encoder import BaseEncoder


class MAESTEncoder(BaseEncoder):
    """
    A wrapper for MAEST (Music Audio Efficient Spectrogram Transformer).

    Input: 16 kHz waveform; mel-spectrogram is computed inside the model.
    Per-block output (MAEST forward_features): concat(cls, dist, mean(patch_tokens)) -> (B, 2304).
    """

    NAME = "MAEST"
    _DIM_PER_PART = 768  # cls, dist, mean_patch each 768

    def __init__(
        self,
        arch: str = "discogs-maest-10s-pw-129e",
        pretrained: bool = True,
        checkpoint: Optional[str] = None,
        train_mode: str = "freeze",
        device: Optional[str] = None,
        distilled_type: str = "mean",  # one of ["mean", "separated"]
        pooling: str = "concat",  # concat (2304) | cls | dist | mean_patch (768)
    ) -> None:
        """
        Initialize the MAEST encoder.

        Args:
            arch (str): Model architecture (e.g. discogs-maest-10s-pw-129e, discogs-maest-30s-pw-129e).
            pretrained (bool): If True, load weights from official release.
            checkpoint (str, optional): Path to local .ckpt; overrides pretrained if set.
            train_mode (str): "freeze" to freeze base parameters, "full" for full fine-tuning.
            device (str, optional): Device to load model on. If None, defaults to "cpu".
                PyTorch Lightning will move the model to the correct device automatically.
            distilled_type (str): "mean" or "separated"; passed to get_maest (classification head).
            pooling (str): "concat" (2304-d) or "cls"/"dist"/"mean_patch" (768-d slice per layer).
        """
        super().__init__()

        # MAEST repo must be a sibling of MARBLE (git clone https://github.com/palonso/MAEST).
        _maest_root = Path(__file__).resolve().parents[4] / "MAEST"
        sys.path.insert(0, str(_maest_root))
        from models import get_maest

        if device is None:
            device = "cpu"

        if pooling not in ("concat", "cls", "dist", "mean_patch"):
            raise ValueError(f"pooling must be one of concat, cls, dist, mean_patch, got {pooling!r}")

        self.model = get_maest(
            arch=arch,
            pretrained=pretrained if not checkpoint else False,
            checkpoint=checkpoint,
            distilled_type=distilled_type,
        )

        self.sample_rate = 16000
        self.num_layers = 12
        self.pooling = pooling
        self.embed_dim = 2304 if pooling == "concat" else self._DIM_PER_PART

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

        self.train_mode = train_mode

    def forward(
        self,
        x: torch.Tensor,
        *args,
        output_hidden_states: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Perform a forward pass through the MAEST encoder.

        Args:
            x (torch.Tensor): Waveform (batch_size, num_samples) or (batch_size, 1, num_samples), 16 kHz.
            output_hidden_states (bool): If True, return all intermediate hidden states.

        Returns:
            Tuple of torch.Tensor, one per layer. Each tensor has shape (batch_size, 2304) or (batch_size, 768) if pooling is cls/dist/mean_patch.
        """
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(device=model_device, dtype=model_dtype)

        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        if not output_hidden_states:
            _, emb = self.model(x, transformer_block=self.num_layers - 1, melspectrogram_input=False)
            return (self._pool(emb),)

        hidden_states = []
        for i in range(self.num_layers):
            _, emb = self.model(x, transformer_block=i, melspectrogram_input=False)
            hidden_states.append(self._pool(emb))

        return tuple(hidden_states)

    def _pool(self, emb: torch.Tensor) -> torch.Tensor:
        """Slice or pass through: MAEST returns concat(cls, dist, mean_patch) -> (B, 2304)."""
        if self.pooling == "concat":
            return emb
        if self.pooling == "cls":
            return emb[:, : self._DIM_PER_PART]
        if self.pooling == "dist":
            return emb[:, self._DIM_PER_PART : 2 * self._DIM_PER_PART]
        return emb[:, 2 * self._DIM_PER_PART :]

    @property
    def SAMPLING_RATE(self) -> int:
        """Audio sampling rate expected by the model."""
        return self.sample_rate

    @property
    def NUM_FEATURES(self) -> Optional[int]:
        """Hidden dimension of the model."""
        return self.embed_dim

    @property
    def N_TRANSFORMER_LAYERS(self) -> int:
        """Number of transformer layers in the backbone."""
        return self.num_layers

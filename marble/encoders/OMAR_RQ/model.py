# marble/encoders/OMAR_RQ/model.py
from typing import Optional, Tuple

import torch

from marble.core.base_encoder import BaseEncoder

try:
    from omar_rq import get_model
    OMAR_RQ_AVAILABLE = True
except ImportError:
    OMAR_RQ_AVAILABLE = False
    get_model = None


class OMAR_RQ_Encoder(BaseEncoder):
    """
    A wrapper for OMAR-RQ music audio representation model.

    Supports all OMAR-RQ variants:
    - base, multicodebook (16kHz, mel spectrogram)
    - multifeature, multifeature-25hz, multifeature-25hz-fsq (24kHz, audio)
    """

    NAME = "OMAR-RQ"

    def __init__(
        self,
        model_id: str,
        train_mode: str = "freeze",  # one of ["freeze", "full"]
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the OMAR-RQ encoder.

        Args:
            model_id (str): HuggingFace model ID (e.g., "mtg-upf/omar-rq-multifeature-25hz-fsq")
                or local path to model directory.
            train_mode (str): "freeze" to freeze base parameters, "full" for full fine-tuning.
            device (str, optional): Device to load model on. If None, defaults to "cpu".
                PyTorch Lightning will move the model to the correct device automatically.
        """
        if not OMAR_RQ_AVAILABLE:
            raise ImportError(
                "omar-rq is not installed. Please install it with: pip install omar-rq"
            )

        super().__init__()

        if device is None:
            device = "cpu"

        self.model = get_model(
            model_id=model_id,
            device=device,
            quantization_targets=False,
            load_weights=True,
        )

        self.sample_rate = self.model.sr
        self.embedding_rate = self.model.eps

        if hasattr(self.model.net, "layers"):
            self.num_layers = len(self.model.net.layers)
        elif hasattr(self.model.net, "transformer"):
            self.num_layers = len(self.model.net.transformer)
        elif hasattr(self.model.net, "depth"):
            self.num_layers = self.model.net.depth
        else:
            raise ValueError(
                f"Cannot determine number of layers from model.net. "
                f"Available attributes: {dir(self.model.net)}"
            )

        if hasattr(self.model.net, "embed_dim"):
            self.embed_dim = self.model.net.embed_dim
        else:
            self.embed_dim = None

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
        Perform a forward pass through the OMAR-RQ encoder.

        Args:
            x (torch.Tensor): Waveform tensor, shape (batch_size, num_samples), values in [-1, 1].
            output_hidden_states (bool): If True, return all intermediate hidden states.
            *args, **kwargs: Additional arguments passed to the underlying model.

        Returns:
            Tuple of torch.Tensor, one per layer. Each tensor has shape (batch_size, seq_len, hidden_dim).
        """
        model_parameters = next(self.model.parameters())
        model_dtype = model_parameters.dtype
        x = x.to(device=model_parameters.device, dtype=model_dtype)

        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        if output_hidden_states:
            layers = set(range(self.num_layers))
        else:
            layers = {self.num_layers - 1}  # last layer

        embeddings = self.model.extract_embeddings(x, layers=layers)

        if embeddings.ndim == 4:
            hidden_states = tuple(embeddings[i] for i in range(embeddings.shape[0]))
        elif embeddings.ndim == 3:
            hidden_states = (embeddings,)
        else:
            raise ValueError(
                f"Unexpected embedding shape: {embeddings.shape}. "
                f"Expected 3D (B, T, C) or 4D (L, B, T, C)."
            )

        return hidden_states

    @property
    def TOKEN_RATE(self) -> float:
        """Number of feature frames per second of audio (embedding rate)."""
        return self.embedding_rate

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
        """Number of transformer/conformer layers in the backbone."""
        return self.num_layers

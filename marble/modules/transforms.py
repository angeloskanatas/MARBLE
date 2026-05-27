# marble/modules/transforms.py
import hashlib
import math
import random
import re
from typing import TYPE_CHECKING, Sequence, Dict, Optional, Union, Tuple, List

if TYPE_CHECKING:
    import audiomentations

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from marble.core.base_transform import BaseEmbTransform, BaseAudioTransform

MAX_AUDIO_CHANNELS = 16


def _apply_audiomentations_to_waveform(
    waveform_np: np.ndarray,
    augmentation: 'audiomentations.Compose',
    sample_rate: int,
    max_channels: int = MAX_AUDIO_CHANNELS
) -> np.ndarray:
    """Apply audiomentations augmentation to a numpy waveform array.
    
    Args:
        waveform_np: Numpy array of shape (T,) or (C, T) or (T, C)
        augmentation: audiomentations.Compose object
        sample_rate: Audio sample rate
        max_channels: Maximum expected number of audio channels
    
    Returns:
        Augmented numpy array with same shape as input
    """
    if waveform_np.ndim == 1:
        return augmentation(samples=waveform_np, sample_rate=sample_rate)
    elif waveform_np.ndim == 2:
        dim0, dim1 = waveform_np.shape
        
        if dim0 <= max_channels and dim0 < dim1:
            waveform_format = 'channels_first'
        elif dim1 <= max_channels and dim1 < dim0:
            waveform_format = 'channels_last'
        else:
            waveform_format = 'channels_first' if dim0 <= dim1 else 'channels_last'
        
        if waveform_format == 'channels_first':
            waveform_for_aug = waveform_np
        else:
            waveform_for_aug = waveform_np.T
        
        augmented_np = augmentation(samples=waveform_for_aug, sample_rate=sample_rate)
        
        if waveform_format == 'channels_last':
            augmented_np = augmented_np.T
        
        return augmented_np
    else:
        raise ValueError(f"Unexpected waveform shape: {waveform_np.shape}")


############################## Audio Transforms ##############################
class AudioTransformDataset(torch.utils.data.Dataset):
    """Sequentially apply BaseAudioTransform instances on raw waveforms."""
    def __init__(self, base_dataset, transforms: list[BaseAudioTransform]):
        self.base = base_dataset
        self.transforms = transforms
        # assume base_dataset has sample_rate attribute
        self.sample_rate = getattr(base_dataset, "sample_rate", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # base[idx] returns:
        #   waveform: Tensor of shape [C, T] (or [1, T] for mono)
        #   label: any (e.g. int)
        #   path: str
        #   dataset_idx: int (optional, for deterministic seeding)
        base_item = self.base[idx]
        if len(base_item) > 3:
            waveform, label, path, dataset_idx = base_item
        else:
            waveform, label, path = base_item
            dataset_idx = idx

        # ensure waveform is [C, T]
        assert waveform.ndim == 2 and waveform.shape[0] > 0, \
            f"Expected waveform shape [C, T], got {waveform.shape}"

        sample = {
            "input_features": waveform,            # Tensor [C, T]
            "sampling_rate": self.sample_rate,  # int
            "audio_path": path,  # str
            "dataset_idx": dataset_idx  # int
        }

        # apply each transform in sequence
        for t in self.transforms:
            sample = t(sample)

        # final waveform
        final_input = sample["input_features"]         # Tensor [C, T] or [T] (for mert)
        return final_input, label, path


class AudioLayerNorm(BaseAudioTransform):
    """
    Normalize each channel to zero‐mean, unit‐variance over time.

    Args:
        eps (float): to avoid div by zero.
        affine (bool): if True, learn scale & bias per channel.
    """
    def __init__(self, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # gamma, beta: each [1, 1] (broadcast to [C, T])
            self.gamma = nn.Parameter(torch.ones(1, 1))
            self.beta  = nn.Parameter(torch.zeros(1, 1))

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # w: [C, T]
        w = sample["input_features"]
        mean = w.mean(dim=-1, keepdim=True)  # [C, 1]
        std  = w.std(dim=-1, keepdim=True)   # [C, 1]
        # normalized: [C, T]
        w_norm = (w - mean) / (std + self.eps)
        if self.affine:
            # broadcast gamma, beta to [C, T]
            w_norm = w_norm * self.gamma + self.beta
        sample["input_features"] = w_norm          # [C, T]
        return sample
    

class RandomCrop(BaseAudioTransform):
    def __init__(self, crop_size: int):
        """
        Args:
            crop_size (int): target length in samples (T_out).
        """
        super().__init__()
        self.crop_size = crop_size

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # waveform: [C, T]
        waveform = sample["input_features"]
        C, T = waveform.shape
        if T <= self.crop_size:
            pad = self.crop_size - T
            # pad to [C, crop_size]
            waveform = F.pad(waveform, (0, pad))
        else:
            start = random.randint(0, T - self.crop_size)
            # crop to [C, crop_size]
            waveform = waveform[:, start : start + self.crop_size]
        sample["input_features"] = waveform       # [C, crop_size]
        return sample


class AddNoise(BaseAudioTransform):
    """
    Adds random Gaussian noise to the waveform based on a random SNR."""
    def __init__(self, snr_min: float = 5.0, snr_max: float = 20.0): 
        super().__init__()
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self, sample):
        # waveform: [C, T]
        waveform = sample["input_features"]
        # 随机采样一个 SNR
        snr = torch.empty(1).uniform_(self.snr_min, self.snr_max).item() # scalar
        rms = waveform.pow(2).mean().sqrt() # scalar
        # noise: [C, T]
        noise_std = rms / (10 ** (snr / 20))
        noise = torch.randn_like(waveform) * noise_std
        sample["input_features"] = waveform + noise
        return sample


class AudiomentationsTransform(BaseAudioTransform):
    """Apply audiomentations augmentations to waveforms.
    
    Wraps audiomentations library as a BaseAudioTransform.
    """
    
    def __init__(
        self,
        augmentation_config: Optional[dict] = None,
        augmentation_pipeline: Optional['audiomentations.Compose'] = None,
        sample_rate: Optional[int] = None,
        seed: Optional[int] = None,
        aug_idx: Optional[int] = None,
    ):
        """
        Args:
            augmentation_config: Config dict for building augmentation pipeline
            augmentation_pipeline: Pre-built audiomentations.Compose instance (optional)
            sample_rate: Audio sample rate (required if using augmentation_config)
            seed: Base seed for deterministic augmentation
            aug_idx: Augmentation index for multiview (for deterministic seeding)
        """
        super().__init__()
        if augmentation_pipeline is not None:
            self.augmentation = augmentation_pipeline
        elif augmentation_config is not None and sample_rate is not None:
            self.augmentation = self._build_pipeline(augmentation_config, sample_rate)
        else:
            raise ValueError("Must provide either augmentation_pipeline or (augmentation_config and sample_rate)")
        self.seed = seed
        self.aug_idx = aug_idx
    
    def _build_pipeline(self, config: dict, sample_rate: int) -> 'audiomentations.Compose':
        """Build audiomentations pipeline from config."""
        try:
            import audiomentations  # noqa: F401
            import inspect
        except ImportError:
            raise ImportError("audiomentations library is required for AudiomentationsTransform")
        
        from audiomentations import Compose
        
        transforms_list = config.get('transforms', None)
        if transforms_list is None:
            transforms_list = [
                {'AddGaussianNoise': {'min_amplitude': 0.001, 'max_amplitude': 0.01, 'p': 0.8}},
                {'TimeStretch': {'min_rate': 0.95, 'max_rate': 1.05, 'p': 0.7}},
                {'Gain': {'min_gain_db': -3, 'max_gain_db': 3, 'p': 0.6}},
            ]
        
        transforms = []
        for aug_dict in transforms_list:
            for aug_name, aug_params in aug_dict.items():
                aug_module = __import__('audiomentations', fromlist=[aug_name])
                aug_class = getattr(aug_module, aug_name)
                
                aug_params = aug_params.copy()
                
                sig = inspect.signature(aug_class.__init__)
                valid_params = {}
                
                for param_name, param_value in aug_params.items():
                    if param_name in sig.parameters:
                        valid_params[param_name] = param_value
                    else:
                        print(f"Warning: {aug_name} does not accept parameter '{param_name}', skipping")
                
                if 'sample_rate' in sig.parameters and 'sample_rate' not in valid_params:
                    valid_params['sample_rate'] = sample_rate
                
                try:
                    transforms.append(aug_class(**valid_params))
                except TypeError as e:
                    raise ValueError(
                        f"Failed to instantiate {aug_name} with parameters {valid_params}. "
                        f"Available parameters: {list(sig.parameters.keys())}. Error: {e}"
                    )
        
        return Compose(transforms)

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation to waveform.
        
        Args:
            sample: Dict with "input_features" (waveform [C, T]), "sampling_rate", and optionally "audio_path"
        
        Returns:
            Dict with augmented waveform in "input_features"
        """
        waveform = sample["input_features"]
        sample_rate = sample["sampling_rate"]
        
        if self.seed is not None:
            seed = self.seed
            audio_path = sample.get("audio_path", None)
            dataset_idx = sample.get("dataset_idx", None)
            if audio_path is not None:
                path_hash = int(hashlib.md5(str(audio_path).encode()).hexdigest()[:8], 16)
                seed = seed + path_hash
            if dataset_idx is not None:
                seed = seed + dataset_idx
            if self.aug_idx is not None:
                seed += self.aug_idx
            np.random.seed(seed)
            random.seed(seed)
        
        waveform_np = waveform.detach().cpu().numpy()
        
        try:
            augmented_np = _apply_audiomentations_to_waveform(
                waveform_np, self.augmentation, sample_rate
            )
            sample["input_features"] = torch.from_numpy(augmented_np)
        except (ValueError, RuntimeError, AttributeError, TypeError) as e:
            print(f"Warning: Augmentation failed, using original waveform: {e}")
        
        return sample


class Resample(BaseAudioTransform):
    def __init__(self, orig_freq: int, new_freq: int):
        """
        Args:
            orig_freq (int): original sampling rate.
            new_freq  (int): desired sampling rate.
        """
        super().__init__()
        self.resampler = torchaudio.transforms.Resample(orig_freq, new_freq)

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input waveform: [C, T]
        out = self.resampler(sample["input_features"])
        # output waveform: [C, T_new]
        sample["input_features"] = out
        return sample


class Spectrogram(BaseAudioTransform):
    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        power: float = 2.0,
    ):
        """
        Args:
            n_fft (int): FFT window size.
            win_length (int): window length.
            hop_length (int): hop length between frames.
            power (float): exponent for magnitude.
        """
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length or n_fft,
            hop_length=hop_length or (win_length or n_fft)//2,
            power=power,
        )

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input waveform: [C, T]
        S = self.spec(sample["input_features"])
        # spectrogram: [C, F, T']
        sample["input_features"] = S
        return sample


class MelSpectrogram(BaseAudioTransform):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 400,
        n_mels: int = 80,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
    ):
        """
        Args:
            sample_rate (int): sampling rate.
            n_fft (int): FFT window size.
            n_mels (int): number of Mel bins.
            win_length (int): window length.
            hop_length (int): hop between frames.
        """
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length or n_fft,
            hop_length=hop_length or (win_length or n_fft)//2,
            n_mels=n_mels,
        )

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input waveform: [C, T]
        M = self.melspec(sample["input_features"])
        # mel spectrogram: [C, n_mels, T']
        sample["input_features"] = M
        return sample


############################## Embedding Transforms ##############################


class LayerSelector(BaseEmbTransform):
    """
    Selects a subset of hidden‐state layers.
    支持整型列表，也支持形如 "start..end" 的字符串范围。

    New: added support for "all" to automatically select all layers
    """
    RANGE_RE = re.compile(r"^(\d+)\.\.(\d+)$")

    def __init__(self, layers: Sequence[Union[int, str]]):
        super().__init__()
        self.layers_config = layers
        if isinstance(layers, (list, tuple)) and any(x == "all" or x == "ALL" for x in layers):
            self.layers = None
            print("LayerSelector initialized with 'all' - will auto-detect layers at runtime")
        else:
            self.layers = self._parse_layers(layers)
            print(f"LayerSelector initialized with layers: {self.layers}")

    def _parse_layers(self, layers):
        parsed = []
        for x in layers:
            if isinstance(x, str):
                if x.lower() == "all":
                    # should not reach here if handled in __init__, but just in case
                    return None
                m = self.RANGE_RE.match(x.strip())
                if m:
                    start, end = map(int, m.groups())
                    if end < start:
                        raise ValueError(f"Range end ({end}) < start ({start})")
                    parsed.extend(range(start, end+1))
                else:
                    # 如果不是范围，就尝试转成单个 int
                    parsed.append(int(x))
            else:
                parsed.append(int(x))
        return parsed

    def forward(self, hidden_states: Sequence[torch.Tensor], **kwargs) -> torch.Tensor:
        # detect all layers if "all" was specified
        if self.layers is None:
            self.layers = list(range(len(hidden_states)))
        
        selected = [hidden_states[i] for i in self.layers]
        stacked = torch.stack(selected, dim=1)
        assert stacked.ndim == 4, \
            f"Expected 4D tensor after stacking, got {stacked.ndim}D"
        return stacked


class LayerWeightedSum(BaseEmbTransform):
    """
    Learns a weighted sum over L layers via a 1×1 Conv1d.
    """
    def __init__(self, num_layers: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=num_layers, out_channels=1, kernel_size=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Weighted sum over layers, of shape
                (batch_size, 1, seq_len, hidden_size).
        """
        if isinstance(x, tuple):
            x = torch.stack(x, dim=1)
        x_flat = rearrange(x, 'b l t h -> b l (t h)')
        y = self.conv(x_flat)
        return rearrange(y, 'b 1 (t h) -> b 1 t h', h=x.size(-1))


class SoftmaxWeightedSum(BaseEmbTransform):
    """
    ELMo / Zhou et al. (2025) softmax-normalized weighted sum over L layers.

    output = gamma * sum_i( softmax(w)_i * layer_i )

    Unlike LayerWeightedSum (Conv1d, unconstrained weights with bias),
    this uses a learnable vector passed through softmax so weights are
    positive and sum to 1.  A learnable scalar gamma allows rescaling.
    """
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if isinstance(x, tuple):
            x = torch.stack(x, dim=1)
        # x: (B, L, T, H)
        w = F.softmax(self.weights, dim=0)          # (L,)
        y = torch.einsum('l, b l t h -> b t h', w, x)  # (B, T, H)
        y = self.gamma * y
        return y.unsqueeze(1)                        # (B, 1, T, H)


class LayerStack(BaseEmbTransform):
    """
    Concatenates selected layers along hidden dim.
    (B, L, T, H) → (B, 1, T, L*H)
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if isinstance(x, tuple):
            x = torch.stack(x, dim=1)
        return rearrange(x, 'b l t h -> b 1 t (l h)')


class MLPReduce(BaseEmbTransform):
    """
    Flattens layers & hidden dims and reduces via an MLP.
    """
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(num_layers * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Reduced representation of shape
                (batch_size, 1, seq_len, hidden_size).
        """
        if isinstance(x, tuple):
            x = torch.stack(x, dim=1)
        xt = rearrange(x, 'b l t h -> (b t) (l h)')
        y = self.fc(xt)
        return rearrange(y, '(b t) h -> b 1 t h', t=x.size(2))


class HConv(BaseEmbTransform):
    """
    Hierarchical 1D convolution over the layer dimension (Shih & Harwath, 2024).
    Applies floor(log_stride(L)) Conv1d layers with ReLU, progressively reducing
    the layer dimension. Convolutions treat hidden_size as channels and slide
    across layers.

    Reference: "Interface Design for Self-Supervised Speech Models" (Interspeech 2024)
    """
    def __init__(self, num_layers: int, hidden_size: int,
                 kernel_size: int = 5, stride: int = 3):
        super().__init__()

        L = num_layers
        num_convs = math.floor(math.log(L) / math.log(stride))
        if num_convs < 1:
            raise ValueError(
                f"num_layers={num_layers} too small for stride={stride}, "
                f"need at least {stride + 1} layers"
            )

        convs = []
        for i in range(num_convs):
            padding = kernel_size // 2
            L_out = (L + 2 * padding - kernel_size) // stride + 1

            if i == num_convs - 1:
                out_ch = hidden_size // L_out
            else:
                out_ch = hidden_size

            convs.append(nn.Conv1d(hidden_size, out_ch, kernel_size, stride, padding))
            convs.append(nn.ReLU())

            L = L_out

        self.transforms = nn.Sequential(*convs)
        self._raw_output_dim = out_ch * L

        if self._raw_output_dim != hidden_size:
            self.proj = nn.Linear(self._raw_output_dim, hidden_size)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer-stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Reduced representation of shape
                (batch_size, 1, seq_len, hidden_size).
        """
        if isinstance(x, tuple):
            x = torch.stack(x, dim=1)
        b, l, t, h = x.shape
        # (B, L, T, H) -> (B*T, H, L): conv over layer dimension
        x = rearrange(x, 'b l t h -> (b t) h l')
        x = self.transforms(x)
        # (B*T, C', L') -> (B*T, C'*L')
        x = x.reshape(b * t, -1)
        if self.proj is not None:
            x = self.proj(x)
        return rearrange(x, '(b t) h -> b 1 t h', b=b, t=t)


class TimeAdaptivePool(BaseEmbTransform):
    """
    Applies adaptive average pooling over time to a fixed length.
    """
    def __init__(self, target_frames: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(target_frames)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Time‐pooled tensor of shape
                (batch_size, num_layers, target_frames, hidden_size).
        """
        x2 = rearrange(x, 'b l t h -> (b l) h t')
        y = self.pool(x2)
        return rearrange(y, '(b l) h t -> b l t h', b=x.size(0), l=x.size(1))


class LinearInterpolation(BaseEmbTransform):
    """
    Linearly resamples the time axis to a fixed number of frames.
    """
    def __init__(self, target_frames: int, align_corners: bool = False):
        super().__init__()
        self.target_frames = target_frames
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer-stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Time-resampled tensor of shape
                (batch_size, num_layers, target_frames, hidden_size).
        """
        b, l, t, h = x.shape
        # Treat hidden_size as channels for 1D interpolation over time
        x2 = rearrange(x, 'b l t h -> (b l) h t')  # (B*L, H, T)
        y = F.interpolate(x2, size=self.target_frames, mode='linear',
                          align_corners=self.align_corners)
        return rearrange(y, '(b l) h t -> b l t h', b=b, l=l)


class TimeAvgPool(BaseEmbTransform):
    """
    Computes simple average pooling over the time dimension.
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Time‐averaged tensor of shape
                (batch_size, num_layers, 1, hidden_size).
        """
        return reduce(x, 'b l t h -> b l 1 h', 'mean')


class TimeMaxPool(BaseEmbTransform):
    """
    Computes max pooling over the time dimension.
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Time‐max-pooled tensor of shape
                (batch_size, num_layers, 1, hidden_size).
        """
        return reduce(x, 'b l t h -> b l 1 h', 'max')


class TimeAttentionPool(BaseEmbTransform):
    """
    Attention pooling over time: Linear(in_dim, 1) scores, softmax, weighted sum.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.attention = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Attention-pooled tensor of shape
                (batch_size, num_layers, 1, hidden_size).
        """
        b, l, t, h = x.shape
        flat = rearrange(x, 'b l t h -> (b l) t h')
        scores = self.attention(flat)
        weights = F.softmax(scores, dim=1)
        out = (weights * flat).sum(dim=1)
        return rearrange(out, '(b l) h -> b l 1 h', b=b, l=l)


class TimeQueryAttentionPool(BaseEmbTransform):
    """
    Query-based attention over time: learned query(ies), multi-head cross-attention,
    then mean over queries.
    """
    def __init__(
        self,
        in_dim: int,
        num_queries: int = 1,
        num_heads: int = 8,
        use_batchnorm: bool = True,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if in_dim % num_heads != 0:
            raise ValueError(f"in_dim ({in_dim}) must be divisible by num_heads ({num_heads})")
        self.in_dim = in_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, in_dim) * 0.02)
        self.k = nn.Linear(in_dim, in_dim, bias=qkv_bias)
        self.v = nn.Linear(in_dim, in_dim, bias=qkv_bias)
        self.use_batchnorm = use_batchnorm
        self.bn = (
            nn.BatchNorm1d(in_dim, affine=False, eps=1e-6)
            if use_batchnorm
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Query-attention-pooled tensor of shape
                (batch_size, num_layers, 1, hidden_size).
        """
        b, l, t, h = x.shape
        flat = rearrange(x, 'b l t h -> (b l) t h')
        n = flat.shape[0]
        if self.use_batchnorm:
            flat = self.bn(flat.permute(0, 2, 1)).permute(0, 2, 1)
        cls_token = self.cls_token.expand(n, -1, -1)
        q = cls_token.reshape(
            n, self.num_queries, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k = (
            self.k(flat)
            .reshape(n, t, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(flat)
            .reshape(n, t, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(n, self.num_queries, self.in_dim)
        x_cls = x_cls.mean(dim=1)
        return rearrange(x_cls, '(b l) h -> b l 1 h', b=b, l=l)


class TimeEfficientProbing(BaseEmbTransform):
    """
    Efficient Probing (Psomas et al., ICLR 2026).

    Parameter-efficient multi-query cross-attention that eliminates the
    key projection. M learnable queries q_j attend directly to the raw
    input features (Eq. 10: â_j = X^T q_j), and a single value projection
    W_V is split into M per-query slices so each query extracts a different
    feature subspace.

    Key differences from TimeQueryAttentionPool (≈ MHCA):
      - No K projection (K = X, identity)
      - Per-query V slicing (each query gets out_dim/M dimensions)
      - Single-head attention (num_heads=1)

    Output shape: (B, L, 1, in_dim // d_out).
    With d_out=1 (default), output dim = in_dim (same as TimeAvgPool).
    """
    def __init__(
        self,
        in_dim: int,
        num_queries: int = 32,
        d_out: int = 1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_queries = num_queries
        self.d_out = d_out
        self.out_dim = in_dim // d_out
        self.scale = in_dim ** -0.5

        if self.out_dim % num_queries != 0:
            raise ValueError(
                f"out_dim ({self.out_dim} = in_dim//d_out = {in_dim}//{d_out}) "
                f"must be divisible by num_queries ({num_queries})"
            )

        # M learnable queries in the input feature space — no Q or K projection
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, in_dim) * 0.02)
        # Only W_V is learned; split into M per-query slices in forward()
        self.v = nn.Linear(in_dim, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, l, t, h = x.shape
        flat = rearrange(x, 'b l t h -> (b l) t h')
        n = flat.shape[0]
        out_dim = self.out_dim
        m = self.num_queries

        cls_token = self.cls_token.expand(n, -1, -1)  # (N, M, H)

        # Attention logits: â_j = (q_j * scale)^T x  — no K projection
        attn = torch.bmm(cls_token * self.scale, flat.transpose(1, 2))  # (N, M, T)
        attn = attn.softmax(dim=-1)

        # Per-query V slicing: W_V x reshaped so each query gets out_dim/M dims
        v = self.v(flat)  # (N, T, out_dim)
        v = v.reshape(n, t, m, out_dim // m).permute(0, 2, 1, 3)  # (N, M, T, d)

        # Weighted aggregation
        out = torch.matmul(attn.unsqueeze(2), v)  # (N, M, 1, d)
        out = out.reshape(n, out_dim)  # (N, out_dim) — concatenated query outputs

        return rearrange(out, '(b l) h -> b l 1 h', b=b, l=l)


class LayerCrossAttention(BaseEmbTransform):
    """
    Attentive Multi-Layer Fusion (Ciernik et al., 2026).

    A learnable query token attends to temporally-pooled layer
    representations via multi-head cross-attention, learning task-adaptive
    layer weighting.

    The original paper uses both CLS and AP (avg-pooled) tokens per layer
    (2|L| tokens). Since audio models lack CLS tokens, we use only the
    temporally-pooled representation per layer (|L| tokens).

    Supports both sequence-level and frame-level input:
    - Sequence-level: (B, L, 1, H) → (B, 1, 1, H)
    - Frame-level:    (B, L, T, H) → (B, 1, T, H)  (attention applied per frame)

    Note: uses doubled head dimension (head_dim = 2 * dim // num_heads)
    following Ciernik's ablation study; separate LayerNorm on Q, K, V
    following their AttentiveBlock design.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        double_head_dim: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_batchnorm: bool = True,
        input_noise_std: float = 0.0,
        input_noise_prob: float = 0.0,
        l2_normalize: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.input_noise_std = input_noise_std
        self.input_noise_prob = input_noise_prob
        self.l2_normalize = l2_normalize

        head_dim = (hidden_dim // num_heads) * (2 if double_head_dim else 1)
        all_head_dim = head_dim * num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Separate LayerNorm for Q, K, V (AttentiveBlock design)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)

        # Full Q/K/V projections
        self.q_proj = nn.Linear(hidden_dim, all_head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, all_head_dim, bias=False)
        self.out_proj = nn.Linear(all_head_dim, hidden_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # BatchNorm on attention output (Ciernik's AttentiveProbeModel, affine=True from their ablation)
        self.bn = (
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True)
            if use_batchnorm
            else nn.Identity()
        )

        # Learnable query token
        self.query_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query_token, std=0.02)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, num_layers, t, h = x.shape
        frame_level = t > 1

        if frame_level:
            # Frame-level (beat tracking): (B, L, T, H) → (B*T, L, H)
            # Apply cross-attention independently per frame
            x_layers = x.permute(0, 2, 1, 3).reshape(b * t, num_layers, h)
        else:
            # Sequence-level: (B, L, 1, H) → (B, L, H)
            x_layers = x.squeeze(2)

        n = x_layers.shape[0]  # B or B*T

        # L2 normalize input features (Ciernik Appendix A.1)
        if self.l2_normalize:
            x_layers = torch.nn.functional.normalize(x_layers, p=2, dim=-1)

        # Gaussian noise on input (Ciernik: N(0, 0.05) with p=0.5, training only)
        if self.training and self.input_noise_std > 0 and self.input_noise_prob > 0:
            if torch.rand(1).item() < self.input_noise_prob:
                x_layers = x_layers + torch.randn_like(x_layers) * self.input_noise_std

        query = self.query_token.expand(n, -1, -1)  # (N, 1, H)

        # LayerNorm (Ciernik: norm_q(q+pos) with pos=0, norm_k(kv+pos), norm_v(kv))
        q = self.norm_q(query)
        k = self.norm_k(x_layers)
        v = self.norm_v(x_layers)

        # Project Q/K/V
        q = self.q_proj(q).reshape(n, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(n, num_layers, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(n, num_layers, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention over layers
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N, heads, 1, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(n, 1, -1)  # (N, 1, all_head_dim)
        out = self.out_proj(out)  # (N, 1, H)
        out = self.proj_drop(out)

        # BatchNorm on attention output (Ciernik: bn(query_tokens[:, 0, :]))
        out = self.bn(out[:, 0, :]).unsqueeze(1)  # (N, H) → BN → (N, 1, H)

        if frame_level:
            # Reshape back: (B*T, 1, H) → (B, 1, T, H)
            out = out.reshape(b, t, h).unsqueeze(1)
        else:
            out = out.unsqueeze(2)  # (B, 1, 1, H)

        return out


class TimeStridedPool(BaseEmbTransform):
    """
    Pools only every n-th frame (default: even frames), then averages.
    Avoids zigzag cancellation by skipping alternating frames.
    Output shape: (B, L, 1, H) — same as TimeAvgPool.
    """
    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return reduce(x[:, :, ::self.stride, :], 'b l t h -> b l 1 h', 'mean')


class TimeVelocityPool(BaseEmbTransform):
    """
    Pools velocity vectors (frame-to-frame differences) instead of raw frames.
    Captures directional patterns that mean pooling of raw frames destroys.
    Output shape: (B, L, 1, H) — same as TimeAvgPool.
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        velocities = x[:, :, 1:, :] - x[:, :, :-1, :]
        return reduce(velocities, 'b l t h -> b l 1 h', 'mean')


class TimeEvenOddConcatPool(BaseEmbTransform):
    """
    Averages even and odd frames separately, concatenates along hidden dim.
    Captures both DC (mean trajectory) and AC (oscillation phase).
    Output shape: (B, L, 1, 2*H) — decoder in_dim must be doubled.
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        even_mean = reduce(x[:, :, ::2, :], 'b l t h -> b l 1 h', 'mean')
        odd_mean = reduce(x[:, :, 1::2, :], 'b l t h -> b l 1 h', 'mean')
        return torch.cat([even_mean, odd_mean], dim=-1)


class TimeRectifiedPool(BaseEmbTransform):
    """
    Demodulates the zigzag: subtracts running mean, takes abs, then pools.
    Like AM demodulation — recovers the oscillation envelope.
    Output shape: (B, L, 1, H) — same as TimeAvgPool.
    """
    def __init__(self, window: int = 5):
        super().__init__()
        self.window = window

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, l, t, h = x.shape
        # Running mean via avg_pool1d
        flat = rearrange(x, 'b l t h -> (b l) h t')
        pad = self.window // 2
        running_mean = F.avg_pool1d(
            F.pad(flat, (pad, pad), mode='replicate'),
            kernel_size=self.window, stride=1
        )
        # Truncate to match original length (avg_pool1d output may differ by 1)
        running_mean = running_mean[:, :, :t]
        residual = flat[:, :, :t] - running_mean
        rectified = residual.abs()
        rectified = rearrange(rectified, '(b l) h t -> b l t h', b=b, l=l)
        return reduce(rectified, 'b l t h -> b l 1 h', 'mean')


class TimeLastNConcat(BaseEmbTransform):
    """
    Concatenates the last N frames over the time dimension.
    """
    def __init__(self, n_frames: int = 1):
        super().__init__()
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        self.n_frames = n_frames

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Concatenated last N frames of shape
                (batch_size, num_layers, 1, n_frames * hidden_size).
        """
        seq_len = x.shape[2]
        if seq_len < self.n_frames:
            last_n = x
            padding = torch.zeros(
                x.shape[0], x.shape[1], self.n_frames - seq_len, x.shape[3],
                device=x.device, dtype=x.dtype
            )
            last_n = torch.cat([padding, last_n], dim=2)
        else:
            last_n = x[:, :, -self.n_frames:, :]
        
        return rearrange(last_n, 'b l n h -> b l 1 (n h)')


class TimeInterpolation(BaseEmbTransform):
    """
    Interpolates the time dimension to a new fixed length.
    """
    def __init__(self, target_frames: int, mode: str = "linear", align_corners: bool = False):
        super().__init__()
        self.target_frames = target_frames
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Interpolated tensor of shape
                (batch_size, num_layers, target_frames, hidden_size).
        """
        x2 = rearrange(x, 'b l t h -> (b l) h t')
        y = F.interpolate(
            x2,
            size=self.target_frames,
            mode=self.mode,
            align_corners=self.align_corners if self.mode in ("linear", "bilinear", "trilinear") else None
        )
        return rearrange(y, '(b l) h t -> b l t h', b=x.size(0), l=x.size(1))

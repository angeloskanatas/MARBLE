# marble/tasks/ExtractRepresentations/extract.py
from pathlib import Path
from typing import Dict, Tuple, Optional
from functools import partial
import hashlib
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config
from marble.modules.transforms import (
    TimeAvgPool, TimeMaxPool, TimeLastNConcat, AudioTransformDataset,
    _apply_audiomentations_to_waveform, MAX_AUDIO_CHANNELS
)

try:
    import audiomentations
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False

def _compute_augmentation_seed(
    augmentation_seed: Optional[int],
    audio_path: str,
    dataset_idx: Optional[int],
    aug_idx: int
) -> Optional[int]:
    """Compute deterministic seed for augmentation.
    
    Args:
        augmentation_seed: Base seed from config
        audio_path: Audio file path
        dataset_idx: Original dataset index
        aug_idx: Augmentation index
    
    Returns:
        Computed seed or None if augmentation_seed is None
    """
    if augmentation_seed is None:
        return None
    
    path_hash = int(hashlib.md5(str(audio_path).encode()).hexdigest()[:8], 16)
    seed = augmentation_seed + path_hash
    if dataset_idx is not None:
        seed += dataset_idx
    seed += aug_idx
    return seed


class ExtractRepresentationsTask(BaseTask):
    """Extracts frame-level and sequence-level embeddings.
    
    Saves:
    - Frame-level: per-file .npy (shape: T, H)
    - Sequence-level: single .dat memmap per layer (shape: N, H)
    """

    def __init__(
        self,
        sample_rate: int,
        encoder: dict,
        emb_transforms: list[dict],
        extraction: dict,
        use_ema: bool = False,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            encoder: Encoder config dict
            emb_transforms: List of embedding transform configs
            extraction: Extraction config dict with keys:
                - split: "train", "val", or "test"
                - output_dir: Directory to save embeddings
                - max_samples: Optional max number of samples to extract (none = all)
                - subset_fraction: Optional fraction of dataset to use (0.0-1.0)
                - sampling_seed: Random seed for deterministic sample selection (default: 17)
                - save_frame_level: Whether to save frame-level embeddings (default: True)
                - save_sequence_level: Whether to save sequence-level embeddings (default: True)
                - memmap_flush_interval: Flush memmap files every N batches (default: 100)
            use_ema: Whether to use EMA (not used for extraction, kept for compatibility)
        """
        enc = instantiate_from_config(encoder)
        all_transforms = [instantiate_from_config(cfg) for cfg in emb_transforms]
        
        pooling_classes = (TimeAvgPool, TimeMaxPool, TimeLastNConcat)
        pooling_idx = None
        for i, transform in enumerate(all_transforms):
            if isinstance(transform, pooling_classes):
                pooling_idx = i
                break
        
        if pooling_idx is not None:
            frame_transforms = all_transforms[:pooling_idx]
            sequence_transforms = all_transforms[pooling_idx:]
        else:
            frame_transforms = all_transforms
            sequence_transforms = []
        
        super().__init__(
            encoder=enc,
            emb_transforms=frame_transforms,
            decoders=None,
            losses=None,
            metrics=None,
            sample_rate=sample_rate,
            use_ema=use_ema,
        )
        
        self.sample_rate = sample_rate
        self.sequence_transforms = nn.ModuleList(sequence_transforms)
        
        self.split = extraction.get('split', 'test')
        if self.split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{self.split}'")
        
        self.output_dir = Path(extraction.get('output_dir', 'output/extracted_embeddings'))
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.output_dir / '.write_test'
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot write to output_dir '{self.output_dir}': {e}")
        
        max_samples_raw = extraction.get('max_samples')
        if max_samples_raw is not None:
            self.max_samples = int(max_samples_raw)
            if self.max_samples <= 0:
                raise ValueError(f"max_samples must be positive, got {self.max_samples}")
        else:
            self.max_samples = None
        
        subset_fraction_raw = extraction.get('subset_fraction')
        if subset_fraction_raw is not None:
            self.subset_fraction = float(subset_fraction_raw)
            if not (0.0 < self.subset_fraction <= 1.0):
                raise ValueError(f"subset_fraction must be in (0.0, 1.0], got {self.subset_fraction}")
        else:
            self.subset_fraction = None
        
        if self.max_samples is not None and self.subset_fraction is not None:
            raise ValueError("Cannot specify both max_samples and subset_fraction")
        
        self.sampling_seed = extraction.get('sampling_seed', 17)
        
        self.save_frame_level = extraction.get('save_frame_level', True)
        self.save_sequence_level = extraction.get('save_sequence_level', True)
        
        if not self.save_frame_level and not self.save_sequence_level:
            raise ValueError("At least one of save_frame_level or save_sequence_level must be True")
        
        augmentation_config = extraction.get('augmentation', None)
        if augmentation_config and AUDIOMENTATIONS_AVAILABLE:
            self.num_augmentations = augmentation_config.get('num_augmentations', 2)
            self.augmentation_seed = augmentation_config.get('seed', None)
            self.augmentation = self._build_augmentation_pipeline(
                augmentation_config, sample_rate
            )
            self._augmentation_config = augmentation_config
            self._use_transform_augmentation = True
            seed_msg = f" (seed={self.augmentation_seed})" if self.augmentation_seed is not None else ""
            print(f"Augmentations: {self.num_augmentations} per sample{seed_msg}")
        else:
            self.num_augmentations = 0
            self.augmentation = None
            self.augmentation_seed = None
            self._augmentation_config = None
            self._use_transform_augmentation = False
            if augmentation_config and not AUDIOMENTATIONS_AVAILABLE:
                print("Warning: Augmentation config provided but audiomentations not installed.")
        
        self._num_samples_processed = 0
        self._layer_dirs_cache = {}
        self._sequence_memmaps = {}
        self._sequence_sample_idx = {}
        self._memmap_flush_interval = extraction.get('memmap_flush_interval', 100)
        self._sample_to_audio_path = []

    def _build_augmentation_pipeline(self, config: dict, sample_rate: int) -> Optional['audiomentations.Compose']:
        """Build audiomentations pipeline from config.
        
        Args:
            config: Augmentation config dict with 'transforms' key
            sample_rate: Audio sample rate
        
        Returns:
            Compose object with augmentation transforms, or None if unavailable
        """
        if not AUDIOMENTATIONS_AVAILABLE:
            return None
        
        from audiomentations import Compose
        import inspect
        
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

    def _extract_hidden_states(self, encoder_output) -> Tuple[torch.Tensor, ...]:
        """Extract hidden_states tuple from encoder output (handles dict, tuple, tensor)."""
        if hasattr(encoder_output, 'hidden_states'):
            return encoder_output.hidden_states
        elif isinstance(encoder_output, (tuple, list)):
            return tuple(encoder_output) if isinstance(encoder_output, list) else encoder_output
        else:
            return (encoder_output,)

    def _is_already_pooled(self, hidden_states: Tuple[torch.Tensor, ...]) -> bool:
        """Check if hidden_states are pooled (no time dimension)."""
        if not hidden_states or not isinstance(hidden_states[0], torch.Tensor):
            return False
        first_hs = hidden_states[0]
        return first_hs.ndim == 2 or (first_hs.ndim == 3 and first_hs.shape[1] == 1)

    def _get_frame_level_embeddings(self, encoder_output) -> Dict[int, torch.Tensor]:
        """Extract frame-level embeddings (B, T, H) per layer. Returns empty dict if already pooled."""
        hidden_states = self._extract_hidden_states(encoder_output)
        
        if self._is_already_pooled(hidden_states):
            return {}
        
        hidden_states_transformed = hidden_states
        for transform in self.emb_transforms:
            hidden_states_transformed = transform(hidden_states_transformed)
        
        frame_embs = {}
        if isinstance(hidden_states_transformed, torch.Tensor):
            if hidden_states_transformed.ndim == 4:
                num_layers = hidden_states_transformed.shape[1]
                for layer_idx in range(num_layers):
                    frame_embs[layer_idx] = hidden_states_transformed[:, layer_idx, :, :]
            elif hidden_states_transformed.ndim == 3:
                frame_embs[0] = hidden_states_transformed
            elif hidden_states_transformed.ndim == 2:
                return {}
        elif isinstance(hidden_states_transformed, (tuple, list)):
            for layer_idx, layer_hidden_state in enumerate(hidden_states_transformed):
                if isinstance(layer_hidden_state, torch.Tensor) and layer_hidden_state.ndim == 3:
                    frame_embs[layer_idx] = layer_hidden_state
        
        return frame_embs

    def _pool_frame_embeddings(self, frame_embs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Pool frame-level (B, T, H) to sequence-level (B, H) embeddings per layer."""
        if not self.sequence_transforms:
            return {
                layer_idx: frame_emb.mean(dim=1) if frame_emb.ndim == 3 else frame_emb
                for layer_idx, frame_emb in frame_embs.items()
            }
        
        layer_indices = sorted(frame_embs.keys())
        if not layer_indices:
            return {}
        
        if len(layer_indices) == 1:
            layer_idx = layer_indices[0]
            frame_emb = frame_embs[layer_idx]
            hidden_with_layer_dim = frame_emb.unsqueeze(1)
            for transform in self.sequence_transforms:
                hidden_with_layer_dim = transform(hidden_with_layer_dim)
            
            if hidden_with_layer_dim.ndim == 4 and hidden_with_layer_dim.shape[1] == 1:
                return {layer_idx: hidden_with_layer_dim[:, 0, 0, :]}
            elif hidden_with_layer_dim.ndim == 3 and hidden_with_layer_dim.shape[1] == 1:
                return {layer_idx: hidden_with_layer_dim[:, 0, :]}
            elif hidden_with_layer_dim.ndim == 2:
                return {layer_idx: hidden_with_layer_dim}
            else:
                return {layer_idx: frame_emb.mean(dim=1)}
        
        stacked_layers = torch.stack([frame_embs[idx] for idx in layer_indices], dim=1)
        pooled_hidden = stacked_layers
        for transform in self.sequence_transforms:
            pooled_hidden = transform(pooled_hidden)
        
        sequence_embs = {}
        if isinstance(pooled_hidden, torch.Tensor):
            if pooled_hidden.ndim == 4:
                for i, layer_idx in enumerate(layer_indices):
                    sequence_embs[layer_idx] = pooled_hidden[:, i, 0, :]
            elif pooled_hidden.ndim == 3:
                if pooled_hidden.shape[1] == 1:
                    sequence_embs[layer_indices[0]] = pooled_hidden[:, 0, :]
                else:
                    for i, layer_idx in enumerate(layer_indices):
                        sequence_embs[layer_idx] = pooled_hidden[:, i, :]
            elif pooled_hidden.ndim == 2:
                sequence_embs[layer_indices[0]] = pooled_hidden
        else:
            for layer_idx, frame_emb in frame_embs.items():
                sequence_embs[layer_idx] = frame_emb.mean(dim=1)
        
        return sequence_embs

    def _get_sequence_level_embeddings(self, encoder_output) -> Dict[int, torch.Tensor]:
        """Extract sequence-level embeddings (B, H) per layer. Fallback when frame_embs unavailable."""
        hidden_states = self._extract_hidden_states(encoder_output)
        
        if self._is_already_pooled(hidden_states):
            sequence_embs = {}
            for layer_idx, hs in enumerate(hidden_states):
                if hs.ndim == 2:  # (B, H)
                    sequence_embs[layer_idx] = hs
                elif hs.ndim == 3 and hs.shape[1] == 1:  # (B, 1, H)
                    sequence_embs[layer_idx] = hs[:, 0, :]
            return sequence_embs
        
        hidden_states_transformed = hidden_states
        for transform in self.emb_transforms:
            hidden_states_transformed = transform(hidden_states_transformed)
        for transform in self.sequence_transforms:
            hidden_states_transformed = transform(hidden_states_transformed)
        
        sequence_embs = {}
        if isinstance(hidden_states_transformed, torch.Tensor):
            if hidden_states_transformed.ndim == 4:
                num_layers = hidden_states_transformed.shape[1]
                for layer_idx in range(num_layers):
                    sequence_embs[layer_idx] = hidden_states_transformed[:, layer_idx, 0, :]
            elif hidden_states_transformed.ndim == 3:
                if hidden_states_transformed.shape[1] == 1:
                    sequence_embs[0] = hidden_states_transformed[:, 0, :]
                else:
                    sequence_embs[0] = hidden_states_transformed.mean(dim=1)
            elif hidden_states_transformed.ndim == 2:
                sequence_embs[0] = hidden_states_transformed
        elif isinstance(hidden_states_transformed, (tuple, list)):
            for layer_idx, layer_hidden_state in enumerate(hidden_states_transformed):
                if isinstance(layer_hidden_state, torch.Tensor):
                    if layer_hidden_state.ndim == 3:
                        sequence_embs[layer_idx] = layer_hidden_state[:, 0, :] if layer_hidden_state.shape[1] == 1 else layer_hidden_state.mean(dim=1)
                    elif layer_hidden_state.ndim == 2:
                        sequence_embs[layer_idx] = layer_hidden_state
        
        if not sequence_embs:
            if isinstance(hidden_states_transformed, torch.Tensor):
                shape_info = hidden_states_transformed.shape
            elif isinstance(hidden_states_transformed, (tuple, list)):
                shape_info = [x.shape if isinstance(x, torch.Tensor) else type(x) for x in hidden_states_transformed]
            else:
                shape_info = type(hidden_states_transformed)
            raise ValueError(f"Failed to extract sequence-level embeddings. Unexpected shape after transforms: {shape_info}")
        
        return sequence_embs

    @staticmethod
    def _multiview_collate_audio_static(
        batch: list,
        augmentation: Optional['audiomentations.Compose'] = None,
        num_augmentations: int = 0,
        sample_rate: int = 24000,
        augmentation_seed: Optional[int] = None
    ) -> Tuple:
        """Static collate function that creates multiple augmented versions.
        
        Args:
            batch: List of (waveform, target, audio_path) tuples
            augmentation: audiomentations.Compose object or None
            num_augmentations: Number of augmentations to create
            sample_rate: Audio sample rate
        
        Returns:
            If augmentation enabled: tuple of (original_batch, aug0_batch, aug1_batch, ...)
            If augmentation disabled: single batch tuple (waveforms, targets, audio_paths)
        
        Note: All tensors are kept on CPU to support multiprocessing (num_workers > 0).
        """
        waveforms = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        audio_paths = [item[2] for item in batch]
        dataset_indices = [item[3] for item in batch] if len(batch[0]) > 3 else None
        
        waveforms_cpu = []
        for w in waveforms:
            if isinstance(w, torch.Tensor):
                waveforms_cpu.append(w.cpu() if w.device.type != 'cpu' else w)
            elif isinstance(w, (list, tuple, np.ndarray)):
                waveforms_cpu.append(torch.as_tensor(w))
            else:
                raise ValueError(f"Expected tensor, list, tuple, or numpy array, got {type(w)}")
        if not waveforms_cpu:
            raise ValueError("No valid waveforms found in batch")
        original_batch = (torch.stack(waveforms_cpu), targets, audio_paths)
        
        if augmentation is None or num_augmentations == 0:
            return original_batch
        
        augmented_batches = []
        for aug_idx in range(num_augmentations):
            augmented_waveforms = []
            for i, (waveform, audio_path) in enumerate(zip(waveforms_cpu, audio_paths)):
                try:
                    dataset_idx = dataset_indices[i] if dataset_indices is not None else i
                    seed = _compute_augmentation_seed(augmentation_seed, audio_path, dataset_idx, aug_idx)
                    if seed is not None:
                        np.random.seed(seed)
                        random.seed(seed)
                    
                    waveform_np = waveform.numpy()
                    augmented_np = _apply_audiomentations_to_waveform(
                        waveform_np, augmentation, sample_rate
                    )
                    augmented_waveforms.append(torch.from_numpy(augmented_np))
                except (ValueError, RuntimeError, AttributeError, TypeError) as e:
                    print(f"Warning: Augmentation failed for sample, using original: {e}")
                    augmented_waveforms.append(waveform)
            
            augmented_batch = (torch.stack(augmented_waveforms), targets, audio_paths)
            augmented_batches.append(augmented_batch)
        
        return (original_batch,) + tuple(augmented_batches)

    @staticmethod
    def _multiview_collate_with_feature_extraction(
        batch: list,
        base_dataset: 'Dataset',
        feature_extractors: list,
        augmentation: Optional['audiomentations.Compose'],
        num_augmentations: int,
        sample_rate: int,
        augmentation_seed: Optional[int],
    ) -> Tuple:
        """Collate function that applies augmentations to waveforms before feature extractors.
        
        Args:
            batch: List of (waveform, target, audio_path) tuples from base dataset
            base_dataset: Base dataset (before transforms) - unused but kept for API consistency
            feature_extractors: List of BaseAudioTransform instances (feature extractors only)
            augmentation: audiomentations.Compose object
            num_augmentations: Number of augmented versions to create
            sample_rate: Audio sample rate
            augmentation_seed: Base seed for deterministic augmentation
        
        Returns:
            If augmentation enabled: tuple of (original_batch, aug0_batch, aug1_batch, ...)
            If augmentation disabled: single batch tuple (features, targets, audio_paths)
        
        Note: All tensors are kept on CPU to support multiprocessing (num_workers > 0).
        """
        waveforms = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        audio_paths = [item[2] for item in batch]
        dataset_indices = [item[3] for item in batch] if len(batch[0]) > 3 else None
        
        waveforms_cpu = []
        for w in waveforms:
            if isinstance(w, torch.Tensor):
                waveforms_cpu.append(w.cpu() if w.device.type != 'cpu' else w)
            elif isinstance(w, (list, tuple, np.ndarray)):
                waveforms_cpu.append(torch.as_tensor(w))
            else:
                raise ValueError(f"Expected tensor, list, tuple, or numpy array, got {type(w)}")
        
        if not waveforms_cpu:
            raise ValueError("No valid waveforms found in batch")
        
        def apply_feature_extractors(waveform_batch: list) -> torch.Tensor:
            """Apply feature extractors to a batch of waveforms."""
            batch_features = []
            for waveform in waveform_batch:
                sample = {
                    "input_features": waveform,
                    "sampling_rate": sample_rate
                }
                for transform in feature_extractors:
                    sample = transform(sample)
                feature = sample["input_features"]
                if not isinstance(feature, torch.Tensor):
                    feature = torch.as_tensor(feature)
                if feature.ndim == 0:
                    feature = feature.unsqueeze(0)
                batch_features.append(feature)
            return torch.stack(batch_features)
        
        original_features = apply_feature_extractors(waveforms_cpu)
        original_batch = (original_features, targets, audio_paths)
        
        if augmentation is None or num_augmentations == 0:
            return original_batch
        
        augmented_batches = []
        for aug_idx in range(num_augmentations):
            augmented_waveforms = []
            for i, (waveform, audio_path) in enumerate(zip(waveforms_cpu, audio_paths)):
                try:
                    dataset_idx = dataset_indices[i] if dataset_indices is not None else i
                    seed = _compute_augmentation_seed(augmentation_seed, audio_path, dataset_idx, aug_idx)
                    if seed is not None:
                        np.random.seed(seed)
                        random.seed(seed)
                    
                    waveform_np = waveform.numpy()
                    augmented_np = _apply_audiomentations_to_waveform(
                        waveform_np, augmentation, sample_rate
                    )
                    augmented_waveforms.append(torch.from_numpy(augmented_np))
                except (ValueError, RuntimeError, AttributeError, TypeError) as e:
                    print(f"Warning: Augmentation failed for sample, using original: {e}")
                    augmented_waveforms.append(waveform)
            
            augmented_features = apply_feature_extractors(augmented_waveforms)
            augmented_batches.append((augmented_features, targets, audio_paths))
        
        return (original_batch,) + tuple(augmented_batches)

    def _get_file_id(self, audio_path: str, sample_idx: int, aug_idx: Optional[int] = None) -> str:
        """Generate unique file identifier from audio path and dataset index."""
        base_id = Path(audio_path).stem
        base_id = ''.join(c if (c.isalnum() or c in '._-') else '_' for c in base_id)
        file_id = f"{base_id}_{sample_idx:06d}"
        if aug_idx is not None:
            file_id = f"{file_id}_aug{aug_idx:02d}"
        return file_id
    
    def _get_layer_dir(self, layer_idx: int, emb_type: str) -> Path:
        """Get or create directory for a specific layer and embedding type."""
        cache_key = (layer_idx, emb_type)
        layer_dir = self._layer_dirs_cache.get(cache_key)
        if layer_dir is None:
            layer_dir = self.output_dir / f"layer{layer_idx}" / emb_type
            layer_dir.mkdir(parents=True, exist_ok=True)
            self._layer_dirs_cache[cache_key] = layer_dir
        return layer_dir
    
    def _save_embeddings(
        self,
        embeddings: Dict[int, torch.Tensor],
        num_samples: int,
        audio_paths: list,
        batch_start_count: int,
        emb_type: str,
        aug_idx: Optional[int] = None
    ) -> None:
        """Save embeddings to disk as per-file .npy files.
        
        Args:
            embeddings: Dict mapping layer_idx to tensor of shape (B, ...)
            num_samples: Number of samples to save from batch
            audio_paths: List of audio file paths
            batch_start_count: Starting sample index for file naming
            emb_type: Embedding type directory name (e.g., "frame-level", "sequence-level")
            aug_idx: Optional augmentation index for augmented embeddings
        """
        file_ids = [
            self._get_file_id(audio_paths[i], batch_start_count + i, aug_idx=aug_idx)
            for i in range(num_samples)
        ]
        
        layer_dirs = {}
        for layer_idx in embeddings.keys():
            layer_dirs[layer_idx] = self._get_layer_dir(layer_idx, emb_type)
        
        embeddings_cpu = {
            layer_idx: emb[:num_samples].detach().cpu().numpy()
            for layer_idx, emb in embeddings.items()
        }
        
        for layer_idx, batch_emb in embeddings_cpu.items():
            layer_dir = layer_dirs[layer_idx]
            for sample_idx in range(num_samples):
                try:
                    np.save(layer_dir / f"{file_ids[sample_idx]}.npy", batch_emb[sample_idx])
                except (OSError, IOError) as e:
                    print(f"Error saving {emb_type} embedding (layer {layer_idx}, file_id: {file_ids[sample_idx]}): {e}")
    
    def _save_frame_level_embeddings(
        self,
        frame_embs: Dict[int, torch.Tensor],
        num_samples: int,
        audio_paths: list,
        batch_start_count: int
    ) -> None:
        """Save frame-level embeddings (B, T, H) to disk."""
        self._save_embeddings(
            frame_embs, num_samples, audio_paths, batch_start_count,
            emb_type="frame-level", aug_idx=None
        )
    
    def _init_sequence_memmap(
        self,
        layer_idx: int,
        total_samples: int,
        hidden_dim: int,
        dtype: np.dtype,
        aug_idx: Optional[int] = None
    ) -> None:
        key = (layer_idx, aug_idx)
        if key in self._sequence_memmaps:
            return
        
        emb_type = f"sequence-level_aug{aug_idx}" if aug_idx is not None else "sequence-level"
        layer_dir = self._get_layer_dir(layer_idx, emb_type)
        
        memmap_file = layer_dir / "embeddings.dat"
        memmap = np.memmap(
            memmap_file,
            dtype=dtype,
            mode='w+',
            shape=(total_samples, hidden_dim)
        )
        
        self._sequence_memmaps[key] = memmap
        self._sequence_sample_idx[key] = 0
    
    def _save_sequence_level_embeddings(
        self,
        sequence_embs: Dict[int, torch.Tensor],
        num_samples: int,
        audio_paths: list,
        batch_start_count: int,
        aug_idx: Optional[int] = None
    ) -> None:
        embeddings_cpu = {
            layer_idx: emb[:num_samples].detach().cpu().numpy()
            for layer_idx, emb in sequence_embs.items()
        }
        self._write_sequence_embeddings_to_memmap(embeddings_cpu, num_samples, aug_idx)
    
    def _write_sequence_embeddings_to_memmap(
        self,
        embeddings_cpu: Dict[int, np.ndarray],
        num_samples: int,
        aug_idx: Optional[int] = None
    ) -> None:
        for layer_idx, batch_emb in embeddings_cpu.items():
            key = (layer_idx, aug_idx)
            
            if key not in self._sequence_memmaps:
                total_samples = self._get_total_samples_for_memmap()
                H = batch_emb.shape[1]
                dtype = batch_emb.dtype
                self._init_sequence_memmap(layer_idx, total_samples, H, dtype, aug_idx)
            
            memmap = self._sequence_memmaps[key]
            sample_idx = self._sequence_sample_idx[key]
            
            if sample_idx + num_samples > memmap.shape[0]:
                raise ValueError(
                    f"Memmap bounds exceeded for layer {layer_idx}, aug {aug_idx}: "
                    f"trying to write {num_samples} samples at index {sample_idx}, "
                    f"but memmap only has {memmap.shape[0]} samples"
                )
            
            memmap[sample_idx:sample_idx + num_samples] = batch_emb
            self._sequence_sample_idx[key] = sample_idx + num_samples
    
    def _get_total_samples_for_memmap(self) -> int:
        if hasattr(self, '_total_samples_for_memmap'):
            return self._total_samples_for_memmap
        raise RuntimeError("Total samples not set for memmap initialization")
    
    def _flush_sequence_memmaps(self) -> None:
        for memmap in self._sequence_memmaps.values():
            memmap.flush()
    
    def _close_sequence_memmaps(self) -> None:
        for (layer_idx, aug_idx), memmap in self._sequence_memmaps.items():
            memmap.flush()
            
            actual_samples_written = self._sequence_sample_idx.get((layer_idx, aug_idx), 0)
            
            emb_type = f"sequence-level_aug{aug_idx}" if aug_idx is not None else "sequence-level"
            layer_dir = self._get_layer_dir(layer_idx, emb_type)
            metadata_file = layer_dir / "metadata.json"
            memmap_file = layer_dir / "embeddings.dat"
            
            actual_shape = (actual_samples_written, memmap.shape[1])
            file_size = memmap_file.stat().st_size
            itemsize = memmap.dtype.itemsize
            
            metadata = {
                'shape': list(actual_shape),
                'dtype': str(memmap.dtype),
                'format': 'memmap',
                'aug_idx': aug_idx,
                'allocated_shape': list(memmap.shape),
                'file_size_bytes': file_size,
                'itemsize_bytes': itemsize
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self._sequence_memmaps.clear()
        self._sequence_sample_idx.clear()
        
        if self._sample_to_audio_path:
            mapping_file = self.output_dir / "sample_to_audio_path.json"
            with open(mapping_file, 'w') as f:
                json.dump({
                    'sample_to_audio_path': self._sample_to_audio_path,
                    'num_samples': len(self._sample_to_audio_path)
                }, f, indent=2)
            self._sample_to_audio_path.clear()
    
    def test_step(self, batch, batch_idx: int) -> dict:
        return {}
    
    @rank_zero_only
    def on_test_start(self) -> None:
        self._num_samples_processed = 0
        
        if self.trainer is None or self.trainer.datamodule is None:
            raise RuntimeError("trainer.datamodule is required for extraction")
        datamodule = self.trainer.datamodule
        
        if self.split == 'train':
            dataset = datamodule.train_dataset
        elif self.split == 'val':
            dataset = datamodule.val_dataset
        else:
            dataset = datamodule.test_dataset
        
        base_dataset = dataset
        feature_extractors = []
        
        if isinstance(dataset, AudioTransformDataset):
            base_dataset = dataset.base
            split = self.split
            transform_configs = datamodule.audio_transforms.get(split, [])
            for cfg in transform_configs:
                try:
                    transform = instantiate_from_config(cfg)
                    from marble.core.base_transform import BaseAudioTransform
                    if isinstance(transform, BaseAudioTransform):
                        class_name = transform.__class__.__name__
                        if 'FeatureExtractor' in class_name:
                            feature_extractors.append(transform)
                except Exception:
                    class_path = cfg.get('class_path', '')
                    if 'FeatureExtractor' in class_path:
                        feature_extractors.append(instantiate_from_config(cfg))
        
        use_base_dataset_for_dataloader = False
        
        if self._use_transform_augmentation:
            if self.num_augmentations >= 1:
                if not feature_extractors:
                    collate_fn = partial(
                        self._multiview_collate_audio_static,
                        augmentation=self.augmentation,
                        num_augmentations=self.num_augmentations,
                        sample_rate=self.sample_rate,
                        augmentation_seed=self.augmentation_seed
                    )
                else:
                    use_base_dataset_for_dataloader = True
                    collate_fn = partial(
                        self._multiview_collate_with_feature_extraction,
                        base_dataset=base_dataset,
                        feature_extractors=feature_extractors,
                        augmentation=self.augmentation,
                        num_augmentations=self.num_augmentations,
                        sample_rate=self.sample_rate,
                        augmentation_seed=self.augmentation_seed,
                    )
            else:
                collate_fn = None
        else:
            collate_fn = partial(
                self._multiview_collate_audio_static,
                augmentation=self.augmentation,
                num_augmentations=self.num_augmentations,
                sample_rate=self.sample_rate,
                augmentation_seed=self.augmentation_seed
            )
        
        dataloader_dataset = base_dataset if use_base_dataset_for_dataloader else dataset
        original_total_samples = len(dataloader_dataset)
        total_samples = original_total_samples
        subset_indices = None
        
        if self.max_samples is not None and self.max_samples < original_total_samples:
            rng = random.Random(self.sampling_seed)
            subset_indices = sorted(rng.sample(range(original_total_samples), self.max_samples))
            total_samples = self.max_samples
        elif self.subset_fraction is not None:
            subset_size = int(original_total_samples * self.subset_fraction)
            if subset_size == 0:
                raise ValueError(
                    f"subset_fraction {self.subset_fraction} results in 0 samples "
                    f"from {original_total_samples} total"
                )
            rng = random.Random(self.sampling_seed)
            subset_indices = sorted(rng.sample(range(original_total_samples), subset_size))
            total_samples = subset_size
        
        if subset_indices is not None:
            dataloader_dataset = Subset(dataloader_dataset, subset_indices)
            dataset = Subset(dataset, subset_indices)
        
        loader_kwargs = {
            "batch_size": datamodule.batch_size,
            "shuffle": False,
            "num_workers": datamodule.num_workers,
            "pin_memory": True,
        }
        if collate_fn is not None:
            loader_kwargs["collate_fn"] = collate_fn
        if datamodule.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
            loader_kwargs["persistent_workers"] = True
        dataloader = DataLoader(dataloader_dataset, **loader_kwargs)
        
        batch_size = getattr(dataloader, 'batch_size', None)
        if batch_size is None:
            raise ValueError("Cannot extract from IterableDataset - batch_size is None. Use a regular Dataset.")
        if batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size}")
        expected_batches = (total_samples + batch_size - 1) // batch_size
        
        print(f"\nExtracting from {self.split} split")
        if self.max_samples is not None:
            print(f"Dataset: {original_total_samples:,} samples, extracting {total_samples:,} (max_samples={self.max_samples:,})")
        elif self.subset_fraction is not None:
            print(f"Dataset: {original_total_samples:,} samples, extracting {total_samples:,} ({self.subset_fraction:.2%})")
        else:
            print(f"Dataset: {total_samples:,} samples")
        print(f"Batch size: {batch_size}, Workers: {datamodule.num_workers}, Expected batches: {expected_batches:,}")
        print(f"Output: {self.output_dir}")
        saving_types = []
        if self.save_frame_level:
            saving_types.append("frame-level")
        if self.save_sequence_level:
            saving_types.append("sequence-level")
        print(f"Saving: {', '.join(saving_types)}")
        if self.num_augmentations > 0:
            print(f"Augmentations: {self.num_augmentations} per sample")
        print()
        
        self.eval()
        self._layer_dirs_cache.clear()
        self._sequence_memmaps.clear()
        self._sequence_sample_idx.clear()
        self._total_samples_for_memmap = total_samples
        warned_frame_level = False
        
        # TODO: For DDP/multi-GPU extraction, use per-rank files (embeddings_rank{rank}.dat)
        # and merge after extraction?
        
        with torch.no_grad():
            sample_count = 0
            samples_saved = 0
            batch_count = 0
            pbar = tqdm(total=total_samples, desc="Extracting embeddings", unit="sample")
            
            try:
                for batch in dataloader:
                    try:
                        if isinstance(batch, list):
                            batch = tuple(batch)
                        
                        if not isinstance(batch, tuple) or len(batch) < 2:
                            raise ValueError(f"Expected batch tuple with at least 2 elements, got {type(batch)} with length {len(batch)}")
                        
                        is_augmented = (
                            self.num_augmentations > 0
                            and len(batch) > 1
                            and isinstance(batch[0], (tuple, list))
                            and len(batch[0]) >= 2
                        )
                        
                        if is_augmented:
                            batches_to_process = tuple(
                                tuple(item) if isinstance(item, list) else item
                                for item in batch
                            )
                            aug_indices = [None] + list(range(self.num_augmentations))
                        else:
                            batches_to_process = (batch,)
                            aug_indices = [None]
                        
                        first_batch_item = batches_to_process[0]
                        if not isinstance(first_batch_item, tuple) or len(first_batch_item) < 2:
                            raise ValueError(f"Expected batch item tuple with at least 2 elements, got {type(first_batch_item)} with length {len(first_batch_item)}")
                        
                        first_waveform = first_batch_item[0]
                        if not isinstance(first_waveform, torch.Tensor):
                            if isinstance(first_waveform, (list, tuple, np.ndarray)):
                                first_waveform = torch.as_tensor(first_waveform)
                            else:
                                raise ValueError(f"Expected tensor, got {type(first_waveform)}")
                        
                        if first_waveform.ndim < 1:
                            raise ValueError(f"Expected tensor with at least 1 dimension, got {first_waveform.ndim} dimensions")
                        
                        batch_size_actual = first_waveform.shape[0]
                        if batch_size_actual == 0:
                            raise ValueError("Batch size is 0")
                        
                        samples_to_take = batch_size_actual
                        frame_embs = None
                        base_audio_paths = None
                        views_succeeded = 0
                        pending_embeddings = {}
                        
                        for batch_item, aug_idx in zip(batches_to_process, aug_indices):
                            waveform = batch_item[0]
                            if not isinstance(waveform, torch.Tensor):
                                waveform = torch.as_tensor(waveform)
                            
                            audio_paths = batch_item[2] if len(batch_item) > 2 else None
                            
                            if audio_paths is None:
                                raise ValueError("Batch must contain audio paths (batch[2]) for per-file saving")
                            
                            if waveform.device.type != self.device.type:
                                waveform = waveform.to(self.device)
                            
                            if not isinstance(audio_paths, (list, tuple)):
                                audio_paths = [str(audio_paths)] * batch_size_actual
                            elif len(audio_paths) != batch_size_actual:
                                raise ValueError(f"audio_paths length ({len(audio_paths)}) doesn't match batch_size ({batch_size_actual})")
                            
                            if aug_idx is None:
                                base_audio_paths = audio_paths
                                if self.save_sequence_level:
                                    self._sample_to_audio_path.extend(base_audio_paths)
                            
                            try:
                                encoder_output = self.encoder(waveform)
                            except RuntimeError as e:
                                error_msg = str(e).lower()
                                is_oom = (
                                    'out of memory' in error_msg or 
                                    'cuda' in error_msg or 
                                    'mps' in error_msg
                                )
                                if is_oom:
                                    print(f"OOM error in batch {batch_count + 1}, aug {aug_idx}: {e}. Skipping entire batch.")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                    torch.mps.empty_cache()
                                else:
                                    print(f"Error in batch {batch_count + 1}, aug {aug_idx}: {e}. Skipping entire batch.")
                                break
                            
                            views_succeeded += 1
                            
                            if aug_idx is None and self.save_frame_level:
                                frame_embs = self._get_frame_level_embeddings(encoder_output)
                                if not frame_embs and not warned_frame_level:
                                    print("Warning: Model returns already-pooled embeddings. Skipping frame-level extraction.")
                                    warned_frame_level = True
                            
                            if self.save_sequence_level:
                                if aug_idx is None and frame_embs is not None:
                                    sequence_embs = self._pool_frame_embeddings(frame_embs)
                                else:
                                    sequence_embs = self._get_sequence_level_embeddings(encoder_output)
                                
                                if sequence_embs:
                                    pending_embeddings[aug_idx] = {
                                        layer_idx: emb[:samples_to_take].detach().cpu().numpy()
                                        for layer_idx, emb in sequence_embs.items()
                                    }
                            
                            encoder_output = None
                        
                        all_views_ok = (views_succeeded == len(aug_indices))
                        all_seq_ok = (not self.save_sequence_level) or (len(pending_embeddings) == len(aug_indices))
                        
                        if all_views_ok and all_seq_ok:
                            if self.save_frame_level and frame_embs:
                                if base_audio_paths is None:
                                    raise ValueError("base_audio_paths is None but frame-level saving is enabled")
                                self._save_frame_level_embeddings(
                                    frame_embs, samples_to_take, base_audio_paths, sample_count
                                )
                            
                            if self.save_sequence_level:
                                for aug_idx in aug_indices:
                                    if aug_idx in pending_embeddings:
                                        sequence_embs_np = pending_embeddings[aug_idx]
                                        self._write_sequence_embeddings_to_memmap(
                                            sequence_embs_np, samples_to_take, aug_idx
                                        )
                            
                            sample_count += samples_to_take
                            samples_saved += samples_to_take
                            pbar.update(samples_to_take)
                            pbar.set_description(f"Extracting embeddings (batch {batch_count + 1}/{expected_batches}, saved: {samples_saved}/{total_samples})")
                            
                            if (batch_count + 1) % self._memmap_flush_interval == 0:
                                self._flush_sequence_memmaps()
                        elif all_views_ok and not all_seq_ok:
                            print(f"Warning: Batch {batch_count + 1} - all views succeeded but sequence embeddings missing for some views. Skipping batch.")
                        
                        batch_count += 1
                    
                    except (FileNotFoundError, OSError) as e:
                        print(f"Error loading batch {batch_count + 1}/{expected_batches}: {e}. Skipping.")
                        batch_count += 1
                        continue
                    except (ValueError, RuntimeError) as e:
                        print(f"Error processing batch {batch_count + 1}/{expected_batches}: {e}. Skipping.")
                        batch_count += 1
                        continue
            finally:
                pbar.close()
                
                self._flush_sequence_memmaps()
                self._close_sequence_memmaps()
            
            print(f"\nExtraction complete: {samples_saved:,}/{total_samples:,} samples saved")
            if sample_count > samples_saved:
                print(f"Skipped {sample_count - samples_saved:,} samples due to errors")
        
        self._num_samples_processed = samples_saved

    @rank_zero_only
    def on_test_epoch_end(self) -> None:
        print(f"Output: {self.output_dir}")

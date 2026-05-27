# marble/tasks/GTZANGenre/datamodule.py

import json
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule
from marble.tasks.GTZANGenre.embedding_dataset import GTZANGenreEmbeddingDataset




class _GTZANGenreAudioBase(Dataset):
    """
    Base dataset for GTZAN genre audio:
    - Splits each audio file into non-overlapping clips of length `clip_seconds` (last clip zero-padded).
    """
    LABEL2IDX = {
        'blues': 0, 'classical': 1, 'country': 2, 'disco': 3,
        'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7,
        'reggae': 8, 'rock': 9
    }
    EXAMPLE_JSONL = {
        "audio_path": "data/GTZAN/genres/blues/blues.00012.wav",
        "label": "blues",
        "duration": 30.013333333333332,
        "sample_rate": 22050,
        "num_samples": 661794,
        "bit_depth": 16,
        "channels": 1
    }
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        jsonl: str,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.channel_mode = channel_mode
        if channel_mode not in ["first", "mix", "random"]:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.min_clip_ratio = min_clip_ratio

        # 读取元数据
        with open(jsonl, 'r') as f:
            self.meta = [json.loads(line) for line in f]

        # Build index map: (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
        self.index_map: List[Tuple[int, int, int, int, int]] = []
        self.resamplers = {}
        for file_idx, info in enumerate(self.meta):
            orig_sr = info['sample_rate']
            # Prepare resampler if needed
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

            orig_clip_frames = int(self.clip_seconds * orig_sr)
            orig_channels = info['channels']
            total_samples = info['num_samples']

            # Number of full clips and remainder
            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            # Decide whether to keep the last shorter clip
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full

            for slice_idx in range(n_slices):
                self.index_map.append(
                    (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
                )


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Load and return one audio clip and its label.

        Inputs:
            idx: int - index of the clip

        Returns:
            waveform: torch.Tensor, shape (self.channels, self.clip_len_target)
            label: int
            path: str
        """
        # Unpack mapping info
        file_idx, slice_idx, orig_sr, orig_clip, orig_channels = self.index_map[idx]
        info = self.meta[file_idx]
        path = info['audio_path']
        label = self.LABEL2IDX[info['label']]

        # Compute frame offset and load clip
        offset = slice_idx * orig_clip
        waveform, _ = torchaudio.load(
            path,
            frame_offset=offset,
            num_frames=orig_clip
        )  # (orig_channels, orig_clip)

        # Channel alignment / downmixing
        if orig_channels >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(dim=0, keepdim=True)
                elif self.channel_mode == "random":
                    # 将 mix 作为一个选项
                    choice = torch.randint(0, orig_channels + 1, (1,)).item()
                    if choice == orig_channels:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        waveform = waveform[choice:choice+1]
                else:
                    raise ValueError(f"Unknown channel_mode: {self.channel_mode}")
            else:
                waveform = waveform[:self.channels]
        else:
            # Repeat last channel to pad to desired channels
            last = waveform[-1:].repeat(self.channels - orig_channels, 1)
            waveform = torch.cat([waveform, last], dim=0)

        # Resample if needed
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # Pad to target length if short
        if waveform.size(1) < self.clip_len_target:
            pad = self.clip_len_target - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))
        
        # Final shape: (self.channels, self.clip_len_target)
        return waveform, label, path


class GTZANGenreAudioTrain(_GTZANGenreAudioBase):
    """
    训练集：DataModule 中设置 shuffle=True。
    """
    pass


class GTZANGenreAudioVal(_GTZANGenreAudioBase):
    """
    验证集：DataModule 中设置 shuffle=False。
    """
    pass


class GTZANGenreAudioTest(GTZANGenreAudioVal):
    """
    测试集：同验证集逻辑。
    """
    pass


class GTZANGenreDataModule(BaseDataModule):
    pass


class GTZANGenreMultiLayerEmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted GTZAN embeddings with MULTIPLE layers.
    Returns batches of shape (B, L, H) instead of (B, H).
    """

    def __init__(
        self,
        embedding_root: str,
        layer_indices: list,
        train_jsonl: str,
        val_jsonl: str,
        test_jsonl: str,
        batch_size: int = 8,
        num_workers: int = 8,
    ):
        super().__init__()
        from marble.tasks.GTZANGenre.embedding_dataset import GTZANGenreMultiLayerEmbeddingDataset
        self._dataset_cls = GTZANGenreMultiLayerEmbeddingDataset

        self.embedding_root = Path(embedding_root)
        self.layer_indices = layer_indices
        self.train_jsonl = train_jsonl
        self.val_jsonl = val_jsonl
        self.test_jsonl = test_jsonl
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_dataset = self._dataset_cls(
                self.embedding_root / "train",
                self.layer_indices,
                self.train_jsonl,
            )
            self.val_dataset = self._dataset_cls(
                self.embedding_root / "val",
                self.layer_indices,
                self.val_jsonl,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = self._dataset_cls(
                self.embedding_root / "test",
                self.layer_indices,
                self.test_jsonl,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )


class GTZANGenreEmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted GTZAN embeddings."""

    def __init__(
        self,
        embedding_root: str,
        layer_idx: int,
        train_jsonl: str,
        val_jsonl: str,
        test_jsonl: str,
        batch_size: int = 8,
        num_workers: int = 8,
        shuffle_labels: bool = False,
        shuffle_seed: int = 42,
    ):
        super().__init__()
        self.embedding_root = Path(embedding_root)
        self.layer_idx = layer_idx
        self.train_jsonl = train_jsonl
        self.val_jsonl = val_jsonl
        self.test_jsonl = test_jsonl
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_labels = shuffle_labels
        self.shuffle_seed = shuffle_seed

    def _shuffle_dataset_labels(self, dataset, rng):
        """Randomly permute the label list, breaking embedding-label correspondence."""
        import numpy as np
        perm = rng.permutation(len(dataset._labels))
        dataset._labels = [dataset._labels[i] for i in perm]

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_dataset = GTZANGenreEmbeddingDataset(
                self.embedding_root / "train",
                self.layer_idx,
                self.train_jsonl,
            )
            self.val_dataset = GTZANGenreEmbeddingDataset(
                self.embedding_root / "val",
                self.layer_idx,
                self.val_jsonl,
            )
            if self.shuffle_labels:
                import numpy as np
                rng = np.random.RandomState(self.shuffle_seed)
                self._shuffle_dataset_labels(self.train_dataset, rng)
                self._shuffle_dataset_labels(self.val_dataset, rng)
        if stage in (None, "test", "predict"):
            self.test_dataset = GTZANGenreEmbeddingDataset(
                self.embedding_root / "test",
                self.layer_idx,
                self.test_jsonl,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

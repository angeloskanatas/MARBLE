# marble/tasks/NSynthI/datamodule.py

import json
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule
from marble.tasks.NSynthI.embedding_dataset import NSynthIEmbeddingDataset, LABEL2IDX


class _NSynthIAudioBase(Dataset):
    """
    Base dataset for NSynth instrument family classification audio.
    Each NSynth sample is a 4-second note at 16kHz (64000 samples).
    """
    LABEL2IDX = LABEL2IDX
    EXAMPLE_JSONL = {
        "audio_path": "data/NSynth/nsynth-train/audio/bass_synthetic_034-025-100.wav",
        "label": "bass",
        "duration": 4.0,
        "sample_rate": 16000,
        "num_samples": 64000,
        "bit_depth": 16,
        "channels": 1,
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

        with open(jsonl, "r") as f:
            self.meta = [json.loads(line) for line in f]

        # Build index map: (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
        self.index_map: List[Tuple[int, int, int, int, int]] = []
        self.resamplers = {}
        for file_idx, info in enumerate(self.meta):
            orig_sr = info["sample_rate"]
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

            orig_clip_frames = int(self.clip_seconds * orig_sr)
            orig_channels = info["channels"]
            total_samples = info["num_samples"]

            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
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
        file_idx, slice_idx, orig_sr, orig_clip, orig_channels = self.index_map[idx]
        info = self.meta[file_idx]
        path = info["audio_path"]
        label = self.LABEL2IDX[info["label"]]

        offset = slice_idx * orig_clip
        waveform, _ = torchaudio.load(path, frame_offset=offset, num_frames=orig_clip)

        # Channel alignment
        if orig_channels >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(dim=0, keepdim=True)
                elif self.channel_mode == "random":
                    choice = torch.randint(0, orig_channels + 1, (1,)).item()
                    if choice == orig_channels:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        waveform = waveform[choice : choice + 1]
            else:
                waveform = waveform[: self.channels]
        else:
            last = waveform[-1:].repeat(self.channels - orig_channels, 1)
            waveform = torch.cat([waveform, last], dim=0)

        # Resample if needed
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # Pad to target length if short
        if waveform.size(1) < self.clip_len_target:
            pad = self.clip_len_target - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))

        return waveform, label, path


class NSynthIAudioTrain(_NSynthIAudioBase):
    pass


class NSynthIAudioVal(_NSynthIAudioBase):
    pass


class NSynthIAudioTest(NSynthIAudioVal):
    pass


class NSynthIDataModule(BaseDataModule):
    pass


class NSynthIMultiLayerEmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted NSynth instrument embeddings with MULTIPLE layers.
    Returns batches of shape (B, L, H) instead of (B, H).
    """

    def __init__(
        self,
        embedding_root: str,
        layer_indices: list,
        train_jsonl: str,
        val_jsonl: str,
        test_jsonl: str,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        from marble.tasks.NSynthI.embedding_dataset import NSynthIMultiLayerEmbeddingDataset
        self._dataset_cls = NSynthIMultiLayerEmbeddingDataset

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


class NSynthIEmbeddingDataModule(pl.LightningDataModule):
    """DataModule for probing on pre-extracted NSynth instrument embeddings."""

    def __init__(
        self,
        embedding_root: str,
        layer_idx: int,
        train_jsonl: str,
        val_jsonl: str,
        test_jsonl: str,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.embedding_root = Path(embedding_root)
        self.layer_idx = layer_idx
        self.train_jsonl = train_jsonl
        self.val_jsonl = val_jsonl
        self.test_jsonl = test_jsonl
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_dataset = NSynthIEmbeddingDataset(
                self.embedding_root / "train",
                self.layer_idx,
                self.train_jsonl,
            )
            self.val_dataset = NSynthIEmbeddingDataset(
                self.embedding_root / "val",
                self.layer_idx,
                self.val_jsonl,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = NSynthIEmbeddingDataset(
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

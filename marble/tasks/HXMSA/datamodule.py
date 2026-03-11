# marble/tasks/HXMSA/datamodule.py

import json
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule
from marble.tasks.HXMSA.embedding_dataset import (
    HXMSAEmbeddingDataset, LABEL2IDX, IDX2LABEL, normalize_label,
)


class _HXMSAAudioBase(Dataset):
    """
    Base dataset for HXMSA (Harmonix Music Structure Analysis).

    Each JSONL entry is one **segment** from a track (intro, verse, chorus, etc.).
    The entry stores `segment_offset_samples` so we load only the relevant portion
    of the full-track audio.  Long segments are split into non-overlapping clips
    of `clip_seconds`; the last clip is zero-padded if shorter.

    Label set (6 classes, after normalization):
        0: intro, 1: verse, 2: chorus, 3: bridge, 4: outro, 5: inst
    """

    LABEL2IDX = LABEL2IDX
    IDX2LABEL = IDX2LABEL
    NUM_CLASSES = len(IDX2LABEL)

    EXAMPLE_JSONL = {
        "audio_path": "data/HXMSA/tracks/0001_12step.mp3",
        "label": "verse",
        "track_id": "0001_12step",
        "segment_start": 8.495568,
        "segment_end": 25.486704,
        "segment_offset_samples": 374655,
        "duration": 16.991136,
        "sample_rate": 44100,
        "num_samples": 749310,
        "bit_depth": 16,
        "channels": 2,
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

        for info in self.meta:
            normed = normalize_label(info["label"])
            if normed not in self.LABEL2IDX:
                raise ValueError(
                    f"Label '{info['label']}' normalized to '{normed}' "
                    f"which is not in LABEL2IDX"
                )

        # Build index map: (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
        self.index_map: List[Tuple[int, int, int, int, int]] = []
        self.resamplers = {}
        for file_idx, info in enumerate(self.meta):
            orig_sr = info["sample_rate"]
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                    orig_sr, self.sample_rate
                )

            orig_clip_frames = int(self.clip_seconds * orig_sr)
            orig_channels = info["channels"]
            total_samples = info["num_samples"]  # segment sample count

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
        track_id = info["track_id"]
        label = self.LABEL2IDX[normalize_label(info["label"])]

        # Offset into the full track: segment base + clip position within segment
        base_offset = info.get("segment_offset_samples", 0)
        offset = base_offset + slice_idx * orig_clip
        waveform, _ = torchaudio.load(
            path, frame_offset=offset, num_frames=orig_clip
        )

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

        # Return a composite key: "track_id||segment_start" — unique per segment.
        # Multiple clips from the same segment share this key (same label).
        # The track_id prefix enables test-time aggregation per track.
        segment_key = f"{track_id}||{info['segment_start']}"
        return waveform, label, segment_key


class HXMSAAudioTrain(_HXMSAAudioBase):
    pass


class HXMSAAudioVal(_HXMSAAudioBase):
    pass


class HXMSAAudioTest(HXMSAAudioVal):
    pass


class HXMSADataModule(BaseDataModule):
    pass


class HXMSAEmbeddingDataModule(pl.LightningDataModule):
    """DataModule for probing on pre-extracted HXMSA embeddings."""

    def __init__(
        self,
        embedding_root: str,
        layer_idx: int,
        train_jsonl: str,
        val_jsonl: str,
        test_jsonl: str,
        batch_size: int = 16,
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
            self.train_dataset = HXMSAEmbeddingDataset(
                self.embedding_root / "train",
                self.layer_idx,
                self.train_jsonl,
            )
            self.val_dataset = HXMSAEmbeddingDataset(
                self.embedding_root / "val",
                self.layer_idx,
                self.val_jsonl,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = HXMSAEmbeddingDataset(
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

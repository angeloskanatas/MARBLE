# marble/tasks/HXMSAStructure/datamodule.py

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule
from marble.tasks.HXMSA.embedding_dataset import normalize_label, LABEL2IDX, IDX2LABEL
from marble.tasks.HXMSAStructure.embedding_dataset import HXMSAStructureEmbeddingDataset


class _HXMSAStructureAudioBase(Dataset):
    """
    Audio dataset for full-track structure segmentation.

    Reads per-track JSONL (one entry per track with segments list).
    Loads full tracks, splits into non-overlapping clips of `clip_seconds`.
    For each clip, generates per-frame section labels + boundary marks.
    """

    LABEL2IDX = LABEL2IDX
    IDX2LABEL = IDX2LABEL
    NUM_CLASSES = len(IDX2LABEL)

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        jsonl: str,
        label_freq: int,
        boundary_num_neighbors: int = 2,
        channel_mode: str = "first",
        min_clip_ratio: float = 0.5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.channel_mode = channel_mode
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.label_freq = label_freq
        self.boundary_num_neighbors = boundary_num_neighbors
        self.min_clip_ratio = min_clip_ratio

        with open(jsonl, "r") as f:
            self.meta = [json.loads(line) for line in f]

        # Pre-parse segments
        self.track_segments: List[List[dict]] = []
        for info in self.meta:
            segs = []
            for s in info["segments"]:
                segs.append({
                    "start": s["start"],
                    "end": s["end"],
                    "label_idx": self.LABEL2IDX[normalize_label(s["label"])],
                })
            self.track_segments.append(segs)

        # Resamplers
        self.resamplers = {}
        for info in self.meta:
            orig_sr = int(info["sample_rate"])
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

        # Build index map: (track_idx, slice_idx, orig_sr, orig_clip_frames)
        self.index_map: List[Tuple[int, int, int, int]] = []
        for track_idx, info in enumerate(self.meta):
            orig_sr = int(info["sample_rate"])
            total_samples = int(info["num_samples"])
            orig_clip_frames = int(self.clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue

            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full

            for slice_idx in range(n_slices):
                self.index_map.append((track_idx, slice_idx, orig_sr, orig_clip_frames))

    def __len__(self) -> int:
        return len(self.index_map)

    def _generate_frame_labels(
        self, segments, clip_start, clip_end, label_len
    ):
        """Generate per-frame section labels and boundary mask for a clip."""
        frame_labels = np.full(label_len, -1, dtype=np.int32)
        boundaries = np.zeros(label_len, dtype=np.float32)

        for seg in segments:
            if seg["end"] <= clip_start or seg["start"] >= clip_end:
                continue
            rel_start = max(0.0, seg["start"] - clip_start)
            rel_end = min(clip_end - clip_start, seg["end"] - clip_start)
            fs = max(0, min(int(round(rel_start * self.label_freq)), label_len))
            fe = max(0, min(int(round(rel_end * self.label_freq)), label_len))
            frame_labels[fs:fe] = seg["label_idx"]

        for seg in segments:
            t = seg["start"]
            if clip_start < t < clip_end:
                fi = int(round((t - clip_start) * self.label_freq))
                if 0 <= fi < label_len:
                    boundaries[fi] = 1.0

        if self.boundary_num_neighbors > 0:
            widened = np.copy(boundaries)
            for k in range(1, self.boundary_num_neighbors + 1):
                w = 1.0 / (k + 1)  # k=1 → 0.5, k=2 → 0.333; stays ≤ 0.5 for clean eval
                widened[k:] += boundaries[:-k] * w
                widened[:-k] += boundaries[k:] * w
            boundaries = np.clip(widened, 0.0, 1.0)

        # Forward/backward fill unlabeled frames
        for i in range(1, label_len):
            if frame_labels[i] == -1 and frame_labels[i - 1] != -1:
                frame_labels[i] = frame_labels[i - 1]
        for i in range(label_len - 2, -1, -1):
            if frame_labels[i] == -1 and frame_labels[i + 1] != -1:
                frame_labels[i] = frame_labels[i + 1]
        frame_labels[frame_labels == -1] = self.LABEL2IDX["inst"]

        return frame_labels, boundaries

    def __getitem__(self, idx: int):
        track_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[idx]
        info = self.meta[track_idx]
        path = info["audio_path"]
        segments = self.track_segments[track_idx]

        # Load audio clip
        offset = slice_idx * orig_clip_frames
        waveform, _ = torchaudio.load(path, frame_offset=offset, num_frames=orig_clip_frames)

        # Guard: MP3 num_frames can be inaccurate; offset may exceed actual file length
        if waveform.size(1) == 0:
            waveform = torch.zeros(self.channels, self.clip_len_target)
            clip_start = slice_idx * (orig_clip_frames / orig_sr)
            clip_end = clip_start + self.clip_seconds
            label_len = int(self.label_freq * self.clip_seconds)
            frame_labels, boundaries = self._generate_frame_labels(
                segments, clip_start, clip_end, label_len
            )
            targets = {
                "frame_labels": torch.from_numpy(frame_labels).long(),
                "boundaries": torch.from_numpy(boundaries).float(),
            }
            return waveform, targets, info["track_id"]

        # Channel handling
        orig_ch = waveform.size(0)
        if orig_ch >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(dim=0, keepdim=True)
                else:
                    waveform = waveform[0:1]
            else:
                waveform = waveform[:self.channels]
        else:
            last = waveform[-1:].repeat(self.channels - orig_ch, 1)
            waveform = torch.cat([waveform, last], dim=0)

        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        if waveform.size(1) < self.clip_len_target:
            waveform = F.pad(waveform, (0, self.clip_len_target - waveform.size(1)))
        elif waveform.size(1) > self.clip_len_target:
            waveform = waveform[:, :self.clip_len_target]

        # Generate labels
        clip_start = slice_idx * (orig_clip_frames / orig_sr)
        clip_end = clip_start + self.clip_seconds
        label_len = int(self.label_freq * self.clip_seconds)
        frame_labels, boundaries = self._generate_frame_labels(
            segments, clip_start, clip_end, label_len
        )

        targets = {
            "frame_labels": torch.from_numpy(frame_labels).long(),
            "boundaries": torch.from_numpy(boundaries).float(),
        }
        return waveform, targets, info["track_id"]


class HXMSAStructureAudioTrain(_HXMSAStructureAudioBase):
    pass


class HXMSAStructureAudioVal(_HXMSAStructureAudioBase):
    pass


class HXMSAStructureAudioTest(HXMSAStructureAudioVal):
    pass


class HXMSAStructureDataModule(BaseDataModule):
    pass


class HXMSAStructureEmbeddingDataModule(pl.LightningDataModule):
    """DataModule for probing on pre-extracted frame-level structure embeddings."""

    def __init__(
        self,
        embedding_root: str,
        layer_idx: int,
        train_jsonl: str,
        val_jsonl: str,
        test_jsonl: str,
        clip_seconds: float,
        label_freq: int,
        boundary_num_neighbors: int = 2,
        min_clip_ratio: float = 0.5,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.embedding_root = Path(embedding_root)
        self.layer_idx = layer_idx
        self.train_jsonl = train_jsonl
        self.val_jsonl = val_jsonl
        self.test_jsonl = test_jsonl
        self.clip_seconds = clip_seconds
        self.label_freq = label_freq
        self.boundary_num_neighbors = boundary_num_neighbors
        self.min_clip_ratio = min_clip_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        kw = dict(
            layer_idx=self.layer_idx,
            clip_seconds=self.clip_seconds,
            label_freq=self.label_freq,
            boundary_num_neighbors=self.boundary_num_neighbors,
            min_clip_ratio=self.min_clip_ratio,
        )
        if stage in (None, "fit"):
            self.train_dataset = HXMSAStructureEmbeddingDataset(
                embedding_dir=self.embedding_root / "train",
                jsonl=self.train_jsonl,
                **kw,
            )
            self.val_dataset = HXMSAStructureEmbeddingDataset(
                embedding_dir=self.embedding_root / "val",
                jsonl=self.val_jsonl,
                **kw,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = HXMSAStructureEmbeddingDataset(
                embedding_dir=self.embedding_root / "test",
                jsonl=self.test_jsonl,
                **kw,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, prefetch_factor=2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, prefetch_factor=2,
        )

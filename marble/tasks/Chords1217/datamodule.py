# marble/tasks/Chords1217/datamodule.py

import os
import json
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from marble.core.base_datamodule import BaseDataModule, BaseAudioDataset
from marble.utils.utils import chord_to_majmin, id2chord_str


class _Chords1217AudioBase(BaseAudioDataset):
    """
    Base dataset for chord recognition tasks.
    - Splits each audio file into clips of length `clip_seconds`.
    - For each clip, returns:
        * waveform tensor of shape (channels, clip_len_target)
        * targets dict with:
            - "chord": 1D LongTensor of length label_len (frame-level chord indices, 0–24)
    """
    EXAMPLE_JSONL = {
        "audio_path": "data/Chords1217/audio/TRWTQYM149E35B4C40.flac", 
        "label": 
            [
                {"start_time": 0.0, "end_time": 0.42124700000000004, "chord_str": "N"}, 
                {"start_time": 0.42124700000000004, "end_time": 1.2562470000000001, 
                "chord_str": "N"}, 
                {"start_time": 1.2562470000000001, "end_time": 21.840737000000004, "chord_str": "E:maj"}, 
                {"start_time": 21.840737, "end_time": 23.048174, "chord_str": "F#:maj"}, 
                {"start_time": 23.048174000000003, "end_time": 24.290442000000002, "chord_str": "B:maj"}, 
                {"start_time": 24.290442000000002, "end_time": 25.474660000000004, "chord_str": "E:maj"}, 
                {"start_time": 25.47466, "end_time": 26.682097, "chord_str": "C#:maj"}, 
                {"start_time": 26.682097000000002, "end_time": 27.877925, "chord_str": "F#:maj"}, 
                {"start_time": 27.877925, "end_time": 29.073753, "chord_str": "B:maj"}, 
                {"start_time": 29.073753000000004, "end_time": 29.372436000000004, "chord_str": "E:maj"}, 
                {"start_time": 29.372436, "end_time": 31.546678, "chord_str": "D:maj/2"}, 
                {"start_time": 31.546678000000004, "end_time": 52.293707000000005, "chord_str": "E:maj"}, 
                {"start_time": 52.293707000000005, "end_time": 53.466315, "chord_str": "F#:maj"}, 
                {"start_time": 53.466315, "end_time": 54.720192000000004, "chord_str": "B:maj"}, 
                {"start_time": 54.720192000000004, "end_time": 55.939240000000005, "chord_str": "E:maj"}, 
                {"start_time": 55.939240000000005, "end_time": 57.158288000000006, "chord_str": "C#:maj"}, 
                {"start_time": 57.158288000000006, "end_time": 58.365725000000005, "chord_str": "F#:maj"}, 
                {"start_time": 58.365725000000005, "end_time": 59.584773000000006, "chord_str": "B:maj"}, 
                {"start_time": 59.584773000000006, "end_time": 59.893728, "chord_str": "E:maj"}, 
                {"start_time": 59.893728, "end_time": 62.022868, "chord_str": "D:maj/2"}, 
                {"start_time": 62.022868, "end_time": 76.26493500000001, "chord_str": "E:maj"}, 
                {"start_time": 76.264934999, "end_time": 96.71347999900001, "chord_str": "E:maj"}, 
                {"start_time": 96.71348, "end_time": 97.94413800000001, "chord_str": "F#:maj"}, 
                {"start_time": 97.94413800000001, "end_time": 99.16318600000001, "chord_str": "B:maj"}, 
                {"start_time": 99.16318600000001, "end_time": 100.393843, "chord_str": "E:maj"}, 
                {"start_time": 100.393843, "end_time": 101.578061, "chord_str": "C#:maj"}, 
                {"start_time": 101.578061, "end_time": 102.80871900000001, "chord_str": "F#:maj"}, 
                {"start_time": 102.80871900000001, "end_time": 103.969716, "chord_str": "B:maj"}, 
                {"start_time": 103.969716, "end_time": 104.27789100000001, "chord_str": "E:maj"}, 
                {"start_time": 104.27789100000001, "end_time": 106.45425100000001, "chord_str": "D:maj/2"}, 
                {"start_time": 106.45425100000001, "end_time": 114.15805800000001, "chord_str": "E:maj"}, 
                {"start_time": 114.15805800000001, "end_time": 118.987755, "chord_str": "N"}
            ], 
        "duration": 119.013875, 
        "sample_rate": 24000, 
        "num_samples": 2856333, 
        "bit_depth": 0, 
        "channels": 1
        }

    def __init__(self, jsonl: str, sample_rate: int, channels: int,
                 clip_seconds: float, label_freq: int, channel_mode: str="first",
                 min_clip_ratio: float=1.0, backend: Optional[str] = None):
        super().__init__(
            jsonl=jsonl,
            sample_rate=sample_rate,
            channels=channels,
            clip_seconds=clip_seconds,
            label_freq=label_freq,
            channel_mode=channel_mode,
            min_clip_ratio=min_clip_ratio,
            backend=backend
        )
        # Build a per-file list of (start_time, chord_idx) sorted by start_time
        self.chords_meta: List[List[Tuple[float, int]]] = []
        for info in self.meta:
            ann_list = info.get("label", [])
            annotated: List[Tuple[float, int]] = []
            for seg in ann_list:
                t0 = float(seg["start_time"])
                idx = chord_to_majmin(seg["chord_str"])
                annotated.append((t0, idx))
            annotated.sort(key=lambda x: x[0])
            self.chords_meta.append(annotated)

    def get_targets(self, file_idx: int, slice_idx: int, orig_sr: int, orig_clip_frames: int):
        """
        Generate a frame-level chord label sequence for a single clip.
        
        Steps:
          1. Determine clip start/end times in seconds.
          2. Collect annotations whose start_time falls within [clip_start, clip_end).
          3. If no annotation covers clip_start, either carry over the last chord from before the clip
             or insert “no chord” (24) at clip_start.
          4. Append a “no chord” sentinel at clip_end to mark the final segment.
          5. For each frame (indexed by label_freq), assign the chord active at that frame’s time.
             If encountering a temporary “no chord” marker, inherit the previous chord label.

        Args:
            file_idx (int): Index of the audio file in self.meta.
            slice_idx (int): Which slice number (0-based) within that file.
            orig_sr (int): Original sample rate of the audio file.
            orig_clip_frames (int): Number of samples per clip at the original sample rate
                                    (i.e., floor(clip_seconds * orig_sr)).

        Returns:
            torch.LongTensor: A 1D tensor of length `label_len = int(self.label_freq * self.clip_seconds)`,
                              where each element is a chord index (0–23 for chords, 24 for “no chord”).
        """
        # 1. Get sorted (time, chord_idx) annotations for this file
        chord_ann = self.chords_meta[file_idx]

        # 2. Compute clip start/end in seconds
        clip_start = slice_idx * (orig_clip_frames / orig_sr)
        clip_end = clip_start + self.clip_seconds

        # 3. Prepare an array of length label_len to hold chord indices
        label_len = int(self.label_freq * self.clip_seconds)
        chord_seq = np.zeros(label_len, dtype=np.int64)

        # If no annotations exist, fill all frames with “no chord” (24)
        if not chord_ann:
            chord_seq[:] = 24
            return torch.from_numpy(chord_seq)

        # 4. Collect segments that start inside the clip
        segments: List[Tuple[float, int]] = []
        for t0, cidx in chord_ann:
            if t0 >= clip_end:
                break
            if t0 >= clip_start:
                segments.append((t0, cidx))

        # 5. If the first segment starts after clip_start, check for carryover or set “no chord”
        if not segments or segments[0][0] > clip_start:
            prev_chord = None
            for t0, cidx in reversed(chord_ann):
                if t0 < clip_start:
                    prev_chord = cidx
                    break
            # Carry over previous chord if it overlaps; otherwise start with “no chord”
            start_label = prev_chord if prev_chord is not None else 24
            segments.insert(0, (clip_start, start_label))

        # 6. Append a “no chord” sentinel at clip_end
        segments.append((clip_end, 24))

        # 7. Iterate over frames and assign labels
        seg_ptr = 0
        current_label = segments[0][1]
        next_change = segments[1][0]

        for i in range(label_len):
            t = clip_start + i / self.label_freq
            # Advance to the next segment if time passes its start
            while t >= next_change and seg_ptr + 1 < len(segments) - 1:
                seg_ptr += 1
                current_label = segments[seg_ptr][1]
                next_change = segments[seg_ptr + 1][0] if seg_ptr + 1 < len(segments) else clip_end

            # If current_label is “no chord” but not at the very beginning, inherit previous chord
            if current_label == 24 and seg_ptr > 0:
                current_label = segments[seg_ptr - 1][1]

            chord_seq[i] = current_label

        return torch.from_numpy(chord_seq)


class Chords1217AudioTrain(_Chords1217AudioBase):
    """Training split: shuffle in DataLoader."""
    pass


class Chords1217AudioVal(_Chords1217AudioBase):
    """Validation split: no shuffling."""
    pass


class Chords1217AudioTest(Chords1217AudioVal):
    """Test split: same behavior as validation."""
    pass

class Chords1217DataModule(BaseDataModule):
    pass


class Chords1217EmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted Chords1217 embeddings (frame-level).
    """

    def __init__(
        self,
        embedding_root: str,
        layer_idx: int,
        train_jsonl: str,
        val_jsonl: str,
        test_jsonl: str,
        clip_seconds: float,
        label_freq: int,
        batch_size: int = 8,
        num_workers: int = 8,
        min_clip_ratio: float = 0.8,
    ):
        super().__init__()
        from pathlib import Path
        from marble.tasks.Chords1217.embedding_dataset import Chords1217EmbeddingDataset
        self._dataset_cls = Chords1217EmbeddingDataset

        self.embedding_root = Path(embedding_root)
        self.layer_idx = layer_idx
        self.train_jsonl = train_jsonl
        self.val_jsonl = val_jsonl
        self.test_jsonl = test_jsonl
        self.clip_seconds = clip_seconds
        self.label_freq = label_freq
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_clip_ratio = min_clip_ratio

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_dataset = self._dataset_cls(
                embedding_dir=self.embedding_root / "train",
                layer_idx=self.layer_idx,
                jsonl=self.train_jsonl,
                clip_seconds=self.clip_seconds,
                label_freq=self.label_freq,
                min_clip_ratio=self.min_clip_ratio,
            )
            self.val_dataset = self._dataset_cls(
                embedding_dir=self.embedding_root / "val",
                layer_idx=self.layer_idx,
                jsonl=self.val_jsonl,
                clip_seconds=self.clip_seconds,
                label_freq=self.label_freq,
                min_clip_ratio=self.min_clip_ratio,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = self._dataset_cls(
                embedding_dir=self.embedding_root / "test",
                layer_idx=self.layer_idx,
                jsonl=self.test_jsonl,
                clip_seconds=self.clip_seconds,
                label_freq=self.label_freq,
                min_clip_ratio=self.min_clip_ratio,
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

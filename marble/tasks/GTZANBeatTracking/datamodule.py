# marble/tasks/GTZANBeatTracking/datamodule.py

import os
import json
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule
from marble.utils.utils import widen_temporal_events
from marble.utils.utils import times_to_mask, mask_to_times
from marble.tasks.GTZANBeatTracking.embedding_dataset import GTZANBeatTrackingEmbeddingDataset


class _GTZANBeatTrackingAudioBase(Dataset):
    """
    Base dataset for GTZAN beat/downbeat/tempo tasks.
    - Splits each audio file into clips of length `clip_seconds`.
    - For each clip, returns:
        * waveform tensor of shape (channels, clip_len_target)
        * targets dict with:
            - "beat":     1D tensor of length label_len (beat onset mask)
            - "downbeat": 1D tensor of length label_len (downbeat onset mask)
            - "tempo":    scalar tensor (BPM) for the entire track
    """
    EXAMPLE_JSONL = {
        "audio_path": "data/GTZAN/genres/blues/blues.00029.wav", 
        "label": {
            # "beat" and "downbeat" are lists of onset times in seconds
            "beat": [0.428173, 0.898377, 1.362776, 1.827176, 2.29738, 2.755974, 3.191348, 3.649942, 4.114341, 4.584545, 5.048944, 5.513343, 5.977742, 6.436337, 6.900736, 7.35933, 7.829534, 8.299738, 8.764137, 9.228536, 9.71035, 10.186359, 10.668173, 11.138377, 11.614386, 12.08459, 12.554795, 13.024999, 13.495203, 13.994432, 14.476246, 14.94645, 15.509534, 15.991348, 16.461552, 16.925951, 17.396155, 17.860554, 18.340235, 18.804634, 19.274838, 19.739237, 20.203636, 20.66223, 21.132434, 21.730348, 22.212162, 22.682366, 23.15257, 23.616969, 24.087173, 24.586402, 25.068217, 25.573251, 26.171164, 26.641368, 27.111573, 27.581777, 28.06359, 28.527989, 28.980779, 29.450983], 
            "downbeat": [0.428173, 2.29738, 4.114341, 5.977742, 7.829534, 9.71035, 11.614386, 13.495203, 15.509534, 17.396155, 19.274838, 21.132434, 23.15257, 25.068217, 27.111573, 28.980779], 
            "tempo": 126.46, 
            "meter": "4/4"
        }, 
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
        label_freq: int,
        num_neighbors: int = 0,
        use_local_bpm: bool = True,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
    ):
        super().__init__()
        assert isinstance(sample_rate, (int, float)), "sample_rate must be an integer or float"
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.channel_mode = channel_mode
        self.clip_seconds = clip_seconds
        # Target clip length in samples after resampling
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.label_freq = label_freq
        self.num_neighbors = num_neighbors
        self.use_local_bpm = use_local_bpm
        self.min_clip_ratio = min_clip_ratio

        # Validate channel_mode if we expect mono output
        if self.channels == 1 and self.channel_mode not in ["first", "mix", "random"]:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}")

        # Load metadata entries from JSONL file
        with open(jsonl, 'r') as f:
            self.meta: List[dict] = [json.loads(line) for line in f]

        # Pre-extract beat times, downbeat times, and tempo (BPM) from metadata
        self.beat_times_meta: List[np.ndarray] = []
        self.db_times_meta:   List[np.ndarray] = []
        self.tempo_list:      List[float]     = []
        for info in self.meta:
            beat_arr = np.array(info['label'].get('beat', []), dtype=np.float32)
            db_arr   = np.array(info['label'].get('downbeat', []), dtype=np.float32)
            tempo_bpm = float(info['label'].get('tempo', 0.0))
            self.beat_times_meta.append(beat_arr)
            self.db_times_meta.append(db_arr)
            self.tempo_list.append(tempo_bpm)

        # Pre-create resamplers for any original sample rate != target sample_rate
        self.resamplers = {}
        for info in self.meta:
            orig_sr = int(info['sample_rate'])
            if orig_sr != self.sample_rate:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

        # Build an index map: for each file, compute how many clips can be extracted
        # and store tuples of (file_idx, slice_idx, orig_sr, orig_clip_frames)
        self.index_map: List[Tuple[int, int, int, int]] = []
        for file_idx, info in enumerate(self.meta):
            orig_sr = int(info['sample_rate'])
            total_samples = int(info['num_samples'])
            orig_clip_frames = int(self.clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue  # skip if clip length is zero or metadata is invalid

            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full

            for slice_idx in range(n_slices):
                self.index_map.append((file_idx, slice_idx, orig_sr, orig_clip_frames))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Returns:
            waveform: torch.Tensor of shape (channels, clip_len_target)
            targets: dict {
                "beat":     torch.Tensor(shape=(label_len,), dtype=float32),
                "downbeat": torch.Tensor(shape=(label_len,), dtype=float32),
                "tempo":    torch.Tensor(shape=(), dtype=float32)
            }
            audio_path: str - path to the audio file
        """
        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[idx]
        info = self.meta[file_idx]
        audio_path = info['audio_path']

        # 1. Load and preprocess waveform
        waveform = self._load_and_preprocess(
            path=audio_path,
            slice_idx=slice_idx,
            orig_sr=orig_sr,
            orig_clip_frames=orig_clip_frames
        )
        # waveform.shape == (self.channels, self.clip_len_target)

        # 2. Generate beat and downbeat masks for this clip
        beat_arr = self.beat_times_meta[file_idx]   # numpy array of full-track beat times
        db_arr   = self.db_times_meta[file_idx]     # numpy array of full-track downbeat times

        # Compute clip start/end times in seconds (based on original sampling rate)
        clip_start_time = slice_idx * (orig_clip_frames / orig_sr)
        clip_end_time   = clip_start_time + self.clip_seconds

        # Compute length of label sequence: floor(label_freq * clip_seconds)
        label_len = int(self.label_freq * self.clip_seconds)
        beat_mask = np.zeros(label_len, dtype=np.float32)
        db_mask   = np.zeros(label_len, dtype=np.float32)

        # Populate beat mask: if a beat time falls in [clip_start_time, clip_end_time),
        # map it to the corresponding frame index in the mask array
        for t in beat_arr:
            if clip_start_time <= t < clip_end_time:
                rel = t - clip_start_time
                frame_idx = int(round(rel * self.label_freq))
                if 0 <= frame_idx < label_len:
                    beat_mask[frame_idx] = 1.0

        # Populate downbeat mask similarly
        for t in db_arr:
            if clip_start_time <= t < clip_end_time:
                rel = t - clip_start_time
                frame_idx = int(round(rel * self.label_freq))
                if 0 <= frame_idx < label_len:
                    db_mask[frame_idx] = 1.0
        
        # clip bpm estimation
        if self.use_local_bpm:
            est_tempo = beat_mask.sum() / self.clip_seconds * 60.0  # Estimate tempo from beat mask
        else:
            est_tempo = self.tempo_list[file_idx]
        tempo_tensor = torch.tensor(est_tempo, dtype=torch.float32)

        # Optionally widen events by num_neighbors frames
        if self.num_neighbors > 0:
            beat_mask = widen_temporal_events(beat_mask, self.num_neighbors)
            db_mask   = widen_temporal_events(db_mask, self.num_neighbors)

        beat_tensor = torch.from_numpy(beat_mask)
        db_tensor   = torch.from_numpy(db_mask)

        label = {
            "beat":     beat_tensor,
            "downbeat": db_tensor,
            "tempo":    tempo_tensor
        }
        
        return waveform, label, audio_path

    def _load_and_preprocess(
        self,
        path: str,
        slice_idx: int,
        orig_sr: int,
        orig_clip_frames: int
    ) -> torch.Tensor:
        """
        Load a slice of audio from disk and apply:
        - channel selection / downmixing
        - resampling to self.sample_rate
        - padding or truncation to ensure shape (self.channels, self.clip_len_target)

        Returns:
            Tensor of shape (self.channels, self.clip_len_target)
        """
        offset = slice_idx * orig_clip_frames
        waveform, _ = torchaudio.load(
            path,
            frame_offset=offset,
            num_frames=orig_clip_frames
        )  # waveform.shape == (orig_ch, actual_frames)

        orig_ch = waveform.size(0)

        # 1. Channel handling / downmixing
        if orig_ch >= self.channels:
            # If we only need fewer channels than original
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1, :]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(dim=0, keepdim=True)
                elif self.channel_mode == "random":
                    # Use 50% chance to mix, 50% chance to pick a random channel
                    if torch.rand(()) < 0.5:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        idx = torch.randint(0, orig_ch, ())
                        waveform = waveform[idx:idx+1, :]
                else:
                    raise ValueError(f"Unknown channel_mode: {self.channel_mode}")
            else:
                # Keep the first self.channels channels
                waveform = waveform[: self.channels, :]
        else:
            # orig_ch < self.channels: replicate last channel to match desired channel count
            deficit = self.channels - orig_ch
            tail = waveform[-1:, :].repeat(deficit, 1)
            waveform = torch.cat([waveform, tail], dim=0)

        # 2. Resample if needed
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # 3. Pad or truncate to exact length
        cur_len = waveform.size(1)
        if cur_len < self.clip_len_target:
            pad_amt = self.clip_len_target - cur_len
            waveform = F.pad(waveform, (0, pad_amt), mode='constant', value=0.0)
        elif cur_len > self.clip_len_target:
            waveform = waveform[:, : self.clip_len_target]

        return waveform


class GTZANBeatTrackingAudioTrain(_GTZANBeatTrackingAudioBase):
    """Training split: shuffle in DataLoader."""
    pass


class GTZANBeatTrackingAudioVal(_GTZANBeatTrackingAudioBase):
    """Validation split: no shuffling."""
    pass


class GTZANBeatTrackingAudioTest(GTZANBeatTrackingAudioVal):
    """Test split: same behavior as validation."""
    pass


class GTZANBeatTrackingDataModule(BaseDataModule):
    pass


class GTZANBeatTrackingEmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted GTZAN beat-tracking embeddings.
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
        use_local_bpm: bool = True,
        train_num_neighbors: int = 2,
        eval_num_neighbors: int = 0,
    ):
        super().__init__()
        from pathlib import Path

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
        self.use_local_bpm = use_local_bpm
        self.train_num_neighbors = train_num_neighbors
        self.eval_num_neighbors = eval_num_neighbors

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_dataset = GTZANBeatTrackingEmbeddingDataset(
                embedding_dir=self.embedding_root / "train",
                layer_idx=self.layer_idx,
                jsonl=self.train_jsonl,
                clip_seconds=self.clip_seconds,
                label_freq=self.label_freq,
                num_neighbors=self.train_num_neighbors,
                use_local_bpm=self.use_local_bpm,
                min_clip_ratio=self.min_clip_ratio,
            )
            self.val_dataset = GTZANBeatTrackingEmbeddingDataset(
                embedding_dir=self.embedding_root / "val",
                layer_idx=self.layer_idx,
                jsonl=self.val_jsonl,
                clip_seconds=self.clip_seconds,
                label_freq=self.label_freq,
                num_neighbors=self.eval_num_neighbors,
                use_local_bpm=self.use_local_bpm,
                min_clip_ratio=self.min_clip_ratio,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = GTZANBeatTrackingEmbeddingDataset(
                embedding_dir=self.embedding_root / "test",
                layer_idx=self.layer_idx,
                jsonl=self.test_jsonl,
                clip_seconds=self.clip_seconds,
                label_freq=self.label_freq,
                num_neighbors=self.eval_num_neighbors,
                use_local_bpm=self.use_local_bpm,
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


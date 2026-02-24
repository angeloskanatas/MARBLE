# marble/tasks/HookTheoryKey/datamodule.py

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule


class _HookTheoryKeyAudioBase(Dataset):
    """
    Base dataset for HookTheoryKey audio:
    - Splits each audio file into non-overlapping clips of length `clip_seconds` (last clip zero-padded).
    - This is a regression task with two labels: arousal and valence.
    """
    LABEL2IDX = {     # mir_eval format key annotation
        'C major': 0,
        'Db major': 1,
        'D major': 2,
        'Eb major': 3,
        'E major': 4,
        'F major': 5,
        'Gb major': 6,
        'G major': 7,
        'Ab major': 8,
        'A major': 9,
        'Bb major': 10,
        'B major': 11,
        'C minor': 12,
        'Db minor': 13,
        'D minor': 14,
        'Eb minor': 15,
        'E minor': 16,
        'F minor': 17,
        'Gb minor': 18,
        'G minor': 19,
        'Ab minor': 20,
        'A minor': 21,
        'Bb minor': 22,
        'B minor': 23
    }
    FULL_LABEL2IDX = {
        'A dorian': 0,
        'A harmonicMinor': 1,
        'A locrian': 2,
        'A lydian': 3,
        'A major': 4,
        'A minor': 5,
        'A mixolydian': 6,
        'A phrygian': 7,
        'A phrygianDominant': 8,
        'Ab dorian': 9,
        'Ab harmonicMinor': 10,
        'Ab lydian': 11,
        'Ab major': 12,
        'Ab minor': 13,
        'Ab mixolydian': 14,
        'Ab phrygian': 15,
        'Ab phrygianDominant': 16,
        'B dorian': 17,
        'B harmonicMinor': 18,
        'B locrian': 19,
        'B lydian': 20,
        'B major': 21,
        'B minor': 22,
        'B mixolydian': 23,
        'B phrygian': 24,
        'B phrygianDominant': 25,
        'Bb dorian': 26,
        'Bb harmonicMinor': 27,
        'Bb locrian': 28,
        'Bb lydian': 29,
        'Bb major': 30,
        'Bb minor': 31,
        'Bb mixolydian': 32,
        'Bb phrygian': 33,
        'Bb phrygianDominant': 34,
        'C dorian': 35,
        'C harmonicMinor': 36,
        'C locrian': 37,
        'C lydian': 38,
        'C major': 39,
        'C minor': 40,
        'C mixolydian': 41,
        'C phrygian': 42,
        'C phrygianDominant': 43,
        'D dorian': 44,
        'D harmonicMinor': 45,
        'D locrian': 46,
        'D lydian': 47,
        'D major': 48,
        'D minor': 49,
        'D mixolydian': 50,
        'D phrygian': 51,
        'Db dorian': 52,
        'Db harmonicMinor': 53,
        'Db locrian': 54,
        'Db lydian': 55,
        'Db major': 56,
        'Db minor': 57,
        'Db mixolydian': 58,
        'Db phrygian': 59,
        'Db phrygianDominant': 60,
        'E dorian': 61,
        'E harmonicMinor': 62,
        'E locrian': 63,
        'E lydian': 64,
        'E major': 65,
        'E minor': 66,
        'E mixolydian': 67,
        'E phrygian': 68,
        'E phrygianDominant': 69,
        'Eb dorian': 70,
        'Eb harmonicMinor': 71,
        'Eb locrian': 72,
        'Eb lydian': 73,
        'Eb major': 74,
        'Eb minor': 75,
        'Eb mixolydian': 76,
        'Eb phrygian': 77,
        'Eb phrygianDominant': 78,
        'F dorian': 79,
        'F harmonicMinor': 80,
        'F locrian': 81,
        'F lydian': 82,
        'F major': 83,
        'F minor': 84,
        'F mixolydian': 85,
        'F phrygian': 86,
        'F phrygianDominant': 87,
        'G dorian': 88,
        'G harmonicMinor': 89,
        'G locrian': 90,
        'G lydian': 91,
        'G major': 92,
        'G minor': 93,
        'G mixolydian': 94,
        'G phrygian': 95,
        'Gb dorian': 96,
        'Gb harmonicMinor': 97,
        'Gb locrian': 98,
        'Gb lydian': 99,
        'Gb major': 100,
        'Gb minor': 101,
        'Gb mixolydian': 102,
        'Gb phrygian': 103,
        'Gb phrygianDominant': 104
    }

    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
    IDX2FULL_LABEL = {v: k for k, v in FULL_LABEL2IDX.items()}

    EXAMPLE_JSONL = {
        "audio_path": "data/HookTheory/hooktheory_clips/ZdPoDNVRgnM.mp3", 
        "ori_uid": "5BmEGm-mraE", 
        "label": "D major", 
        "duration": 10.68, 
        "sample_rate": 48000, 
        "num_samples": 512640, 
        "bit_depth": 16, 
        "channels": 2
    }
    
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        jsonl: str,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
        maj_minor_only: bool = True
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
        
        # 过滤标签
        self.maj_minor_only = maj_minor_only
        if self.maj_minor_only:
            print(f"Filtering labels to major and minor keys only. Total {len(self.meta)} entries before filtering.")
            self.meta = [
                info for info in self.meta
                if info['label'] in self.LABEL2IDX
            ]
            print(f"Filtered to {len(self.meta)} entries with major/minor keys.")

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
        ori_uid = info['ori_uid']
        label = self.LABEL2IDX[info['label']]  # (1,)

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
        return waveform, label, ori_uid # we do not use path in GS since it is pre split into clips


class HookTheoryKeyAudioTrain(_HookTheoryKeyAudioBase):
    """
    训练集：DataModule 中设置 shuffle=True。
    """
    pass


class HookTheoryKeyAudioVal(_HookTheoryKeyAudioBase):
    """
    验证集：DataModule 中设置 shuffle=False。
    """
    pass


class HookTheoryKeyAudioTest(HookTheoryKeyAudioVal):
    """
    测试集：同验证集逻辑。
    """
    pass


class HookTheoryKeyDataModule(BaseDataModule):
    pass


class HookTheoryKeyEmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted HookTheoryKey embeddings.
    """

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
        from marble.tasks.HookTheoryKey.embedding_dataset import HookTheoryKeyEmbeddingDataset
        self._dataset_cls = HookTheoryKeyEmbeddingDataset

        self.embedding_root = Path(embedding_root)
        self.layer_idx = layer_idx
        self.train_jsonl = train_jsonl
        self.val_jsonl = val_jsonl
        self.test_jsonl = test_jsonl
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_dataset = self._dataset_cls(
                self.embedding_root / "train",
                self.layer_idx,
                self.train_jsonl,
            )
            self.val_dataset = self._dataset_cls(
                self.embedding_root / "val",
                self.layer_idx,
                self.val_jsonl,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = self._dataset_cls(
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

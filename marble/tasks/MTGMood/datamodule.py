# marble/tasks/MTGMood/datamodule.py
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule, BaseAudioDataset


class _MTGMoodAudioBase(BaseAudioDataset):
    LABEL2IDX = {
        'mood/theme---background': 0,
        'mood/theme---film': 1,
        'mood/theme---melancholic': 2,
        'mood/theme---melodic': 3,
        'mood/theme---children': 4,
        'mood/theme---relaxing': 5,
        'mood/theme---documentary': 6,
        'mood/theme---emotional': 7,
        'mood/theme---space': 8,
        'mood/theme---love': 9,
        'mood/theme---drama': 10,
        'mood/theme---adventure': 11,
        'mood/theme---energetic': 12,
        'mood/theme---heavy': 13,
        'mood/theme---dark': 14,
        'mood/theme---calm': 15,
        'mood/theme---action': 16,
        'mood/theme---dramatic': 17,
        'mood/theme---epic': 18,
        'mood/theme---powerful': 19,
        'mood/theme---upbeat': 20,
        'mood/theme---slow': 21,
        'mood/theme---inspiring': 22,
        'mood/theme---soft': 23,
        'mood/theme---meditative': 24,
        'mood/theme---fun': 25,
        'mood/theme---happy': 26,
        'mood/theme---positive': 27,
        'mood/theme---romantic': 28,
        'mood/theme---sad': 29,
        'mood/theme---hopeful': 30,
        'mood/theme---motivational': 31,
        'mood/theme---deep': 32,
        'mood/theme---uplifting': 33,
        'mood/theme---ballad': 34,
        'mood/theme---soundscape': 35,
        'mood/theme---dream': 36,
        'mood/theme---movie': 37,
        'mood/theme---fast': 38,
        'mood/theme---nature': 39,
        'mood/theme---cool': 40,
        'mood/theme---corporate': 41,
        'mood/theme---travel': 42,
        'mood/theme---funny': 43,
        'mood/theme---sport': 44,
        'mood/theme---commercial': 45,
        'mood/theme---advertising': 46,
        'mood/theme---holiday': 47,
        'mood/theme---christmas': 48,
        'mood/theme---sexy': 49,
        'mood/theme---game': 50,
        'mood/theme---groovy': 51,
        'mood/theme---retro': 52,
        'mood/theme---summer': 53,
        'mood/theme---party': 54,
        'mood/theme---trailer': 55
        }
    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

    EXAMPLE_JSONL = {
        "audio_path": "data/MTG/audio/48/948.low.flac", 
        "label": ["mood/theme---background"], 
        "duration": 212.66666666666666, 
        "sample_rate": 44100, 
        "num_samples": 9378600, 
        "bit_depth": 16, 
        "channels": 1
        }

    def __init__(self, jsonl: str, sample_rate: int, channels: int,
                 clip_seconds: float, channel_mode: str="first",
                 min_clip_ratio: float=1.0, backend: Optional[str] = None,
                 clips_per_file: Optional[int] = None,
                 clip_selection_seed: Optional[int] = None):
        super().__init__(
            jsonl=jsonl,
            sample_rate=sample_rate,
            channels=channels,
            clip_seconds=clip_seconds,
            channel_mode=channel_mode,
            min_clip_ratio=min_clip_ratio,
            backend=backend,
            clips_per_file=clips_per_file,
            clip_selection_seed=clip_selection_seed,
        )

    def get_targets(self, file_idx: int, slice_idx: int, orig_sr: int, orig_clip_frames: int):
        info = self.meta[file_idx]
        label_indices = [self.LABEL2IDX[tag] for tag in info['label']]
        num_labels = len(self.LABEL2IDX)
        label = torch.zeros(num_labels, dtype=torch.int)
        label[label_indices] = 1
        return label


class MTGMoodAudioTrain(_MTGMoodAudioBase):
    pass


class MTGMoodAudioVal(_MTGMoodAudioBase):
    pass


class MTGMoodAudioTest(MTGMoodAudioVal):
    pass


class MTGMoodDataModule(BaseDataModule):
    pass


class MTGMoodEmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted MTGMood embeddings.
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
        from marble.tasks.MTGMood.embedding_dataset import MTGMoodEmbeddingDataset
        self._dataset_cls = MTGMoodEmbeddingDataset

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

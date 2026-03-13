# marble/tasks/ChordsACE/datamodule.py

from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from marble.tasks.ChordsACE.embedding_dataset import ChordsACEEmbeddingDataset


class ChordsACEEmbeddingDataModule(pl.LightningDataModule):
    """DataModule for decomposed chord probing on pre-extracted embeddings."""

    def __init__(
        self,
        embedding_root: str,
        layer_idx: int,
        train_jsonl: str,
        val_jsonl: str,
        test_jsonl: str,
        clip_seconds: float = 15.0,
        label_freq: int = 25,
        min_clip_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.embedding_root = Path(embedding_root)
        self.layer_idx = layer_idx
        self.train_jsonl = train_jsonl
        self.val_jsonl = val_jsonl
        self.test_jsonl = test_jsonl
        self.clip_seconds = clip_seconds
        self.label_freq = label_freq
        self.min_clip_ratio = min_clip_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        common = dict(
            layer_idx=self.layer_idx,
            clip_seconds=self.clip_seconds,
            label_freq=self.label_freq,
            min_clip_ratio=self.min_clip_ratio,
        )
        if stage in (None, "fit"):
            self.train_dataset = ChordsACEEmbeddingDataset(
                self.embedding_root / "train", jsonl=self.train_jsonl, **common
            )
            self.val_dataset = ChordsACEEmbeddingDataset(
                self.embedding_root / "val", jsonl=self.val_jsonl, **common
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = ChordsACEEmbeddingDataset(
                self.embedding_root / "test", jsonl=self.test_jsonl, **common
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

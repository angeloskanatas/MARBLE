# marble/tasks/MTGInstrument/datamodule.py
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule, BaseAudioDataset


class _MTGInstrumentAudioBase(BaseAudioDataset):
    LABEL2IDX = {
        'instrument---voice': 0,
        'instrument---synthesizer': 1,
        'instrument---piano': 2,
        'instrument---strings': 3,
        'instrument---beat': 4,
        'instrument---guitar': 5,
        'instrument---cello': 6,
        'instrument---keyboard': 7,
        'instrument---trombone': 8,
        'instrument---clarinet': 9,
        'instrument---doublebass': 10,
        'instrument---horn': 11,
        'instrument---trumpet': 12,
        'instrument---violin': 13,
        'instrument---accordion': 14,
        'instrument---bass': 15,
        'instrument---computer': 16,
        'instrument---drummachine': 17,
        'instrument---drums': 18,
        'instrument---electricguitar': 19,
        'instrument---sampler': 20,
        'instrument---acousticguitar': 21,
        'instrument---harmonica': 22,
        'instrument---flute': 23,
        'instrument---pipeorgan': 24,
        'instrument---harp': 25,
        'instrument---electricpiano': 26,
        'instrument---oboe': 27,
        'instrument---saxophone': 28,
        'instrument---percussion': 29,
        'instrument---acousticbassguitar': 30,
        'instrument---orchestra': 31,
        'instrument---bongo': 32,
        'instrument---brass': 33,
        'instrument---viola': 34,
        'instrument---rhodes': 35,
        'instrument---organ': 36,
        'instrument---classicalguitar': 37,
        'instrument---bell': 38,
        'instrument---pad': 39
        }
    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

    EXAMPLE_JSONL = {
        "audio_path": "data/MTG/audio/82/382.low.flac", 
        "label": ["instrument---voice"], 
        "duration": 211.04, 
        "sample_rate": 44100, 
        "num_samples": 9306864, 
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


class MTGInstrumentAudioTrain(_MTGInstrumentAudioBase):
    pass


class MTGInstrumentAudioVal(_MTGInstrumentAudioBase):
    pass


class MTGInstrumentAudioTest(MTGInstrumentAudioVal):
    pass


class MTGInstrumentDataModule(BaseDataModule):
    pass


class MTGInstrumentMultiLayerEmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted MTGInstrument embeddings with MULTIPLE layers.
    Returns batches of shape (B, L, H) instead of (B, H).
    """

    def __init__(
        self,
        embedding_root: str,
        layer_indices: list,
        train_jsonl: str,
        val_jsonl: str,
        test_jsonl: str,
        batch_size: int = 16,
        num_workers: int = 8,
    ):
        super().__init__()
        from marble.tasks.MTGInstrument.embedding_dataset import MTGInstrumentMultiLayerEmbeddingDataset
        self._dataset_cls = MTGInstrumentMultiLayerEmbeddingDataset

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


class MTGInstrumentEmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted MTGInstrument embeddings.
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
        from marble.tasks.MTGInstrument.embedding_dataset import MTGInstrumentEmbeddingDataset
        self._dataset_cls = MTGInstrumentEmbeddingDataset

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

# marble/tasks/MTGGenre/datamodule.py
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule, BaseAudioDataset


class _MTGGenreAudioBase(BaseAudioDataset):
    LABEL2IDX = {
        'genre---rock': 0,
        'genre---pop': 1,
        'genre---classical': 2,
        'genre---popfolk': 3,
        'genre---disco': 4,
        'genre---funk': 5,
        'genre---rnb': 6,
        'genre---ambient': 7,
        'genre---chillout': 8,
        'genre---downtempo': 9,
        'genre---easylistening': 10,
        'genre---electronic': 11,
        'genre---lounge': 12,
        'genre---triphop': 13,
        'genre---breakbeat': 14,
        'genre---techno': 15,
        'genre---newage': 16,
        'genre---jazz': 17,
        'genre---metal': 18,
        'genre---industrial': 19,
        'genre---instrumentalrock': 20,
        'genre---minimal': 21,
        'genre---alternative': 22,
        'genre---experimental': 23,
        'genre---drumnbass': 24,
        'genre---soul': 25,
        'genre---fusion': 26,
        'genre---soundtrack': 27,
        'genre---electropop': 28,
        'genre---world': 29,
        'genre---ethno': 30,
        'genre---trance': 31,
        'genre---orchestral': 32,
        'genre---grunge': 33,
        'genre---chanson': 34,
        'genre---worldfusion': 35,
        'genre---hiphop': 36,
        'genre---groove': 37,
        'genre---instrumentalpop': 38,
        'genre---blues': 39,
        'genre---reggae': 40,
        'genre---dance': 41,
        'genre---club': 42,
        'genre---punkrock': 43,
        'genre---folk': 44,
        'genre---synthpop': 45,
        'genre---poprock': 46,
        'genre---choir': 47,
        'genre---symphonic': 48,
        'genre---indie': 49,
        'genre---progressive': 50,
        'genre---acidjazz': 51,
        'genre---contemporary': 52,
        'genre---newwave': 53,
        'genre---dub': 54,
        'genre---rocknroll': 55,
        'genre---hard': 56,
        'genre---hardrock': 57,
        'genre---house': 58,
        'genre---atmospheric': 59,
        'genre---psychedelic': 60,
        'genre---improvisation': 61,
        'genre---country': 62,
        'genre---electronica': 63,
        'genre---rap': 64,
        'genre---60s': 65,
        'genre---70s': 66,
        'genre---darkambient': 67,
        'genre---idm': 68,
        'genre---latin': 69,
        'genre---postrock': 70,
        'genre---bossanova': 71,
        'genre---singersongwriter': 72,
        'genre---darkwave': 73,
        'genre---swing': 74,
        'genre---medieval': 75,
        'genre---celtic': 76,
        'genre---eurodance': 77,
        'genre---classicrock': 78,
        'genre---dubstep': 79,
        'genre---bluesrock': 80,
        'genre---edm': 81,
        'genre---deephouse': 82,
        'genre---jazzfusion': 83,
        'genre---alternativerock': 84,
        'genre---80s': 85,
        'genre---90s': 86
    }
    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

    EXAMPLE_JSONL = {
        "audio_path": "data/MTG/audio/41/241.low.flac", 
        "label": ["genre---rock"], 
        "duration": 340.1066666666667, 
        "sample_rate": 44100, 
        "num_samples": 14998704, 
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


class MTGGenreAudioTrain(_MTGGenreAudioBase):
    pass


class MTGGenreAudioVal(_MTGGenreAudioBase):
    pass


class MTGGenreAudioTest(MTGGenreAudioVal):
    pass


class MTGGenreDataModule(BaseDataModule):
    pass


class MTGGenreMultiLayerEmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted MTGGenre embeddings with MULTIPLE layers.
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
        from marble.tasks.MTGGenre.embedding_dataset import MTGGenreMultiLayerEmbeddingDataset
        self._dataset_cls = MTGGenreMultiLayerEmbeddingDataset

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


class MTGGenreEmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted MTGGenre embeddings.
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
        from marble.tasks.MTGGenre.embedding_dataset import MTGGenreEmbeddingDataset
        self._dataset_cls = MTGGenreEmbeddingDataset

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

# marble/tasks/MTGTop50/datamodule.py
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule, BaseAudioDataset


class _MTGTop50AudioBase(BaseAudioDataset):
    LABEL2IDX = {
        'genre---rock': 0,
        'genre---pop': 1,
        'genre---classical': 2,
        'instrument---voice': 3,
        'genre---popfolk': 4,
        'genre---funk': 5,
        'genre---ambient': 6,
        'genre---chillout': 7,
        'genre---downtempo': 8,
        'genre---easylistening': 9,
        'genre---electronic': 10,
        'genre---lounge': 11,
        'instrument---synthesizer': 12,
        'genre---triphop': 13,
        'genre---techno': 14,
        'genre---newage': 15,
        'genre---jazz': 16,
        'genre---metal': 17,
        'instrument---piano': 18,
        'genre---alternative': 19,
        'genre---experimental': 20,
        'genre---soundtrack': 21,
        'mood/theme---film': 22,
        'genre---world': 23,
        'instrument---strings': 24,
        'genre---trance': 25,
        'genre---orchestral': 26,
        'instrument---guitar': 27,
        'genre---hiphop': 28,
        'genre---instrumentalpop': 29,
        'mood/theme---relaxing': 30,
        'genre---reggae': 31,
        'mood/theme---emotional': 32,
        'instrument---keyboard': 33,
        'instrument---violin': 34,
        'genre---dance': 35,
        'instrument---bass': 36,
        'instrument---computer': 37,
        'instrument---drummachine': 38,
        'instrument---drums': 39,
        'instrument---electricguitar': 40,
        'genre---folk': 41,
        'instrument---acousticguitar': 42,
        'genre---poprock': 43,
        'genre---indie': 44,
        'mood/theme---energetic': 45,
        'mood/theme---happy': 46,
        'instrument---electricpiano': 47,
        'genre---house': 48,
        'genre---atmospheric': 49
    }
    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

    EXAMPLE_JSONL = {
        "audio_path": "data/MTG/audio-low/41/241.low.mp3", 
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


class MTGTop50AudioTrain(_MTGTop50AudioBase):
    pass


class MTGTop50AudioVal(_MTGTop50AudioBase):
    pass


class MTGTop50AudioTest(MTGTop50AudioVal):
    pass


class MTGTop50DataModule(BaseDataModule):
    pass


class MTGTop50EmbeddingDataModule(pl.LightningDataModule):
    """
    DataModule for probing on pre-extracted MTGTop50 embeddings.
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
        from marble.tasks.MTGTop50.embedding_dataset import MTGTop50EmbeddingDataset
        self._dataset_cls = MTGTop50EmbeddingDataset

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

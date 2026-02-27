# marble/tasks/GS/embedding_dataset.py

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from marble.tasks.GS.datamodule import _GSAudioBase


class GSEmbeddingDataset(Dataset):
    """
    Dataset that loads pre-extracted sequence-level embeddings for one layer.
    Expects extraction output layout: embedding_dir/layer{N}/sequence-level/embeddings.dat
    and embedding_dir/sample_to_audio_path.json. Labels are resolved via jsonl (path -> label).
    """

    LABEL2IDX = _GSAudioBase.LABEL2IDX
    NUM_LABELS = len(LABEL2IDX)

    def __init__(
        self,
        embedding_dir: str,
        layer_idx: int,
        jsonl: str,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.layer_idx = layer_idx
        with open(jsonl, "r") as f:
            self.meta = [json.loads(line) for line in f]
        # Keyed by ori_uid because GS __getitem__ returns ori_uid (not audio_path)
        # as the path element, so sample_to_audio_path.json contains ori_uids
        self._uid_to_label = {
            info["ori_uid"]: self.LABEL2IDX[info["label"]]
            for info in self.meta
        }
        mapping_path = self.embedding_dir / "sample_to_audio_path.json"
        with open(mapping_path, "r") as f:
            data = json.load(f)
        self._sample_to_audio_path: List[str] = data["sample_to_audio_path"]
        layer_dir = self.embedding_dir / f"layer{layer_idx}" / "sequence-level"
        meta_path = layer_dir / "metadata.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        shape = tuple(meta["shape"])
        dtype = np.dtype(meta["dtype"])
        self._memmap = np.memmap(
            layer_dir / "embeddings.dat",
            dtype=dtype,
            mode="r",
            shape=shape,
        )
        self._labels = [
            self._uid_to_label[uid]
            for uid in self._sample_to_audio_path
        ]
        self._ori_uids = list(self._sample_to_audio_path)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        emb = np.copy(self._memmap[idx])
        label = self._labels[idx]
        ori_uid = self._ori_uids[idx]
        return torch.from_numpy(emb).float(), label, ori_uid

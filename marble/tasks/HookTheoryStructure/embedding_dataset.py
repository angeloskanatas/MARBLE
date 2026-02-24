# marble/tasks/HookTheoryStructure/embedding_dataset.py

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from marble.tasks.HookTheoryStructure.datamodule import _HookTheoryStructureAudioBase


class HookTheoryStructureEmbeddingDataset(Dataset):
    """
    Dataset that loads pre-extracted sequence-level embeddings for one layer.
    Expects extraction output layout: embedding_dir/layer{N}/sequence-level/embeddings.dat
    and embedding_dir/sample_to_audio_path.json. Labels are resolved via jsonl (path -> label).
    """

    LABEL2IDX = _HookTheoryStructureAudioBase.LABEL2IDX
    IDX2LABEL = _HookTheoryStructureAudioBase.IDX2LABEL
    NUM_LABELS = len(IDX2LABEL)  # 7 unique classes

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
        # Process labels: join list labels with '_' (matching 1-stage behaviour)
        self._path_to_label = {}
        for info in self.meta:
            label_raw = info["label"]
            if isinstance(label_raw, list):
                label_str = "_".join(label_raw)
            else:
                label_str = label_raw
            self._path_to_label[info["audio_path"]] = self.LABEL2IDX[label_str]
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
            self._path_to_label[path]
            for path in self._sample_to_audio_path
        ]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        emb = np.copy(self._memmap[idx])
        label = self._labels[idx]
        path = self._sample_to_audio_path[idx]
        return torch.from_numpy(emb).float(), label, path

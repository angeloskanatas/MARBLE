# marble/tasks/HXMSA/embedding_dataset.py

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL2IDX = {
    "intro": 0,
    "verse": 1,
    "prechorus": 2,
    "chorus": 3,
    "bridge": 4,
    "outro": 5,
    "inst": 6,
    "other": 7,
}

IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

NUM_CLASSES = len(IDX2LABEL)  # 8


class HXMSAEmbeddingDataset(Dataset):
    """
    Dataset that loads pre-extracted sequence-level embeddings for one layer.

    Expects extraction output layout:
        embedding_dir/layer{N}/sequence-level/embeddings.dat
        embedding_dir/layer{N}/sequence-level/metadata.json
        embedding_dir/sample_to_audio_path.json

    The sample_to_audio_path stores composite keys of the form
    "track_id||segment_start", which uniquely identify each segment.
    Labels are resolved via this key using the JSONL metadata.
    """

    LABEL2IDX = LABEL2IDX
    IDX2LABEL = IDX2LABEL
    NUM_CLASSES = NUM_CLASSES

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

        # Build segment_key → label mapping.
        # Key format: "track_id||segment_start" (matches datamodule output).
        self._key_to_label = {}
        for info in self.meta:
            key = f"{info['track_id']}||{info['segment_start']}"
            self._key_to_label[key] = self.LABEL2IDX[info["label"]]

        # Load sample-to-audio-path mapping (stores composite keys, not audio paths)
        mapping_path = self.embedding_dir / "sample_to_audio_path.json"
        with open(mapping_path, "r") as f:
            data = json.load(f)
        self._sample_to_key: List[str] = data["sample_to_audio_path"]

        # Load embeddings (memory-mapped)
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

        # Resolve labels for each sample
        self._labels = [
            self._key_to_label[key]
            for key in self._sample_to_key
        ]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        emb = np.copy(self._memmap[idx])
        label = self._labels[idx]
        key = self._sample_to_key[idx]
        return torch.from_numpy(emb).float(), label, key

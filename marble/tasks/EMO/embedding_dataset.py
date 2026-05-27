import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class EMOEmbeddingDataset(Dataset):
    """
    Dataset that loads pre-extracted sequence-level embeddings for one layer.
    Expects extraction output layout:
      embedding_dir/layer{N}/sequence-level/embeddings.dat
      embedding_dir/sample_to_audio_path.json
    Labels are resolved via jsonl (path -> [arousal, valence]).
    """

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

        self._path_to_label = {
            info["audio_path"]: np.array(info["label"], dtype=np.float32)
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
            self._path_to_label[path]
            for path in self._sample_to_audio_path
        ]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        emb = np.copy(self._memmap[idx])
        label = self._labels[idx]
        path = self._sample_to_audio_path[idx]
        return torch.from_numpy(emb).float(), torch.from_numpy(label).float(), path


class EMOMultiLayerEmbeddingDataset(Dataset):
    """
    Dataset that loads pre-extracted sequence-level embeddings for MULTIPLE layers.
    Returns stacked embeddings of shape (num_layers, hidden_dim).
    """

    def __init__(
        self,
        embedding_dir: str,
        layer_indices: List[int],
        jsonl: str,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.layer_indices = layer_indices
        with open(jsonl, "r") as f:
            self.meta = [json.loads(line) for line in f]
        self._path_to_label = {
            info["audio_path"]: np.array(info["label"], dtype=np.float32)
            for info in self.meta
        }
        mapping_path = self.embedding_dir / "sample_to_audio_path.json"
        with open(mapping_path, "r") as f:
            data = json.load(f)
        self._sample_to_audio_path: List[str] = data["sample_to_audio_path"]
        self._labels = [
            self._path_to_label[path]
            for path in self._sample_to_audio_path
        ]
        # Load memmaps for all specified layers
        self._memmaps = []
        for l in layer_indices:
            layer_dir = self.embedding_dir / f"layer{l}" / "sequence-level"
            meta_path = layer_dir / "metadata.json"
            with open(meta_path, "r") as f:
                meta = json.load(f)
            shape = tuple(meta["shape"])
            dtype = np.dtype(meta["dtype"])
            mm = np.memmap(
                layer_dir / "embeddings.dat",
                dtype=dtype,
                mode="r",
                shape=shape,
            )
            self._memmaps.append(mm)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        embs = [torch.from_numpy(np.copy(mm[idx])).float() for mm in self._memmaps]
        label = self._labels[idx]
        path = self._sample_to_audio_path[idx]
        return torch.stack(embs, dim=0), torch.from_numpy(label).float(), path  # (L, H)

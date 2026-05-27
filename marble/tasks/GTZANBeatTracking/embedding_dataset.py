import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from marble.utils.utils import widen_temporal_events


class GTZANBeatTrackingEmbeddingDataset(Dataset):
    """
    Dataset for probing on pre-extracted frame-level GTZAN beat-tracking embeddings.
    
    Expects extraction output layout:
      embedding_dir/layer{N}/frame-level/*.npy
    """

    def __init__(
        self,
        embedding_dir: str,
        layer_idx: int,
        jsonl: str,
        clip_seconds: float,
        label_freq: int,
        num_neighbors: int = 0,
        use_local_bpm: bool = True,
        min_clip_ratio: float = 0.8,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.layer_idx = layer_idx
        self.clip_seconds = clip_seconds
        self.label_freq = label_freq
        self.num_neighbors = num_neighbors
        self.use_local_bpm = use_local_bpm
        self.min_clip_ratio = min_clip_ratio

        with open(jsonl, "r") as f:
            self.meta = [json.loads(line) for line in f]

        self.beat_times_meta: List[np.ndarray] = []
        self.db_times_meta: List[np.ndarray] = []
        self.tempo_list: List[float] = []
        for info in self.meta:
            label = info["label"]
            self.beat_times_meta.append(np.array(label.get("beat", []), dtype=np.float32))
            self.db_times_meta.append(np.array(label.get("downbeat", []), dtype=np.float32))
            self.tempo_list.append(float(label.get("tempo", 0.0)))

        self.index_map: List[Tuple[int, int, int, int]] = []
        for file_idx, info in enumerate(self.meta):
            orig_sr = int(info["sample_rate"])
            total_samples = int(info["num_samples"])
            orig_clip_frames = int(self.clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue

            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full

            for slice_idx in range(n_slices):
                self.index_map.append((file_idx, slice_idx, orig_sr, orig_clip_frames))

        frame_dir = self.embedding_dir / f"layer{layer_idx}" / "frame-level"
        files = sorted(frame_dir.glob("*.npy"))
        pattern = re.compile(r"_(\d{6})(?:_aug\d+)?\.npy$")
        idx_to_file: dict[int, Path] = {}
        for p in files:
            m = pattern.search(p.name)
            if m is None:
                continue
            idx_to_file[int(m.group(1))] = p

        max_idx = len(self.index_map) - 1
        self.sample_indices = sorted(i for i in idx_to_file.keys() if 0 <= i <= max_idx)
        self._idx_to_file = idx_to_file

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int):
        sample_idx = self.sample_indices[idx]
        file_path = self._idx_to_file[sample_idx]
        emb_np = np.load(file_path)
        emb = torch.from_numpy(emb_np).float()
        if emb.ndim == 2:  # (T, H) -> (L=1, T, H)
            emb = emb.unsqueeze(0)
        elif emb.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected embedding shape: {emb.shape} in {file_path}")

        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[sample_idx]
        info = self.meta[file_idx]
        audio_path = info["audio_path"]

        clip_start_time = slice_idx * (orig_clip_frames / orig_sr)
        clip_end_time = clip_start_time + self.clip_seconds
        label_len = int(self.label_freq * self.clip_seconds)

        beat_mask = np.zeros(label_len, dtype=np.float32)
        db_mask = np.zeros(label_len, dtype=np.float32)

        for t in self.beat_times_meta[file_idx]:
            if clip_start_time <= t < clip_end_time:
                rel = t - clip_start_time
                frame_idx = int(round(rel * self.label_freq))
                if 0 <= frame_idx < label_len:
                    beat_mask[frame_idx] = 1.0

        for t in self.db_times_meta[file_idx]:
            if clip_start_time <= t < clip_end_time:
                rel = t - clip_start_time
                frame_idx = int(round(rel * self.label_freq))
                if 0 <= frame_idx < label_len:
                    db_mask[frame_idx] = 1.0

        if self.use_local_bpm:
            est_tempo = beat_mask.sum() / self.clip_seconds * 60.0
        else:
            est_tempo = self.tempo_list[file_idx]

        if self.num_neighbors > 0:
            beat_mask = widen_temporal_events(beat_mask, self.num_neighbors)
            db_mask = widen_temporal_events(db_mask, self.num_neighbors)

        targets = {
            "beat": torch.from_numpy(beat_mask),
            "downbeat": torch.from_numpy(db_mask),
            "tempo": torch.tensor(est_tempo, dtype=torch.float32),
        }
        return emb, targets, audio_path


class GTZANBeatTrackingMultiLayerEmbeddingDataset(Dataset):
    """
    Dataset for probing on pre-extracted frame-level GTZAN beat-tracking embeddings
    with MULTIPLE layers. Returns embeddings of shape (L, T, H) instead of (1, T, H).

    Expects extraction output layout:
      embedding_dir/layer{N}/frame-level/*.npy
    """

    def __init__(
        self,
        embedding_dir: str,
        layer_indices: List[int],
        jsonl: str,
        clip_seconds: float,
        label_freq: int,
        num_neighbors: int = 0,
        use_local_bpm: bool = True,
        min_clip_ratio: float = 0.8,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.layer_indices = layer_indices
        self.clip_seconds = clip_seconds
        self.label_freq = label_freq
        self.num_neighbors = num_neighbors
        self.use_local_bpm = use_local_bpm
        self.min_clip_ratio = min_clip_ratio

        with open(jsonl, "r") as f:
            self.meta = [json.loads(line) for line in f]

        self.beat_times_meta: List[np.ndarray] = []
        self.db_times_meta: List[np.ndarray] = []
        self.tempo_list: List[float] = []
        for info in self.meta:
            label = info["label"]
            self.beat_times_meta.append(np.array(label.get("beat", []), dtype=np.float32))
            self.db_times_meta.append(np.array(label.get("downbeat", []), dtype=np.float32))
            self.tempo_list.append(float(label.get("tempo", 0.0)))

        self.index_map: List[Tuple[int, int, int, int]] = []
        for file_idx, info in enumerate(self.meta):
            orig_sr = int(info["sample_rate"])
            total_samples = int(info["num_samples"])
            orig_clip_frames = int(self.clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue

            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full

            for slice_idx in range(n_slices):
                self.index_map.append((file_idx, slice_idx, orig_sr, orig_clip_frames))

        # Discover files from first layer, then use same filenames for all layers
        ref_layer = layer_indices[0]
        ref_dir = self.embedding_dir / f"layer{ref_layer}" / "frame-level"
        files = sorted(ref_dir.glob("*.npy"))
        pattern = re.compile(r"_(\d{6})(?:_aug\d+)?\.npy$")
        idx_to_filename: dict[int, str] = {}
        for p in files:
            m = pattern.search(p.name)
            if m is None:
                continue
            idx_to_filename[int(m.group(1))] = p.name

        max_idx = len(self.index_map) - 1
        self.sample_indices = sorted(i for i in idx_to_filename.keys() if 0 <= i <= max_idx)
        self._idx_to_filename = idx_to_filename

        # Pre-build frame-level directories for all layers
        self._frame_dirs = [
            self.embedding_dir / f"layer{l}" / "frame-level"
            for l in layer_indices
        ]

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int):
        sample_idx = self.sample_indices[idx]
        filename = self._idx_to_filename[sample_idx]

        # Load from all layers and stack: (L, T, H)
        embs = []
        for frame_dir in self._frame_dirs:
            emb_np = np.load(frame_dir / filename)
            embs.append(torch.from_numpy(emb_np).float())  # (T, H)
        emb = torch.stack(embs, dim=0)  # (L, T, H)

        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[sample_idx]
        info = self.meta[file_idx]
        audio_path = info["audio_path"]

        clip_start_time = slice_idx * (orig_clip_frames / orig_sr)
        clip_end_time = clip_start_time + self.clip_seconds
        label_len = int(self.label_freq * self.clip_seconds)

        beat_mask = np.zeros(label_len, dtype=np.float32)
        db_mask = np.zeros(label_len, dtype=np.float32)

        for t in self.beat_times_meta[file_idx]:
            if clip_start_time <= t < clip_end_time:
                rel = t - clip_start_time
                frame_idx = int(round(rel * self.label_freq))
                if 0 <= frame_idx < label_len:
                    beat_mask[frame_idx] = 1.0

        for t in self.db_times_meta[file_idx]:
            if clip_start_time <= t < clip_end_time:
                rel = t - clip_start_time
                frame_idx = int(round(rel * self.label_freq))
                if 0 <= frame_idx < label_len:
                    db_mask[frame_idx] = 1.0

        if self.use_local_bpm:
            est_tempo = beat_mask.sum() / self.clip_seconds * 60.0
        else:
            est_tempo = self.tempo_list[file_idx]

        if self.num_neighbors > 0:
            beat_mask = widen_temporal_events(beat_mask, self.num_neighbors)
            db_mask = widen_temporal_events(db_mask, self.num_neighbors)

        targets = {
            "beat": torch.from_numpy(beat_mask),
            "downbeat": torch.from_numpy(db_mask),
            "tempo": torch.tensor(est_tempo, dtype=torch.float32),
        }
        return emb, targets, audio_path

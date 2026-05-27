# marble/tasks/HXMSAStructure/embedding_dataset.py

import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from marble.tasks.HXMSA.embedding_dataset import normalize_label, LABEL2IDX, IDX2LABEL

NUM_CLASSES = len(IDX2LABEL)  # 6


class HXMSAStructureEmbeddingDataset(Dataset):
    """
    Frame-level structure segmentation on pre-extracted embeddings.

    Expects per-track JSONL with "segments" field listing all boundaries.
    Embeddings are frame-level .npy files (one per clip) in:
        embedding_dir/layer{N}/frame-level/*.npy

    For each clip, generates:
        - frame_labels: (T,) int tensor — section class per frame
        - boundaries: (T,) float tensor — 1.0 at boundary frames
    """

    LABEL2IDX = LABEL2IDX
    IDX2LABEL = IDX2LABEL
    NUM_CLASSES = NUM_CLASSES

    def __init__(
        self,
        embedding_dir: str,
        layer_idx: int,
        jsonl: str,
        clip_seconds: float,
        label_freq: int,
        boundary_num_neighbors: int = 2,
        min_clip_ratio: float = 0.5,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.layer_idx = layer_idx
        self.clip_seconds = clip_seconds
        self.label_freq = label_freq
        self.boundary_num_neighbors = boundary_num_neighbors
        self.min_clip_ratio = min_clip_ratio

        with open(jsonl, "r") as f:
            self.meta = [json.loads(line) for line in f]

        # Pre-parse segments for each track
        self.track_segments: List[List[dict]] = []
        for info in self.meta:
            segs = []
            for s in info["segments"]:
                segs.append({
                    "start": s["start"],
                    "end": s["end"],
                    "label_idx": self.LABEL2IDX[normalize_label(s["label"])],
                })
            self.track_segments.append(segs)

        # Build index_map: (track_idx, slice_idx, orig_sr, orig_clip_frames)
        self.index_map: List[Tuple[int, int, int, int]] = []
        for track_idx, info in enumerate(self.meta):
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
                self.index_map.append((track_idx, slice_idx, orig_sr, orig_clip_frames))

        # Map .npy files to indices
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

    def _generate_frame_labels(
        self, segments: List[dict], clip_start: float, clip_end: float, label_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate per-frame section labels and boundary mask for a clip.

        Returns:
            frame_labels: (label_len,) int32 — section class at each frame
            boundaries:   (label_len,) float32 — 1.0 at boundary frames
        """
        frame_labels = np.full(label_len, -1, dtype=np.int32)
        boundaries = np.zeros(label_len, dtype=np.float32)

        # Assign section labels to each frame
        for seg in segments:
            if seg["end"] <= clip_start or seg["start"] >= clip_end:
                continue
            # Frame range covered by this segment within the clip
            rel_start = max(0.0, seg["start"] - clip_start)
            rel_end = min(clip_end - clip_start, seg["end"] - clip_start)
            frame_start = int(round(rel_start * self.label_freq))
            frame_end = int(round(rel_end * self.label_freq))
            frame_start = max(0, min(frame_start, label_len))
            frame_end = max(0, min(frame_end, label_len))
            frame_labels[frame_start:frame_end] = seg["label_idx"]

        # Mark boundaries: where segment transitions occur within the clip
        for seg in segments:
            t = seg["start"]
            if clip_start < t < clip_end:
                rel = t - clip_start
                frame_idx = int(round(rel * self.label_freq))
                if 0 <= frame_idx < label_len:
                    boundaries[frame_idx] = 1.0

        # Widen boundary events for soft BCE training targets.
        # Weights must stay <= 0.5 so only the center (1.0) passes the
        # metric's > 0.5 threshold — keeps evaluation clean.
        if self.boundary_num_neighbors > 0:
            widened = np.copy(boundaries)
            for k in range(1, self.boundary_num_neighbors + 1):
                w = 1.0 / (k + 1)  # k=1 → 0.5, k=2 → 0.333
                widened[k:] += boundaries[:-k] * w
                widened[:-k] += boundaries[k:] * w
            boundaries = np.clip(widened, 0.0, 1.0)

        # Fill unlabeled frames with nearest labeled frame's class
        if (frame_labels == -1).any():
            # Simple forward-fill then backward-fill
            for i in range(1, label_len):
                if frame_labels[i] == -1 and frame_labels[i - 1] != -1:
                    frame_labels[i] = frame_labels[i - 1]
            for i in range(label_len - 2, -1, -1):
                if frame_labels[i] == -1 and frame_labels[i + 1] != -1:
                    frame_labels[i] = frame_labels[i + 1]
            # If still unlabeled (no segments overlap this clip), default to "inst"
            frame_labels[frame_labels == -1] = self.LABEL2IDX["inst"]

        return frame_labels, boundaries

    def __getitem__(self, idx: int):
        sample_idx = self.sample_indices[idx]
        file_path = self._idx_to_file[sample_idx]
        emb_np = np.load(file_path)
        emb = torch.from_numpy(emb_np).float()
        if emb.ndim == 2:  # (T, H) -> (L=1, T, H)
            emb = emb.unsqueeze(0)
        elif emb.ndim == 3:  # (L, T, H) already
            pass
        else:
            raise ValueError(f"Unexpected embedding shape: {emb_np.shape} in {file_path}")

        track_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[sample_idx]
        info = self.meta[track_idx]
        segments = self.track_segments[track_idx]

        clip_start_time = slice_idx * (orig_clip_frames / orig_sr)
        clip_end_time = clip_start_time + self.clip_seconds
        label_len = int(self.label_freq * self.clip_seconds)

        frame_labels, boundaries = self._generate_frame_labels(
            segments, clip_start_time, clip_end_time, label_len
        )

        targets = {
            "frame_labels": torch.from_numpy(frame_labels).long(),
            "boundaries": torch.from_numpy(boundaries).float(),
        }
        return emb, targets, info["track_id"]

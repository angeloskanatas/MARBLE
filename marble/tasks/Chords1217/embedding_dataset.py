# marble/tasks/Chords1217/embedding_dataset.py

import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from marble.utils.utils import chord_to_majmin


class Chords1217EmbeddingDataset(Dataset):
    """
    Dataset for probing on pre-extracted frame-level Chords1217 embeddings.

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
        min_clip_ratio: float = 0.8,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.layer_idx = layer_idx
        self.clip_seconds = clip_seconds
        self.label_freq = label_freq
        self.min_clip_ratio = min_clip_ratio

        with open(jsonl, "r") as f:
            self.meta = [json.loads(line) for line in f]

        # Build per-file sorted chord annotations: list of (start_time, chord_idx)
        self.chords_meta: List[List[Tuple[float, int]]] = []
        for info in self.meta:
            ann_list = info.get("label", [])
            annotated: List[Tuple[float, int]] = []
            for seg in ann_list:
                t0 = float(seg["start_time"])
                idx = chord_to_majmin(seg["chord_str"])
                annotated.append((t0, idx))
            annotated.sort(key=lambda x: x[0])
            self.chords_meta.append(annotated)

        # Build index map: (file_idx, slice_idx, orig_sr, orig_clip_frames)
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

        # Load frame-level embedding file paths
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

    def _get_chord_targets(self, file_idx: int, slice_idx: int, orig_sr: int, orig_clip_frames: int):
        """
        Generate a frame-level chord label sequence for a single clip.
        Mirrors the logic in _Chords1217AudioBase.get_targets.
        """
        chord_ann = self.chords_meta[file_idx]

        clip_start = slice_idx * (orig_clip_frames / orig_sr)
        clip_end = clip_start + self.clip_seconds

        label_len = int(self.label_freq * self.clip_seconds)
        chord_seq = np.zeros(label_len, dtype=np.int64)

        if not chord_ann:
            chord_seq[:] = 24
            return torch.from_numpy(chord_seq)

        # Collect segments that start inside the clip
        segments: List[Tuple[float, int]] = []
        for t0, cidx in chord_ann:
            if t0 >= clip_end:
                break
            if t0 >= clip_start:
                segments.append((t0, cidx))

        # If the first segment starts after clip_start, check for carryover
        if not segments or segments[0][0] > clip_start:
            prev_chord = None
            for t0, cidx in reversed(chord_ann):
                if t0 < clip_start:
                    prev_chord = cidx
                    break
            start_label = prev_chord if prev_chord is not None else 24
            segments.insert(0, (clip_start, start_label))

        # Append sentinel at clip_end
        segments.append((clip_end, 24))

        # Assign labels per frame
        seg_ptr = 0
        current_label = segments[0][1]
        next_change = segments[1][0]

        for i in range(label_len):
            t = clip_start + i / self.label_freq
            while t >= next_change and seg_ptr + 1 < len(segments) - 1:
                seg_ptr += 1
                current_label = segments[seg_ptr][1]
                next_change = segments[seg_ptr + 1][0] if seg_ptr + 1 < len(segments) else clip_end

            if current_label == 24 and seg_ptr > 0:
                current_label = segments[seg_ptr - 1][1]

            chord_seq[i] = current_label

        return torch.from_numpy(chord_seq)

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

        targets = self._get_chord_targets(file_idx, slice_idx, orig_sr, orig_clip_frames)
        return emb, targets, audio_path

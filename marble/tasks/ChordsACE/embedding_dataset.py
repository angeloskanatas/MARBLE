# marble/tasks/ChordsACE/embedding_dataset.py
"""
Frame-level embedding dataset with decomposed chord targets
(root, bass, tone activations) for ACE-style probing.

Uses the same JSONL data and extracted embeddings as Chords1217.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple

import mir_eval
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_ROOT_CLASSES = 13   # 0-11 pitch classes + 12 for N
NUM_BASS_CLASSES = 13
NUM_TONE_CLASSES = 12   # 12 semitone activation (binary)


def encode_chord(chord_str: str) -> Tuple[int, int, np.ndarray]:
    """Encode a chord string into (root, bass, tones_bitmap).

    Uses mir_eval.chord.encode which returns absolute pitch classes.

    Returns:
        root: 0-11 for C-B, 12 for N/invalid
        bass: 0-11 for C-B, 12 for N/invalid
        tones: float32 array of shape (12,) with binary activations
    """
    root, bitmap, bass = mir_eval.chord.encode(chord_str)
    tones = bitmap[:12].astype(np.float32)
    if root < 0:
        return 12, 12, np.zeros(12, dtype=np.float32)
    bass_val = bass if bass >= 0 else 12
    return int(root), int(bass_val), tones


class ChordsACEEmbeddingDataset(Dataset):
    """
    Dataset for probing on pre-extracted frame-level embeddings
    with decomposed chord targets (root, bass, tone activations).

    Expects the same extraction layout as Chords1217:
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

        # Build per-file chord annotations: list of (start_time, root, bass, tones)
        self.chords_meta: List[List[Tuple[float, int, int, np.ndarray]]] = []
        for info in self.meta:
            ann_list = info.get("label", [])
            annotated: List[Tuple[float, int, int, np.ndarray]] = []
            for seg in ann_list:
                t0 = float(seg["start_time"])
                root, bass, tones = encode_chord(seg["chord_str"])
                annotated.append((t0, root, bass, tones))
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

    def _get_chord_targets(
        self, file_idx: int, slice_idx: int, orig_sr: int, orig_clip_frames: int
    ):
        """Generate decomposed frame-level chord targets for a single clip.

        Returns:
            root_seq: LongTensor (label_len,) with values 0-12
            bass_seq: LongTensor (label_len,) with values 0-12
            tones_seq: FloatTensor (label_len, 12) with binary activations
        """
        chord_ann = self.chords_meta[file_idx]
        clip_start = slice_idx * (orig_clip_frames / orig_sr)
        clip_end = clip_start + self.clip_seconds
        label_len = int(self.label_freq * self.clip_seconds)

        # Default: N chord
        root_seq = np.full(label_len, 12, dtype=np.int64)
        bass_seq = np.full(label_len, 12, dtype=np.int64)
        tones_seq = np.zeros((label_len, 12), dtype=np.float32)

        if not chord_ann:
            return (
                torch.from_numpy(root_seq),
                torch.from_numpy(bass_seq),
                torch.from_numpy(tones_seq),
            )

        # Collect segments that start inside the clip
        segments: List[Tuple[float, int, int, np.ndarray]] = []
        for t0, r, b, tn in chord_ann:
            if t0 >= clip_end:
                break
            if t0 >= clip_start:
                segments.append((t0, r, b, tn))

        # Handle carryover from before clip_start
        if not segments or segments[0][0] > clip_start:
            prev = None
            for t0, r, b, tn in reversed(chord_ann):
                if t0 < clip_start:
                    prev = (r, b, tn)
                    break
            if prev is not None:
                segments.insert(0, (clip_start, prev[0], prev[1], prev[2]))
            else:
                segments.insert(
                    0, (clip_start, 12, 12, np.zeros(12, dtype=np.float32))
                )

        # Append sentinel at clip_end (N chord)
        segments.append((clip_end, 12, 12, np.zeros(12, dtype=np.float32)))

        # Assign labels per frame
        seg_ptr = 0
        cur_r, cur_b, cur_tn = segments[0][1], segments[0][2], segments[0][3]
        next_change = segments[1][0]

        for i in range(label_len):
            t = clip_start + i / self.label_freq
            while t >= next_change and seg_ptr + 1 < len(segments) - 1:
                seg_ptr += 1
                cur_r = segments[seg_ptr][1]
                cur_b = segments[seg_ptr][2]
                cur_tn = segments[seg_ptr][3]
                next_change = (
                    segments[seg_ptr + 1][0]
                    if seg_ptr + 1 < len(segments)
                    else clip_end
                )

            # Inherit previous chord if current is N (same as Chords1217 logic)
            if cur_r == 12 and seg_ptr > 0:
                cur_r = segments[seg_ptr - 1][1]
                cur_b = segments[seg_ptr - 1][2]
                cur_tn = segments[seg_ptr - 1][3]

            root_seq[i] = cur_r
            bass_seq[i] = cur_b
            tones_seq[i] = cur_tn

        return (
            torch.from_numpy(root_seq),
            torch.from_numpy(bass_seq),
            torch.from_numpy(tones_seq),
        )

    def __getitem__(self, idx: int):
        sample_idx = self.sample_indices[idx]
        file_path = self._idx_to_file[sample_idx]
        emb_np = np.load(file_path)
        emb = torch.from_numpy(emb_np).float()
        if emb.ndim == 2:  # (T, H) -> (1, T, H)
            emb = emb.unsqueeze(0)

        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[sample_idx]
        info = self.meta[file_idx]
        audio_path = info["audio_path"]

        root_t, bass_t, tones_t = self._get_chord_targets(
            file_idx, slice_idx, orig_sr, orig_clip_frames
        )
        return emb, root_t, bass_t, tones_t, audio_path


class ChordsACEMultiLayerEmbeddingDataset(ChordsACEEmbeddingDataset):
    """
    Multi-layer variant: returns embeddings of shape (L, T, H) by stacking
    frame-level files from `layer{N}/frame-level/` for every N in `layer_indices`.
    """

    def __init__(
        self,
        embedding_dir: str,
        layer_indices: List[int],
        jsonl: str,
        clip_seconds: float,
        label_freq: int,
        min_clip_ratio: float = 0.8,
    ):
        # Initialise base class with the first layer (file discovery happens here)
        super().__init__(
            embedding_dir=embedding_dir,
            layer_idx=layer_indices[0],
            jsonl=jsonl,
            clip_seconds=clip_seconds,
            label_freq=label_freq,
            min_clip_ratio=min_clip_ratio,
        )
        self.layer_indices = layer_indices

        # Cache filenames keyed by sample_idx for cross-layer reuse
        self._idx_to_filename = {i: p.name for i, p in self._idx_to_file.items()}
        self._frame_dirs = [
            self.embedding_dir / f"layer{l}" / "frame-level"
            for l in layer_indices
        ]

    def __getitem__(self, idx: int):
        sample_idx = self.sample_indices[idx]
        filename = self._idx_to_filename[sample_idx]

        embs = []
        for frame_dir in self._frame_dirs:
            emb_np = np.load(frame_dir / filename)
            embs.append(torch.from_numpy(emb_np).float())  # (T, H)
        emb = torch.stack(embs, dim=0)  # (L, T, H)

        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[sample_idx]
        info = self.meta[file_idx]
        audio_path = info["audio_path"]

        root_t, bass_t, tones_t = self._get_chord_targets(
            file_idx, slice_idx, orig_sr, orig_clip_frames
        )
        return emb, root_t, bass_t, tones_t, audio_path

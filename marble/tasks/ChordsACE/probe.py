# marble/tasks/ChordsACE/probe.py
"""
Decomposed chord probe with three independent heads (root, bass, tone activations)
and mir_eval evaluation at test time.
"""

import logging
import warnings

import mir_eval
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from einops import reduce

from marble.tasks.ChordsACE.embedding_dataset import (
    NUM_ROOT_CLASSES,
    NUM_BASS_CLASSES,
    NUM_TONE_CLASSES,
)

warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for chord decoding
# ---------------------------------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

INTERVAL_MAP = {
    0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
    6: "b5", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7",
}

# Quality templates: frozenset of semitone offsets relative to root.
# Ordered from most specific (largest) to least specific for matching.
QUALITY_TEMPLATES = [
    ("maj7",    frozenset({0, 4, 7, 11})),
    ("min7",    frozenset({0, 3, 7, 10})),
    ("7",       frozenset({0, 4, 7, 10})),
    ("dim7",    frozenset({0, 3, 6, 9})),
    ("hdim7",   frozenset({0, 3, 6, 10})),
    ("minmaj7", frozenset({0, 3, 7, 11})),
    ("maj",     frozenset({0, 4, 7})),
    ("min",     frozenset({0, 3, 7})),
    ("dim",     frozenset({0, 3, 6})),
    ("aug",     frozenset({0, 4, 8})),
    ("sus4",    frozenset({0, 5, 7})),
    ("sus2",    frozenset({0, 2, 7})),
]


# ---------------------------------------------------------------------------
# Chord decoding (predictions -> mir_eval-parseable chord strings)
# ---------------------------------------------------------------------------

def decode_chord(
    root: int, bass: int, tones_bitmap: np.ndarray, threshold: float = 0.5
) -> str:
    """Decode (root, bass, tones_bitmap) into a mir_eval-parseable chord string.

    Args:
        root: pitch class 0-11, or 12 for N
        bass: pitch class 0-11, or 12 for N
        tones_bitmap: absolute 12-dim activation (not relative to root)
        threshold: activation threshold for tone presence

    Returns:
        Chord string like "C:maj7/b3" or "N"
    """
    if root >= 12 or root < 0:
        return "N"

    note = NOTE_NAMES[root]

    # Get active semitones relative to root
    relative = np.roll(tones_bitmap, -root)[:12]
    active = frozenset(i for i in range(12) if relative[i] > threshold)

    if len(active) == 0:
        return "N"

    # Auto-insert 5th if root + third present but no 5th (following ACE convention)
    has_third = (3 in active) or (4 in active)
    if 0 in active and has_third and 7 not in active:
        active = active | {7}

    # Match against quality templates (most specific first)
    quality = None
    for q_name, template in QUALITY_TEMPLATES:
        if template <= active:
            quality = q_name
            break

    if quality is None:
        # Fallback: use interval notation
        intervals = ",".join(INTERVAL_MAP[i] for i in sorted(active))
        quality = f"({intervals})"

    chord_str = f"{note}:{quality}"

    # Bass inversion
    if 0 <= bass < 12 and bass != root:
        interval = (bass - root) % 12
        chord_str += f"/{INTERVAL_MAP[interval]}"

    return chord_str


def _frames_to_intervals(labels: list[str], clip_seconds: float):
    """Group consecutive identical frame labels into (intervals, labels)."""
    if not labels:
        return np.empty((0, 2)), []

    T = len(labels)
    frame_dur = clip_seconds / T

    out_intervals = []
    out_labels = []
    start = 0
    current = labels[0]

    for i in range(1, T):
        if labels[i] != current:
            out_intervals.append((start * frame_dur, i * frame_dur))
            out_labels.append(current)
            start = i
            current = labels[i]

    out_intervals.append((start * frame_dur, clip_seconds))
    out_labels.append(current)

    return np.array(out_intervals), out_labels


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DecomposedChordLoss(nn.Module):
    """Multi-task loss: CE(root) + CE(bass) + BCE(tones)."""

    def __init__(
        self,
        root_weight: float = 1.0,
        bass_weight: float = 1.0,
        tones_weight: float = 2.0,
        time_dim_mismatch_tol: int = 5,
    ):
        super().__init__()
        self.root_weight = root_weight
        self.bass_weight = bass_weight
        self.tones_weight = tones_weight
        self.tol = time_dim_mismatch_tol
        self.root_ce = nn.CrossEntropyLoss()
        self.bass_ce = nn.CrossEntropyLoss()
        self.tones_bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        root_logits: torch.Tensor,   # (B, T, 13)
        bass_logits: torch.Tensor,   # (B, T, 13)
        tones_logits: torch.Tensor,  # (B, T, 12)
        root_t: torch.Tensor,        # (B, T')
        bass_t: torch.Tensor,        # (B, T')
        tones_t: torch.Tensor,       # (B, T', 12)
    ) -> torch.Tensor:
        T_pred = root_logits.shape[1]
        T_targ = root_t.shape[1]
        diff = abs(T_pred - T_targ)
        if diff > self.tol:
            raise ValueError(
                f"|T_pred={T_pred} - T_targ={T_targ}| = {diff} > tol {self.tol}"
            )
        T = min(T_pred, T_targ)
        root_logits = root_logits[:, :T]
        bass_logits = bass_logits[:, :T]
        tones_logits = tones_logits[:, :T]
        root_t = root_t[:, :T]
        bass_t = bass_t[:, :T]
        tones_t = tones_t[:, :T]

        # Flatten for CE: (B*T, C) vs (B*T,)
        B = root_logits.shape[0]
        root_loss = self.root_ce(
            root_logits.reshape(B * T, -1), root_t.reshape(B * T)
        )
        bass_loss = self.bass_ce(
            bass_logits.reshape(B * T, -1), bass_t.reshape(B * T)
        )
        tones_loss = self.tones_bce(
            tones_logits.reshape(B * T, -1), tones_t.reshape(B * T, -1)
        )
        return (
            self.root_weight * root_loss
            + self.bass_weight * bass_loss
            + self.tones_weight * tones_loss
        )


# ---------------------------------------------------------------------------
# Probe Module
# ---------------------------------------------------------------------------

class DecomposedChordProbe(pl.LightningModule):
    """Three-head chord probe on pre-extracted frame-level embeddings.

    Heads: root (13-class CE), bass (13-class CE), tones (12-way BCE).
    Test-time evaluation uses mir_eval chord metrics.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        lr: float = 1e-3,
        root_weight: float = 1.0,
        bass_weight: float = 1.0,
        tones_weight: float = 2.0,
        time_dim_mismatch_tol: int = 5,
        clip_seconds: float = 15.0,
        label_freq: int = 25,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.clip_seconds = clip_seconds
        self.label_freq = label_freq
        self.tol = time_dim_mismatch_tol

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Independent heads
        self.root_head = nn.Linear(hidden_dim, NUM_ROOT_CLASSES)
        self.bass_head = nn.Linear(hidden_dim, NUM_BASS_CLASSES)
        self.tones_head = nn.Linear(hidden_dim, NUM_TONE_CLASSES)

        # Loss
        self.loss_fn = DecomposedChordLoss(
            root_weight=root_weight,
            bass_weight=bass_weight,
            tones_weight=tones_weight,
            time_dim_mismatch_tol=time_dim_mismatch_tol,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, L, T, H) — pre-extracted embeddings with layer dim.

        Returns:
            root_logits: (B, T, 13)
            bass_logits: (B, T, 13)
            tones_logits: (B, T, 12)
        """
        if x.dim() == 3:  # (B, T, H)
            x = x.unsqueeze(1)  # → (B, 1, T, H)
        # Pool over layers
        x = reduce(x, "b l t h -> b t h", "mean")
        h = self.shared(x)
        return self.root_head(h), self.bass_head(h), self.tones_head(h)

    # ---- steps ----

    def _shared_step(self, batch, split: str):
        emb, root_t, bass_t, tones_t, _paths = batch
        root_logits, bass_logits, tones_logits = self(emb)
        loss = self.loss_fn(root_logits, bass_logits, tones_logits, root_t, bass_t, tones_t)

        # Root accuracy (fast, no decode)
        T = min(root_logits.shape[1], root_t.shape[1])
        root_pred = root_logits[:, :T].argmax(dim=-1)
        root_acc = (root_pred == root_t[:, :T]).float().mean()

        self.log(f"{split}/loss", loss, prog_bar=True)
        self.log(f"{split}/root_acc", root_acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    # ---- test with mir_eval ----

    def on_test_start(self):
        self._test_outputs: list[dict] = []

    def test_step(self, batch, batch_idx):
        emb, root_t, bass_t, tones_t, _paths = batch
        root_logits, bass_logits, tones_logits = self(emb)
        T = min(root_logits.shape[1], root_t.shape[1])

        self._test_outputs.append({
            "root_pred": root_logits[:, :T].argmax(dim=-1).cpu().numpy(),
            "bass_pred": bass_logits[:, :T].argmax(dim=-1).cpu().numpy(),
            "tones_pred": torch.sigmoid(tones_logits[:, :T]).float().cpu().numpy(),
            "root_t": root_t[:, :T].cpu().numpy(),
            "bass_t": bass_t[:, :T].cpu().numpy(),
            "tones_t": tones_t[:, :T].cpu().numpy(),
        })

    def on_test_epoch_end(self):
        # Concatenate all test outputs
        root_pred = np.concatenate([o["root_pred"] for o in self._test_outputs])
        bass_pred = np.concatenate([o["bass_pred"] for o in self._test_outputs])
        tones_pred = np.concatenate([o["tones_pred"] for o in self._test_outputs])
        root_t = np.concatenate([o["root_t"] for o in self._test_outputs])
        bass_t = np.concatenate([o["bass_t"] for o in self._test_outputs])
        tones_t = np.concatenate([o["tones_t"] for o in self._test_outputs])

        N = root_pred.shape[0]  # number of clips

        # Simple accuracy metrics
        root_acc = (root_pred == root_t).mean()
        bass_acc = (bass_pred == bass_t).mean()
        tones_acc = ((tones_pred > 0.5) == (tones_t > 0.5)).mean()

        self.log("test/root_acc", float(root_acc), sync_dist=True)
        self.log("test/bass_acc", float(bass_acc), sync_dist=True)
        self.log("test/tones_acc", float(tones_acc), sync_dist=True)

        # mir_eval evaluation per clip, then average
        metric_names = ["root", "majmin", "thirds", "triads", "sevenths", "mirex"]
        accum = {k: 0.0 for k in metric_names}
        n_valid = 0
        n_failed = 0

        for i in range(N):
            try:
                est_labels = [
                    decode_chord(
                        int(root_pred[i, t]),
                        int(bass_pred[i, t]),
                        tones_pred[i, t],
                    )
                    for t in range(root_pred.shape[1])
                ]
                ref_labels = [
                    decode_chord(
                        int(root_t[i, t]),
                        int(bass_t[i, t]),
                        tones_t[i, t],
                        threshold=0.5,
                    )
                    for t in range(root_t.shape[1])
                ]
                est_intervals, est_labs = _frames_to_intervals(
                    est_labels, self.clip_seconds
                )
                ref_intervals, ref_labs = _frames_to_intervals(
                    ref_labels, self.clip_seconds
                )

                if len(est_intervals) == 0 or len(ref_intervals) == 0:
                    n_failed += 1
                    continue

                intervals, ref_al, est_al = mir_eval.util.merge_labeled_intervals(
                    ref_intervals, ref_labs, est_intervals, est_labs
                )
                durations = mir_eval.util.intervals_to_durations(intervals)

                for metric_name in metric_names:
                    fn = getattr(mir_eval.chord, metric_name)
                    score = mir_eval.chord.weighted_accuracy(
                        fn(ref_al, est_al), durations
                    )
                    accum[metric_name] += score
                n_valid += 1
            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    logger.warning("mir_eval clip %d failed: %s", i, e)
                continue

        if n_failed > 0:
            logger.warning(
                "mir_eval evaluation: %d/%d clips failed", n_failed, N
            )

        if n_valid > 0:
            for k in metric_names:
                self.log(
                    f"test/{k}",
                    accum[k] / n_valid,
                    prog_bar=True,
                    sync_dist=True,
                )

        self._test_outputs.clear()

    # ---- optimizer ----

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val/loss",
            },
        }

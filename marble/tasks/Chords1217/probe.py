# File: marble/tasks/Chords1217/probe.py

import torch
import torch.nn as nn
import lightning.pytorch as pl
import numpy as np

from torchmetrics import Metric
from torchmetrics import MetricCollection
from einops import rearrange

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config


class ProbeAudioTask(BaseTask):
    """
    Chord Recognition Probe Task (Frame‐level classification).
    """
    def test_step(self, batch, batch_idx: int):
        """
        Default test: returns raw logits and labels for aggregation.
        Override in subclass for custom behavior.
        """
        x, y, paths = batch
        logits = self(x)
        mc: MetricCollection = getattr(self, f"test_metrics", None)
        if mc is not None:
            metrics_out = mc(logits, y)
            self.log_dict(metrics_out, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


class ChordAccuracy(Metric):
    """
    Masked Accuracy for frame‐level chord classification.

    - Inputs:
        logits: Tensor of shape (B, T_logits, C) with raw scores for each frame, 
                where C is the number of classes.
        targets: Tensor of shape (B, T_targets) with integer labels in {0..C-1} 
                 or -1 for ignored frames.

    - Behavior:
        1) Checks that batch sizes match.
        2) If |T_logits − T_targets| > time_dim_mismatch_tol, raises ValueError.
        3) Otherwise, crops both logits and targets to T_min = min(T_logits, T_targets).
        4) Flattens logits and targets, then masks out any positions where target == ignore_index.
        5) Counts correct predictions and total valid frames, and accumulates them.
        6) compute() returns overall accuracy = correct_total / total_valid. 
           If total_valid = 0, returns 0.0.
    """

    def __init__(
        self,
        time_dim_mismatch_tol: int = 0,
        ignore_index: int = -1,
        dist_sync_on_step: bool = False
    ):
        """
        Args:
            time_dim_mismatch_tol (int): 
                Maximum allowed difference between the time dimensions 
                of logits (T_logits) and targets (T_targets). 
                Default = 0 (must match exactly).
            ignore_index (int): 
                Any target equal to this value will be ignored. 
                Default = -1.
            dist_sync_on_step (bool): 
                If True, syncs states across processes at each step in DDP. 
                Default = False.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.time_dim_mismatch_tol = time_dim_mismatch_tol
        self.ignore_index = ignore_index

        # Register states to accumulate correct count and total count across updates
        # Use dist_reduce_fx="sum" so that in distributed training these get summed across all processes
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total",   default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Called on each batch to update the metric state.

        Args:
            logits:  Tensor of shape (B, T_logits, C)
            targets: Tensor of shape (B, T_targets) with values in {0..C-1} or ignore_index.
        """
        # 1) Check batch size consistency
        B, T_logits, C = logits.shape
        B_t, T_targets = targets.shape
        if B_t != B:
            raise ValueError(f"Batch size mismatch: logits batch={B}, targets batch={B_t}")

        # 2) Check time‐dimension difference against tolerance
        diff = abs(T_logits - T_targets)
        if diff > self.time_dim_mismatch_tol:
            raise ValueError(
                f"time dimension mismatch too large: |{T_logits} - {T_targets}| = {diff} > tol ({self.time_dim_mismatch_tol})"
            )

        # 3) Crop both to the same minimum time length
        T_min = min(T_logits, T_targets)
        if T_logits != T_min:
            logits = logits[:, :T_min, :]    # (B, T_min, C)
        if T_targets != T_min:
            targets = targets[:, :T_min]     # (B, T_min)

        # 4) Flatten and mask out ignored positions
        flat_logits  = logits.reshape(-1, C)   # shape = (B * T_min, C)
        flat_targets = targets.reshape(-1)     # shape = (B * T_min,)

        # Create a mask for positions where target != ignore_index
        valid_mask = flat_targets != self.ignore_index
        if valid_mask.sum() == 0:
            # No valid frames in this batch → do not update correct/total
            return

        valid_logits  = flat_logits[valid_mask]   # shape = (N, C)
        valid_targets = flat_targets[valid_mask]  # shape = (N,)

        # 5) Compute number of correct predictions in this batch
        preds = torch.argmax(valid_logits, dim=-1)        # (N,)
        batch_correct = (preds == valid_targets).sum()    # scalar
        batch_total   = valid_targets.numel()             # N

        # 6) Accumulate into the metric state
        self.correct += batch_correct
        self.total   += batch_total

    def compute(self) -> torch.Tensor:
        """
        Returns:
            A scalar tensor representing the overall accuracy = correct / total.
            If total == 0, returns 0.0.
        """
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total.float()


class ChordCrossEntropyLoss(nn.Module):
    """
    Masked CrossEntropyLoss for frame‐level chord classification,
    with an option to tolerate small mismatches in the time dimension.
    If the difference between logits’ T and targets’ T exceeds
    `time_dim_mismatch_tol`, a ValueError is raised. Otherwise,
    both tensors are cropped to the same minimum T before computing loss.
    """

    def __init__(self, time_dim_mismatch_tol: int = 0):
        """
        Args:
            time_dim_mismatch_tol (int): 
                The maximum allowed difference between the time dimensions 
                (T) of logits and targets. Defaults to 0 (must match exactly).
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.time_dim_mismatch_tol = time_dim_mismatch_tol

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (B, T_logits, C) with raw scores, 
                    where C is the number of classes.
            targets: Tensor of shape (B, T_targets) with integer labels 
                     in {0..C-1} or -1 for ignored frames.

        Returns:
            A scalar loss. If |T_logits − T_targets| > time_dim_mismatch_tol,
            raises ValueError. Otherwise, both logits and targets are cropped
            to the smaller time dimension, then flattened and masked before
            applying CrossEntropyLoss.
        """
        B, T_logits, C = logits.shape
        B_t, T_targets = targets.shape
        if B_t != B:
            raise ValueError(
                f"Batch size mismatch: logits batch={B}, targets batch={B_t}"
            )

        # 1) Check the absolute difference in time dimensions
        diff = abs(T_logits - T_targets)
        if diff > self.time_dim_mismatch_tol:
            raise ValueError(
                f"time_dim mismatch too large: |{T_logits} - {T_targets}| = {diff} "
                f"> tol ({self.time_dim_mismatch_tol})"
            )

        # 2) If within tolerance, crop both to the same minimum T
        T_min = min(T_logits, T_targets)
        if T_logits != T_min:
            logits = logits[:, :T_min, :]
        if T_targets != T_min:
            targets = targets[:, :T_min]

        # 3) Proceed with flatten + mask + CrossEntropy
        #    logits is now (B, T_min, C), targets is (B, T_min)
        flat_logits = logits.reshape(-1, C)       # (B * T_min, C)
        flat_targets = targets.reshape(-1)        # (B * T_min,)

        # Only keep positions where target >= 0
        valid_mask = flat_targets >= 0            
        if valid_mask.sum() == 0:
            # If there are no valid frames, return zero loss
            return torch.tensor(0.0, device=logits.device)

        valid_logits = flat_logits[valid_mask]    # (N, C)
        valid_targets = flat_targets[valid_mask]  # (N,)

        return self.ce(valid_logits, valid_targets)


class ProbeOnEmbeddingsTask(ProbeAudioTask):
    """
    Chord recognition probe on pre-extracted embeddings (frame-level).
    """

    def __init__(
        self,
        sample_rate: int,
        use_ema: bool,
        encoder: dict,
        emb_transforms: list[dict],
        decoders: list[dict],
        losses: list[dict],
        metrics: dict[str, dict[str, dict]],
    ):
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]
        metric_maps = {
            split: {
                name: instantiate_from_config(cfg)
                for name, cfg in metrics[split].items()
            }
            for split in ("train", "val", "test")
        }
        super(ProbeAudioTask, self).__init__(
            encoder=enc,
            emb_transforms=tfs,
            decoders=decs,
            losses=loss_fns,
            metrics=metric_maps,
            sample_rate=sample_rate,
            use_ema=use_ema,
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        if h.dim() == 3:  # (B, T, H) -> (B, 1, T, H)
            h = h.unsqueeze(1)
        for t in self.emb_transforms:
            h = t(h)
        outputs = [dec(h) for dec in self.decoders]
        return outputs[0] if len(outputs) == 1 else outputs

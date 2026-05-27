# marble/tasks/HXMSAStructure/metrics.py

import numpy as np
import torch
import torchmetrics
import mir_eval


class SegmentBoundaryFMeasure(torchmetrics.Metric):
    """
    Boundary detection F-measure using mir_eval.segment.detection.

    Expects estimated and reference boundary masks (B, T) where nonzero
    entries mark predicted/reference boundaries.  Boundaries are converted
    to time intervals and evaluated with the given tolerance window.

    Accumulates per-sample F1 from mir_eval and returns the macro average.
    """

    def __init__(
        self,
        label_freq: int,
        window: float = 0.5,
        threshold: float = 0.5,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.label_freq = label_freq
        self.window = window
        self.threshold = threshold

        self.add_state("f1_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def _mask_to_intervals(self, mask_np: np.ndarray) -> np.ndarray:
        """Convert a binary boundary mask to interval array for mir_eval."""
        boundary_frames = np.where(mask_np > 0)[0]
        total_time = len(mask_np) / self.label_freq

        # Always include start (0) and end
        times = [0.0]
        for f in boundary_frames:
            t = f / self.label_freq
            if t > 0 and t < total_time:
                times.append(t)
        times.append(total_time)
        times = sorted(set(times))

        intervals = np.array([[times[i], times[i + 1]] for i in range(len(times) - 1)])
        return intervals

    def update(self, est_mask: torch.Tensor, ref_mask: torch.Tensor):
        est_np = (est_mask.detach().cpu().float().numpy() > self.threshold).astype(np.int32)
        ref_np = (ref_mask.detach().cpu().float().numpy() > self.threshold).astype(np.int32)

        B = est_np.shape[0]
        for b in range(B):
            est_intervals = self._mask_to_intervals(est_np[b])
            ref_intervals = self._mask_to_intervals(ref_np[b])

            if len(est_intervals) == 0 or len(ref_intervals) == 0:
                continue

            precision, recall, f1 = mir_eval.segment.detection(
                ref_intervals, est_intervals, window=self.window
            )
            self.f1_sum += f1
            self.count += 1

    def compute(self):
        if int(self.count) == 0:
            return torch.tensor(0.0, device=self.f1_sum.device)
        return (self.f1_sum / self.count).to(dtype=torch.float32)


class FrameClassAccuracy(torchmetrics.Metric):
    """
    Frame-level macro accuracy for structure classification.

    Computes per-class accuracy then averages (macro), matching omar-rq.

    Expects:
        est_labels: (B, T) integer class predictions
        ref_labels: (B, T) integer class references
    """

    def __init__(self, num_classes: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state(
            "class_correct",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "class_total",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, est_labels: torch.Tensor, ref_labels: torch.Tensor):
        if est_labels.shape[-1] != ref_labels.shape[-1]:
            min_t = min(est_labels.shape[-1], ref_labels.shape[-1])
            est_labels = est_labels[..., :min_t]
            ref_labels = ref_labels[..., :min_t]

        est_flat = est_labels.reshape(-1)
        ref_flat = ref_labels.reshape(-1)
        for c in range(self.num_classes):
            mask = ref_flat == c
            self.class_correct[c] += (est_flat[mask] == c).sum()
            self.class_total[c] += mask.sum()

    def compute(self):
        per_class = []
        for c in range(self.num_classes):
            if self.class_total[c] > 0:
                per_class.append(self.class_correct[c].float() / self.class_total[c])
        if len(per_class) == 0:
            return torch.tensor(0.0, device=self.class_correct.device)
        return torch.stack(per_class).mean().to(dtype=torch.float32)

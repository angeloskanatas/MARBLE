# marble/tasks/HXMSAStructure/probe.py
#
# Frame-level structure segmentation probe with dual heads:
#   - Frame classification head (6 classes: intro/verse/chorus/bridge/outro/inst)
#   - Boundary detection head (binary)
#
# Follows the GTZANBeatTracking probe pattern for MARBLE integration,
# with boundary detection inspired by omar-rq/pedro/structbrrrrr.

import torch
import torch.nn as nn
import torch.nn.functional as F

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config
from marble.tasks.HXMSAStructure.metrics import SegmentBoundaryFMeasure, FrameClassAccuracy


class ProbeAudioTask(BaseTask):
    """
    Frame-level structure segmentation probe.
    Dual-head: frame classification + boundary detection.
    """

    def __init__(
        self,
        sample_rate: int,
        use_ema: bool,
        encoder: dict,
        emb_transforms: list[dict],
        decoders: list[dict],
        losses: list[dict],
        fps: int,
        num_classes: int = 6,
        metrics: dict = None,
        loss_weights: list[float] = [1.0, 1.0],
        smooth_window_sec: float = 2.0,
    ):
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]
        self.loss_weights = loss_weights

        self.sample_rate = sample_rate
        self.use_ema = use_ema
        self.label_freq = fps
        self.num_classes = num_classes
        self.smooth_window_sec = smooth_window_sec
        self._smooth_filter_size = max(1, round(smooth_window_sec * fps))

        # Instantiate metrics
        if metrics is None:
            metrics = {}
        metric_maps = {}
        for split in ("val", "test"):
            metric_maps[split] = {}
            for name, cfg in metrics.get(split, {}).items():
                metric_maps[split][name] = instantiate_from_config(cfg)

        super().__init__(
            encoder=enc,
            emb_transforms=tfs,
            decoders=decs,
            losses=loss_fns,
            metrics={},
            sample_rate=sample_rate,
            use_ema=use_ema,
        )

        # Attach metrics
        self.val_frame_acc = metric_maps.get("val", {}).get(
            "frame_acc", FrameClassAccuracy(num_classes=num_classes)
        )
        self.val_boundary_f1 = metric_maps.get("val", {}).get(
            "boundary_f1", SegmentBoundaryFMeasure(label_freq=fps)
        )
        self.test_frame_acc = metric_maps.get("test", {}).get(
            "frame_acc", FrameClassAccuracy(num_classes=num_classes)
        )
        self.test_boundary_f1 = metric_maps.get("test", {}).get(
            "boundary_f1", SegmentBoundaryFMeasure(label_freq=fps)
        )

    def _apply_moving_average(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Apply temporal moving average smoothing to one-hot labels (omar-rq style).

        Args:
            one_hot: (B, T, C) one-hot encoded labels
        Returns:
            (B, T, C) smoothed label distributions
        """
        if self._smooth_filter_size <= 1:
            return one_hot
        B, T, C = one_hot.shape
        x = one_hot.transpose(1, 2)  # (B, C, T) for grouped conv1d
        kernel = torch.ones(
            C, 1, self._smooth_filter_size, device=x.device, dtype=x.dtype
        ) / self._smooth_filter_size
        x = F.conv1d(x, kernel, padding=self._smooth_filter_size // 2, groups=C)
        if x.shape[2] > T:
            x = x[:, :, :T]
        return x.transpose(1, 2)

    def _compute_losses(self, frame_logits, boundary_logits, frame_labels, boundary_labels):
        """Compute frame (soft CE with label smoothing) + boundary (BCE) losses."""
        if self._smooth_filter_size > 1:
            one_hot = F.one_hot(frame_labels, num_classes=self.num_classes).float()
            smoothed = self._apply_moving_average(one_hot)
            log_probs = F.log_softmax(frame_logits, dim=-1)
            loss_frame = -(smoothed * log_probs).sum(dim=-1).mean()
        else:
            loss_frame = self.loss_fns[0](
                frame_logits.reshape(-1, frame_logits.shape[-1]),
                frame_labels.reshape(-1),
            )
        loss_frame = loss_frame * self.loss_weights[0]
        loss_boundary = self.loss_fns[1](
            boundary_logits, boundary_labels
        ) * self.loss_weights[1]
        return loss_frame, loss_boundary

    def training_step(self, batch, batch_idx):
        x, targets, paths = batch
        outputs = self(x)

        frame_logits = outputs["frame_logits"]    # (B, T, C)
        boundary_logits = outputs["boundaries"]   # (B, T)

        frame_labels = targets["frame_labels"]    # (B, T) int
        boundary_labels = targets["boundaries"]   # (B, T) float

        # Handle time dim mismatch
        T_pred = frame_logits.shape[1]
        T_label = frame_labels.shape[1]
        T = min(T_pred, T_label)
        frame_logits = frame_logits[:, :T, :]
        boundary_logits = boundary_logits[:, :T]
        frame_labels = frame_labels[:, :T]
        boundary_labels = boundary_labels[:, :T]

        loss_frame, loss_boundary = self._compute_losses(
            frame_logits, boundary_logits, frame_labels, boundary_labels
        )
        total_loss = loss_frame + loss_boundary

        self.log("train/loss_frame", loss_frame, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/loss_boundary", loss_boundary, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)

        return total_loss

    def _eval_step(self, batch, split):
        x, targets, paths = batch
        outputs = self(x)

        frame_logits = outputs["frame_logits"]
        boundary_logits = outputs["boundaries"]
        frame_labels = targets["frame_labels"]
        boundary_labels = targets["boundaries"]

        T_pred = frame_logits.shape[1]
        T_label = frame_labels.shape[1]
        T = min(T_pred, T_label)
        frame_logits = frame_logits[:, :T, :]
        boundary_logits = boundary_logits[:, :T]
        frame_labels = frame_labels[:, :T]
        boundary_labels = boundary_labels[:, :T]

        # Losses (smoothed labels, consistent with training)
        loss_frame, loss_boundary = self._compute_losses(
            frame_logits, boundary_logits, frame_labels, boundary_labels
        )
        total_loss = loss_frame + loss_boundary
        self.log(f"{split}/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Frame accuracy
        pred_classes = frame_logits.argmax(dim=-1)  # (B, T)
        frame_acc_metric = getattr(self, f"{split}_frame_acc")
        frame_acc_metric.update(pred_classes, frame_labels)

        # Boundary F1 — pass probabilities to the metric (metric handles thresholding)
        boundary_probs = torch.sigmoid(boundary_logits)  # (B, T)
        boundary_f1_metric = getattr(self, f"{split}_boundary_f1")
        boundary_f1_metric.update(boundary_probs, boundary_labels)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def on_validation_epoch_end(self):
        frame_acc = self.val_frame_acc.compute()
        boundary_f1 = self.val_boundary_f1.compute()
        weighted = 0.5 * frame_acc + 0.5 * boundary_f1

        self.log("val/frame_acc", frame_acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/boundary_f1", boundary_f1, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/weighted_score", weighted, prog_bar=True, on_epoch=True, sync_dist=True)

        self.val_frame_acc.reset()
        self.val_boundary_f1.reset()

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def on_test_epoch_end(self):
        frame_acc = self.test_frame_acc.compute()
        boundary_f1 = self.test_boundary_f1.compute()
        weighted = 0.5 * frame_acc + 0.5 * boundary_f1

        self.log("test/frame_acc", frame_acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test/boundary_f1", boundary_f1, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test/weighted_score", weighted, prog_bar=True, on_epoch=True, sync_dist=True)

        self.test_frame_acc.reset()
        self.test_boundary_f1.reset()


class ProbeOnEmbeddingsTask(ProbeAudioTask):
    """Structure segmentation probe on pre-extracted frame-level embeddings."""

    def __init__(
        self,
        sample_rate: int,
        use_ema: bool,
        encoder: dict,
        emb_transforms: list[dict],
        decoders: list[dict],
        losses: list[dict],
        fps: int,
        num_classes: int = 6,
        metrics: dict = None,
        loss_weights: list[float] = [1.0, 1.0],
        smooth_window_sec: float = 2.0,
    ):
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]

        if metrics is None:
            metrics = {}
        metric_maps = {}
        for split in ("val", "test"):
            metric_maps[split] = {}
            for name, cfg in metrics.get(split, {}).items():
                metric_maps[split][name] = instantiate_from_config(cfg)

        self.loss_weights = loss_weights
        self.sample_rate = sample_rate
        self.use_ema = use_ema
        self.label_freq = fps
        self.num_classes = num_classes
        self.smooth_window_sec = smooth_window_sec
        self._smooth_filter_size = max(1, round(smooth_window_sec * fps))

        # Skip ProbeAudioTask.__init__, go directly to BaseTask
        super(ProbeAudioTask, self).__init__(
            encoder=enc,
            emb_transforms=tfs,
            decoders=decs,
            losses=loss_fns,
            metrics={},
            sample_rate=sample_rate,
            use_ema=use_ema,
        )

        self.val_frame_acc = metric_maps.get("val", {}).get(
            "frame_acc", FrameClassAccuracy(num_classes=num_classes)
        )
        self.val_boundary_f1 = metric_maps.get("val", {}).get(
            "boundary_f1", SegmentBoundaryFMeasure(label_freq=fps)
        )
        self.test_frame_acc = metric_maps.get("test", {}).get(
            "frame_acc", FrameClassAccuracy(num_classes=num_classes)
        )
        self.test_boundary_f1 = metric_maps.get("test", {}).get(
            "boundary_f1", SegmentBoundaryFMeasure(label_freq=fps)
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        if h.dim() == 3:  # (B, T, H) -> (B, 1, T, H)
            h = h.unsqueeze(1)
        for t in self.emb_transforms:
            h = t(h)
        outputs = [dec(h) for dec in self.decoders]
        return outputs[0] if len(outputs) == 1 else outputs


class StructureSegmentationDecoder(nn.Module):
    """
    Dual-head decoder for frame-level structure segmentation.

    Takes 4D input [B, L, T, H], mean-pools over L, then:
      - frame_head: Linear(hidden, num_classes)  — section classification
      - boundary_head: Linear(hidden, 1)          — boundary detection

    Returns dict with "frame_logits" (B, T, C) and "boundaries" (B, T).

    If target_frames is set, embeddings are temporally downsampled via
    AdaptiveAvgPool1d before the MLP (omar-rq style, e.g. 2250→150 for 5Hz).
    """

    def __init__(
        self,
        joint_decoder: dict,
        num_classes: int = 6,
        target_frames: int = None,
    ):
        super().__init__()
        self.joint_decoder = instantiate_from_config(joint_decoder)
        self.num_classes = num_classes
        self.target_frames = target_frames
        if target_frames is not None:
            self.pool = nn.AdaptiveAvgPool1d(target_frames)
        else:
            self.pool = None

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4, f"Expected 4D [B, L, T, H], got {x.dim()}D"
        if self.pool is not None:
            B, L, T, H = x.shape
            # Pool time dim: (B*L, H, T) → (B*L, H, target_frames)
            x = x.view(B * L, T, H).transpose(1, 2)
            x = self.pool(x)
            x = x.transpose(1, 2).view(B, L, self.target_frames, H)
        logits = self.joint_decoder(x)  # (B, T', num_classes + 1)
        frame_logits = logits[:, :, :self.num_classes]     # (B, T', C)
        boundary_logits = logits[:, :, self.num_classes]   # (B, T')
        return {
            "frame_logits": frame_logits,
            "boundaries": boundary_logits,
        }

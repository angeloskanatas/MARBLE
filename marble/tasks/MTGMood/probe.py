# marble/tasks/MTGMood/probe.py
from collections import defaultdict
import json

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import mir_eval
import torchmetrics
from torchmetrics import Metric, MetricCollection
import torch.distributed as dist
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config


class ProbeAudioTask(BaseTask):
    """
    MTGMood probe task. Inherits training/val logic, multi-head,
    losses, metrics and EMA support from BaseTask.
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
        # ... (constructor remains unchanged) ...
        # 1) build all submodules from your YAML configs
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]

        # metrics comes in as nested dict: { split: { name: cfg, … }, … }
        metric_maps = {
            split: {
                name: instantiate_from_config(cfg)
                for name, cfg in metrics[split].items()
            }
            for split in ("train", "val", "test")
        }

        # 2) hand everything off to BaseTask
        super().__init__(
            encoder=enc,
            emb_transforms=tfs,
            decoders=decs,
            losses=loss_fns,
            metrics=metric_maps,
            sample_rate=sample_rate,
            use_ema=use_ema,
        )
        
    def _shared_step(self, batch, batch_idx: int, split: str) -> torch.Tensor:
        """
        Common logic for train:
          - unpack batch, forward, sum all loss_fns, log loss and metrics.
        This now computes slice-level metrics ONLY for the training set.
        """
        x, y, uids_or_paths = batch
        logits = self(x)
        probs = torch.sigmoid(logits)

        # compute and log loss
        losses = [fn(logits, y.to(logits.dtype)) for fn in self.loss_fns]
        loss = sum(losses)
        self.log(f"{split}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # compute and log metrics at slice-level for training
        mc: MetricCollection = getattr(self, f"{split}_metrics", None)
        if mc is not None:
            metrics_out = mc(probs, y)
            self.log_dict(metrics_out, prog_bar=(split == "val"), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    # === Hooks for file-level aggregation ===
    def on_validation_start(self) -> None:
        self._val_file_outputs: list[dict] = []

    def validation_step(self, batch, batch_idx):
        x, y, uids = batch
        logits = self(x)

        # We still need to compute and log the validation loss per epoch
        loss = self.loss_fns[0](logits, y.to(logits.dtype))
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # FIX: Move tensors to CPU before storing to prevent device mismatch errors after gathering.
        for uid, logit, lb in zip(uids, logits, y):
            self._val_file_outputs.append({
                "uid": uid,
                "logit": logit.cpu(),
                "label": lb.cpu()
            })
    
    def on_validation_epoch_end(self) -> None:
        # This hook is now robust for distributed training.
        # It follows the pattern: GATHER -> COMPUTE ON RANK 0 -> BROADCAST -> LOG ON ALL RANKS
        if not dist.is_available() or not dist.is_initialized():
            if hasattr(self, '_val_file_outputs') and self._val_file_outputs:
                metrics = self._aggregate_and_compute_metrics(self._val_file_outputs, "val")
                if metrics:
                    self.log_dict(metrics, on_step=False, on_epoch=True)
            if hasattr(self, '_val_file_outputs'):
                self._val_file_outputs.clear()
            return
            
        # Step 1: All ranks gather outputs.
        gathered_outputs: list = [None] * self.trainer.world_size
        dist.all_gather_object(gathered_outputs, self._val_file_outputs)
        
        metrics = None
        
        # Step 2: Only rank 0 computes metrics.
        if self.trainer.is_global_zero:
            all_outputs = [item for sublist in gathered_outputs for item in (sublist or [])]
            metrics = self._aggregate_and_compute_metrics(all_outputs, "val")

        # Step 3: Broadcast the computed metrics from rank 0 to all other ranks.
        metrics_list = [metrics]
        dist.broadcast_object_list(metrics_list, src=0)

        # Step 4: All ranks log the synchronized metrics.
        if metrics_list[0] is not None:
            self.log_dict(metrics_list[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)

        # Step 5: All ranks clear their local output list.
        if hasattr(self, '_val_file_outputs'):
            self._val_file_outputs.clear()

    def on_test_start(self) -> None:
        self._test_file_outputs: list[dict] = []
    
    def test_step(self, batch, batch_idx):
        x, labels, ori_uids = batch
        logits = self(x)

        # FIX: Move tensors to CPU before storing.
        for uid, logit, lb in zip(ori_uids, logits, labels):
            self._test_file_outputs.append({
                "uid": uid,
                "logit": logit.cpu(),
                "label": lb.cpu(),
            })

    def on_test_epoch_end(self) -> None:
        # Applying the same robust distributed pattern as for validation.
        if not dist.is_available() or not dist.is_initialized():
            if hasattr(self, '_test_file_outputs') and self._test_file_outputs:
                metrics = self._aggregate_and_compute_metrics(self._test_file_outputs, "test")
                if metrics:
                    self.log_dict(metrics, on_step=False, on_epoch=True)
            if hasattr(self, '_test_file_outputs'):
                self._test_file_outputs.clear()
            return

        gathered_outputs: list = [None] * self.trainer.world_size
        dist.all_gather_object(gathered_outputs, self._test_file_outputs)
        
        metrics = None
        if self.trainer.is_global_zero:
            all_outputs = [item for sublist in gathered_outputs for item in (sublist or [])]
            metrics = self._aggregate_and_compute_metrics(all_outputs, "test")
            
        metrics_list = [metrics]
        dist.broadcast_object_list(metrics_list, src=0)
        
        if metrics_list[0] is not None:
            self.log_dict(metrics_list[0], on_step=False, on_epoch=True, sync_dist=False)
            
        if hasattr(self, '_test_file_outputs'):
            self._test_file_outputs.clear()

    # === Private helper for aggregation ===
    def _aggregate_and_compute_metrics(self, outputs: list[dict], split: str) -> dict[str, torch.Tensor] | None:
        """Aggregates slice-level logits, computes metrics, and returns a dict."""
        file_dict: dict[str, dict] = {}
        for entry in outputs:
            uid = entry["uid"]
            info = file_dict.setdefault(uid, {"logits": [], "label": entry["label"]})
            info["logits"].append(entry["logit"])

        if self.trainer.is_global_zero:
            print(f"Aggregating {len(file_dict)} files for '{split}' split...")

        batched_logits = []
        batched_labels = []
        for uid, info in file_dict.items():
            # Move tensors to the correct device for computation
            tensors_on_device = [t.to(self.device) for t in info["logits"]]
            arr = torch.stack(tensors_on_device)
            mean_logit = arr.mean(dim=0)
            batched_logits.append(mean_logit)
            batched_labels.append(info["label"].to(self.device))

        if not batched_logits:
            if self.trainer.is_global_zero:
                print(f"Warning: No outputs to aggregate for '{split}' split.")
            return None

        batched_logits = torch.stack(batched_logits)
        batched_labels = torch.stack(batched_labels)

        batched_probs = torch.sigmoid(batched_logits)

        mc: MetricCollection = getattr(self, f"{split}_metrics", None)
        if mc is not None:
            metrics_out = mc(batched_probs, batched_labels)
            return metrics_out # Return the dictionary

        return None


class ProbeOnEmbeddingsTask(ProbeAudioTask):
    """
    MTGMood probe on pre-extracted embeddings.
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
        if h.dim() == 2:
            h = h.unsqueeze(1).unsqueeze(2)
        for t in self.emb_transforms:
            h = t(h)
        outputs = [dec(h) for dec in self.decoders]
        return outputs[0] if len(outputs) == 1 else outputs
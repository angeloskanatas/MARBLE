# marble/tasks/HookTheoryStructure/probe.py
from collections import defaultdict
import json

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import mir_eval
import torchmetrics
from torchmetrics import Metric, MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config
from marble.tasks.HookTheoryStructure.datamodule import _HookTheoryStructureAudioBase


class ProbeAudioTask(BaseTask):
    """
    HookTheoryStructure probe task. Inherits training/val logic, multi-head,
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
        
    def on_test_start(self) -> None:
        # Initialize storage for per-slice test outputs
        self._test_file_outputs: list[dict] = []
    
    def test_step(self, batch, batch_idx):
        x, labels, ori_uids = batch
        logits = self(x)

        for uid, logit, lb in zip(ori_uids, logits, labels):
            self._test_file_outputs.append({
                "uid": uid,
                "logit": logit,
                "label": lb,
            })

    def on_test_epoch_end(self) -> None:
        # Aggregate per-file predictions
        file_dict: dict[str, dict] = {}
        for entry in self._test_file_outputs:
            uid = entry["uid"]
            info = file_dict.setdefault(uid, {"logits": [], "label": entry["label"]})
            info["logits"].append(entry["logit"])

        # aggregate logits and compute file-level metrics
        print(f"Aggregating {len(file_dict)} files with per-slice outputs")
        batched_logits = []
        batched_labels = []
        for uid, info in file_dict.items():
            arr = torch.stack(info["logits"])      # (n_slices, C)
            mean_logit = arr.mean(dim=0)             # (C,)
            batched_logits.append(mean_logit)
            batched_labels.append(info["label"])
        batched_logits = torch.stack(batched_logits)
        batched_labels = torch.stack(batched_labels)
        # compute metrics
        mc: MetricCollection = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc(batched_logits, batched_labels)
            self.log_dict(metrics_out, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


class ProbeOnEmbeddingsTask(ProbeAudioTask):
    """
    HookTheoryStructure probe on pre-extracted embeddings.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        if h.dim() == 2:
            h = h.unsqueeze(1).unsqueeze(2)
        elif h.dim() == 3:  # (B, L, H) -> (B, L, 1, H)
            h = h.unsqueeze(2)
        for t in self.emb_transforms:
            h = t(h)
        outputs = [dec(h) for dec in self.decoders]
        return outputs[0] if len(outputs) == 1 else outputs


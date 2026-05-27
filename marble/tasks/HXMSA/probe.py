# marble/tasks/HXMSA/probe.py

import torch
from torchmetrics import MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config


class ProbeAudioTask(BaseTask):
    """
    HXMSA (Harmonix Music Structure Analysis) probe task.

    Clip-level multi-class classification of structural labels.
    Test-time aggregation: averages logits across all clips from the
    same track before computing metrics.
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
        self._test_file_outputs: list[dict] = []

    def test_step(self, batch, batch_idx):
        x, labels, segment_keys = batch
        logits = self(x)

        for key, logit, lb in zip(segment_keys, logits, labels):
            # Extract track_id from "track_id||segment_start"
            track_id = key.split("||")[0]
            self._test_file_outputs.append({
                "track_id": track_id,
                "logit": logit,
                "label": lb,
            })

    def on_test_epoch_end(self) -> None:
        # Compute clip-level metrics across all test clips.
        # Unlike HookTheoryStructure (where all clips from one file share a label),
        # HXMSA tracks contain multiple segments with different labels, so
        # per-track averaging would be incorrect. We evaluate per-clip directly.
        batched_logits = torch.stack([e["logit"] for e in self._test_file_outputs])
        batched_labels = torch.stack([e["label"] for e in self._test_file_outputs])

        mc: MetricCollection = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc(batched_logits, batched_labels)
            self.log_dict(
                metrics_out,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )


class ProbeOnEmbeddingsTask(ProbeAudioTask):
    """
    HXMSA probe on pre-extracted sequence-level embeddings.
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
        if h.dim() == 2:  # (B, H) → (B, 1, 1, H) for sequence-level
            h = h.unsqueeze(1).unsqueeze(2)
        elif h.dim() == 3:  # (B, L, H) -> (B, L, 1, H)
            h = h.unsqueeze(2)
        for t in self.emb_transforms:
            h = t(h)
        outputs = [dec(h) for dec in self.decoders]
        return outputs[0] if len(outputs) == 1 else outputs

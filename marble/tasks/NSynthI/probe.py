# marble/tasks/NSynthI/probe.py

import torch

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config


class ProbeAudioTask(BaseTask):
    """NSynth instrument family probe task (11-class classification)."""

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
        x, labels, file_paths = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1).cpu()

        for fp, prob, lb in zip(file_paths, probs, labels.cpu()):
            self._test_file_outputs.append({
                "file_path": fp,
                "prob": prob.numpy(),
                "label": int(lb),
            })

    def on_test_epoch_end(self) -> None:
        file_dict: dict[str, dict] = {}
        for entry in self._test_file_outputs:
            fp = entry["file_path"]
            info = file_dict.setdefault(fp, {"probs": [], "label": entry["label"]})
            info["probs"].append(entry["prob"])

        total, correct = 0, 0
        for fp, info in file_dict.items():
            arr = torch.tensor(info["probs"])
            mean_prob = arr.mean(dim=0)
            pred = int(mean_prob.argmax().item())
            total += 1
            correct += int(pred == info["label"])

        file_acc = correct / total if total > 0 else 0.0
        self.log("test/file_acc", file_acc, prog_bar=True, on_epoch=True, sync_dist=True)


class ProbeOnEmbeddingsTask(ProbeAudioTask):
    """NSynth instrument family probe on pre-extracted embeddings."""

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

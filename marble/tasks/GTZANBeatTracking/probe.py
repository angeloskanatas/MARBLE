# marble/tasks/GTZANBeatTracking/probe.py

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl

from einops import reduce
from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config

from marble.tasks.GTZANBeatTracking.madmom.beats import DBNBeatTrackingProcessor
from marble.tasks.GTZANBeatTracking.metrics import TimeEventFMeasure, TempoMAE, TempoAccuracy
from marble.utils.utils import times_to_mask, mask_to_times


class ProbeAudioTask(BaseTask):
    """
    GTZAN Beat/Downbeat/Tempo Probe Task with DBN‐based decoding.

    - Beat tracking: frame‐wise logits → sigmoid → DBN → event times → mask → TimeEventFMeasure
    - Downbeat tracking: same process via DBN on downbeat head
    - Tempo estimation: scalar regression (BPM) → TempoMAE & TempoAccuracy
    """

    def __init__(
        self,
        sample_rate: int,
        use_ema: bool,
        encoder: dict,
        emb_transforms: list[dict],
        decoders: list[dict],
        losses: list[dict],
        fps: int,            # frames per second used by beat/downbeat
        metrics: dict,       # nested dict: { "val": { ... }, "test": { ... } }
        loss_weights: list[float] = [1.0, 1.0, 1.0],
    ):
        # 1) Instantiate encoder, embedding transforms, decoders, and loss functions
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]
        self.loss_weights = loss_weights

        # 2) Save decoding parameters
        self.sample_rate = sample_rate
        self.use_ema = use_ema
        self.label_freq = fps           # e.g. 75 frames/sec

        # 3) Instantiate metrics for 'val' and 'test' splits from the provided config
        metric_maps = {}
        for split in ("val", "test"):
            metric_maps[split] = {}
            for name, cfg in metrics.get(split, {}).items():
                metric_maps[split][name] = instantiate_from_config(cfg)

        # 4) Prepare DBN processors (can be reused across batches/epochs)
        self.beat_dbn = DBNBeatTrackingProcessor(fps=self.label_freq)
        self.dbn_dbn = DBNBeatTrackingProcessor(fps=self.label_freq)

        # 5) Call BaseTask.__init__ once, passing in encoder/transforms/decoders/losses.
        #    We pass metrics={} here because we will manage all updates manually.
        super().__init__(
            encoder=enc,
            emb_transforms=tfs,
            decoders=decs,
            losses=loss_fns,
            metrics={},
            sample_rate=sample_rate,
            use_ema=use_ema,
        )

        # 6) Attach instantiated 'val' metrics as attributes for manual update
        #    (keys must match those in YAML: "beat_f1", "downbeat_f1", "tempo_mae", "tempo_acc")
        self.val_beat_f1   = metric_maps["val"]["beat_f1"]
        self.val_db_f1     = metric_maps["val"]["downbeat_f1"]
        self.val_tempo_mae = metric_maps["val"]["tempo_mae"]
        self.val_tempo_acc = metric_maps["val"]["tempo_acc"]

        # 7) Attach instantiated 'test' metrics similarly
        self.test_beat_f1   = metric_maps["test"]["beat_f1"]
        self.test_db_f1     = metric_maps["test"]["downbeat_f1"]
        self.test_tempo_mae = metric_maps["test"]["tempo_mae"]
        self.test_tempo_acc = metric_maps["test"]["tempo_acc"]


    def training_step(self, batch, batch_idx):
        """
        Standard training: compute three losses (beat, downbeat, tempo) and sum them.
        """
        x, targets, paths = batch
        outputs = self(x)

        # Unpack outputs
        beat_logits = outputs["beat"]            # shape: (B, T)
        db_logits   = outputs["downbeat"]        # shape: (B, T)
        tempo_pred  = outputs["tempo"].squeeze()  # shape: (B,)

        # Unpack targets
        beat_target  = targets["beat"].float()           # shape: (B, T)
        db_target    = targets["downbeat"].float()       # shape: (B, T)
        tempo_target = targets["tempo"].float().squeeze() # shape: (B,)

        # Compute individual losses
        loss_beat  = self.loss_fns[0](beat_logits, beat_target) * self.loss_weights[0]
        loss_db    = self.loss_fns[1](db_logits, db_target) * self.loss_weights[1]
        loss_tempo = self.loss_fns[2](tempo_pred, tempo_target) * self.loss_weights[2]
        total_loss = loss_beat + loss_db + loss_tempo

        # Log training losses
        self.log("train/loss_beat",  loss_beat,  on_step=True,  on_epoch=False, prog_bar=False)
        self.log("train/loss_db",    loss_db,    on_step=True,  on_epoch=False, prog_bar=False)
        self.log("train/loss_tempo", loss_tempo, on_step=True,  on_epoch=False, prog_bar=False)
        self.log("train/total_loss", total_loss, on_step=True,  on_epoch=False, prog_bar=True)

        return total_loss


    def validation_step(self, batch, batch_idx):
        """
        Validation step:
        1) Forward pass → get logits for beat/downbeat/tempo.
        2) Compute and log frame‐wise losses.
        3) For each sample:
           - Convert beat_logits/downbeat_logits to probabilities (sigmoid).
           - Use DBN to decode probabilities into event times (in seconds).
           - Convert event times into binary mask (times_to_mask).
           - Update TimeEventFMeasure for beat and downbeat.
        4) Update TempoMAE and TempoAccuracy with predicted vs. ground‐truth BPM.
        """
        x, targets, paths = batch
        outputs = self(x)

        # Unpack outputs
        beat_logits = outputs["beat"]            # (B, T)
        db_logits   = outputs["downbeat"]        # (B, T)
        tempo_pred  = outputs["tempo"].squeeze()  # (B,)

        # Unpack targets
        beat_target  = targets["beat"].float()           # (B, T)
        db_target    = targets["downbeat"].float()       # (B, T)
        tempo_target = targets["tempo"].float().squeeze() # (B,)

        # Compute frame‐wise losses for monitoring
        loss_beat  = self.loss_fns[0](beat_logits, beat_target) * self.loss_weights[0]
        loss_db    = self.loss_fns[1](db_logits, db_target) * self.loss_weights[1]
        loss_tempo = self.loss_fns[2](tempo_pred, tempo_target) * self.loss_weights[2]
        total_loss = loss_beat + loss_db + loss_tempo

        self.log("val/loss_beat",  loss_beat,  on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss_db",    loss_db,    on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss_tempo", loss_tempo, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Convert beat/downbeat logits to probabilities
        beat_probs = torch.sigmoid(beat_logits)  # (B, T)
        db_probs   = torch.sigmoid(db_logits)    # (B, T)
        B, T = beat_probs.shape

        # For each sample in the batch, decode events and update f1 metrics
        for b in range(B):
            # --- Beat decoding ---
            beat_act = beat_probs[b].detach().cpu().to(torch.float32).numpy()      # (T,)
            est_beat_times = self.beat_dbn(beat_act)             # e.g. [0.5, 1.333, ...] in seconds
            est_beat_mask_np = times_to_mask(est_beat_times, T, self.label_freq)
            est_beat_tensor = torch.from_numpy(est_beat_mask_np).unsqueeze(0)  # (1, T)
            ref_beat_tensor = beat_target[b].unsqueeze(0)                     # (1, T)
            self.val_beat_f1.update(est_beat_tensor, ref_beat_tensor)

            # --- Downbeat decoding ---
            db_act = db_probs[b].detach().cpu().to(torch.float32).numpy()
            est_db_times = self.dbn_dbn(db_act)
            est_db_mask_np = times_to_mask(est_db_times, T, self.label_freq)
            est_db_tensor = torch.from_numpy(est_db_mask_np).unsqueeze(0)     # (1, T)
            ref_db_tensor = db_target[b].unsqueeze(0)                          # (1, T)
            self.val_db_f1.update(est_db_tensor, ref_db_tensor)

        # --- Tempo metrics update (B,)
        self.val_tempo_mae.update(tempo_pred, tempo_target)
        self.val_tempo_acc.update(tempo_pred, tempo_target)


    def on_validation_epoch_end(self,):
        """
        At the end of validation epoch:
        - Compute and log beat_f1, downbeat_f1, tempo_mae, tempo_acc1, tempo_acc2.
        - Reset all metrics for the next epoch.
        """
        beat_f1     = self.val_beat_f1.compute()   # Tensor
        downbeat_f1 = self.val_db_f1.compute()     # Tensor
        tempo_mae   = self.val_tempo_mae.compute() # Tensor
        tempo_acc   = self.val_tempo_acc.compute() # dict {"acc1": Tensor, "acc2": Tensor}

        # Log F1 scores under consistent "_f1" naming
        self.log("val/beat_f1",       beat_f1,     prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/downbeat_f1",   downbeat_f1, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/tempo_mae",     tempo_mae,   prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/tempo_acc1",    tempo_acc["acc1"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/tempo_acc2",    tempo_acc["acc2"], prog_bar=True, on_epoch=True, sync_dist=True)

        # Reset metrics
        self.val_beat_f1.reset()
        self.val_db_f1.reset()
        self.val_tempo_mae.reset()
        self.val_tempo_acc.reset()


    def test_step(self, batch, batch_idx):
        """
        Test step:
        - Similar to validation_step for metrics update, but without logging losses.
        """
        x, targets, paths = batch
        outputs = self(x)

        beat_logits = outputs["beat"]             # (B, T)
        db_logits   = outputs["downbeat"]         # (B, T)
        tempo_pred  = outputs["tempo"].squeeze()   # (B,)

        beat_target  = targets["beat"].float()       # (B, T)
        db_target    = targets["downbeat"].float()   # (B, T)
        tempo_target = targets["tempo"].float().squeeze()  # (B,)

        beat_probs = torch.sigmoid(beat_logits)     # (B, T)
        db_probs   = torch.sigmoid(db_logits)       # (B, T)
        B, T = beat_probs.shape

        for b in range(B):
            # --- Beat decoding ---
            beat_act = beat_probs[b].detach().cpu().to(torch.float32).numpy()
            est_beat_times = self.beat_dbn(beat_act)
            est_beat_mask_np = times_to_mask(est_beat_times, T, self.label_freq)
            est_beat_tensor = torch.from_numpy(est_beat_mask_np).unsqueeze(0)
            ref_beat_tensor = beat_target[b].unsqueeze(0)
            self.test_beat_f1.update(est_beat_tensor, ref_beat_tensor)

            # --- Downbeat decoding ---
            db_act = db_probs[b].detach().cpu().to(torch.float32).numpy()
            est_db_times = self.dbn_dbn(db_act)
            est_db_mask_np = times_to_mask(est_db_times, T, self.label_freq)
            est_db_tensor = torch.from_numpy(est_db_mask_np).unsqueeze(0)
            ref_db_tensor = db_target[b].unsqueeze(0)
            self.test_db_f1.update(est_db_tensor, ref_db_tensor)

        # --- Tempo metrics update (B,)
        self.test_tempo_mae.update(tempo_pred, tempo_target)
        self.test_tempo_acc.update(tempo_pred, tempo_target)


    def on_test_epoch_end(self):
        """
        At the end of test epoch:
        - Compute and log beat_f1, downbeat_f1, tempo_mae, tempo_acc1, tempo_acc2.
        - Reset all test metrics.
        """
        beat_f1     = self.test_beat_f1.compute()
        downbeat_f1 = self.test_db_f1.compute()
        tempo_mae   = self.test_tempo_mae.compute()
        tempo_acc   = self.test_tempo_acc.compute()

        self.log("test/beat_f1",       beat_f1,     prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test/downbeat_f1",   downbeat_f1, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test/tempo_mae",     tempo_mae,   prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test/tempo_acc1",    tempo_acc["acc1"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test/tempo_acc2",    tempo_acc["acc2"], prog_bar=True, on_epoch=True, sync_dist=True)

        self.test_beat_f1.reset()
        self.test_db_f1.reset()
        self.test_tempo_mae.reset()
        self.test_tempo_acc.reset()


class ProbeOnEmbeddingsTask(ProbeAudioTask):
    """
    GTZAN beat-tracking probe on pre-extracted embeddings.
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
        metrics: dict,
        loss_weights: list[float] = [1.0, 1.0, 1.0],
    ):
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]

        metric_maps = {}
        for split in ("val", "test"):
            metric_maps[split] = {}
            for name, cfg in metrics.get(split, {}).items():
                metric_maps[split][name] = instantiate_from_config(cfg)

        self.loss_weights = loss_weights
        self.sample_rate = sample_rate
        self.use_ema = use_ema
        self.label_freq = fps
        self.beat_dbn = DBNBeatTrackingProcessor(fps=self.label_freq)
        self.dbn_dbn = DBNBeatTrackingProcessor(fps=self.label_freq)

        super(ProbeAudioTask, self).__init__(
            encoder=enc,
            emb_transforms=tfs,
            decoders=decs,
            losses=loss_fns,
            metrics={},
            sample_rate=sample_rate,
            use_ema=use_ema,
        )

        self.val_beat_f1 = metric_maps["val"]["beat_f1"]
        self.val_db_f1 = metric_maps["val"]["downbeat_f1"]
        self.val_tempo_mae = metric_maps["val"]["tempo_mae"]
        self.val_tempo_acc = metric_maps["val"]["tempo_acc"]

        self.test_beat_f1 = metric_maps["test"]["beat_f1"]
        self.test_db_f1 = metric_maps["test"]["downbeat_f1"]
        self.test_tempo_mae = metric_maps["test"]["tempo_mae"]
        self.test_tempo_acc = metric_maps["test"]["tempo_acc"]

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        if h.dim() == 3:  # (B, T, H) -> (B, 1, T, H)
            h = h.unsqueeze(1)
        for t in self.emb_transforms:
            h = t(h)
        outputs = [dec(h) for dec in self.decoders]
        return outputs[0] if len(outputs) == 1 else outputs


class BeatDownbeatTempoMultitaskDecoder(nn.Module):
    """
    Multi‐task decoder for GTZAN Beat/Downbeat/Tempo Probe Task.

    Outputs:
      - beat_logits: (B, T)
      - db_logits:   (B, T)
      - tempo_pred:  (B,)
    """
    def __init__(self,
                 joint_decoder: dict,
                 tempo_decoder: dict,
                 fps: int,
                 use_ssl_for_tempo: bool = False
                 ):
        super().__init__()
        self.fps = fps
        # Instantiate the shared MLP decoder for beat & downbeat (output dim=2)
        self.joint_decoder = instantiate_from_config(joint_decoder)
        # Instantiate vector‐to‐BPM decoder
        self.tempo_decoder = instantiate_from_config(tempo_decoder)
        self.use_ssl_for_tempo = use_ssl_for_tempo

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the decoder.
        x: (B, L, T, H) where H is the feature dimension
        Returns dict with keys "beat", "downbeat", "tempo".
        """
        assert x.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {x.dim()}D tensor"
        B, L, T, H = x.shape

        # 1) Decode beat and downbeat logits
        logits = self.joint_decoder(x)
        # Expect shape [B, T, 3]
        assert logits.dim() == 3 and logits.shape[2] == 3, \
            f"Expected [B, T, 2], got {logits.shape}"
        beat_logits = logits[:, :, 0]   # shape: (B, T)
        db_logits   = logits[:, :, 1]   # shape: (B, T)
        tempo_logits = logits[:, :, 2]  # shape: (B, T)

        # 2) Decode tempo prediction
        if self.use_ssl_for_tempo:
            # If using SSL embedding for tempo, pool over (L, T) → (B, H)
            ssl_emb = reduce(x, 'b l t h -> b h', 'mean')
            assert ssl_emb.dim() == 2, f"Expected 2D SSL embedding [B, H], got {ssl_emb.shape}"
            tempo_pred = self.tempo_decoder(tempo_logits, ssl_emb)
        else:
            # Otherwise, only use tempo_logits (flattened or pooled inside tempo decoder)
            tempo_pred = self.tempo_decoder(tempo_logits)

        assert tempo_pred.dim() == 1, f"Expected 1D tensor [B], got {tempo_pred.dim()}D tensor"

        return {
            "beat":     beat_logits,
            "downbeat": db_logits,
            "tempo":    tempo_pred,
        }

class CustomBCEWithLogitsLoss(nn.Module):
    """
    Custom BCEWithLogitsLoss that ignores padding frames.
    """
    def __init__(self, pos_weight: float = 1.0, time_dim_mismatch_tol: int = 5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight) if pos_weight else None)
        self.time_dim_mismatch_tol = time_dim_mismatch_tol
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: (B, T) tensor of logits
        targets: (B, T) tensor of binary targets
        """
        # handle shape mismatch, if abs difference is larger than time_dim_mismatch_tol
        if logits.shape != targets.shape:
            if abs(logits.shape[1] - targets.shape[1]) > self.time_dim_mismatch_tol:
                raise ValueError(f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}")
            # If the difference is within the tolerance, we can pad or trim
            min_length = min(logits.shape[1], targets.shape[1])
            logits = logits[:, :min_length]
            targets = targets[:, :min_length]        
        
        # Compute the loss only on non-padding frames
        loss = self.bce_loss(logits, targets)
        return loss

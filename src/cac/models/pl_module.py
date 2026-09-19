"""CountingLit — the Lightning training module for FSC147 counting.

MSE(density) + w_cnt·L1(sum, count) loss, AdamW, CosineAnnealingLR, per-epoch
val MAE/RMSE. The wrapped model's frozen forward contract is preserved: only
`out["density"]` feeds the loss/gradient in the default regression path.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics import MeanMetric

from cac.hub import hf_token  # noqa: F401


def counting_loss(dens: torch.Tensor, gt_d: torch.Tensor, gt_c: torch.Tensor,
                  w_cnt: float, loss_fn: str = "mse", huber_delta: float = 5.0) -> torch.Tensor:
    """Density regression loss: density term + w_cnt * L1(sum(density), count)."""
    if dens.shape[-2:] != gt_d.shape[-2:]:
        oh, ow = int(dens.shape[-2]), int(dens.shape[-1])
        dens = F.interpolate(dens.float(), size=gt_d.shape[-2:], mode="bilinear", align_corners=False)
        dens = dens * (oh * ow) / float(dens.shape[-2] * dens.shape[-1])  # sum-conserving
    if loss_fn == "huber":
        dens_loss = F.huber_loss(dens.float(), gt_d, delta=huber_delta, reduction="mean")
    else:
        dens_loss = F.mse_loss(dens.float(), gt_d)
    count_loss = F.l1_loss(dens.float().flatten(1).sum(1), gt_c)
    return dens_loss + w_cnt * count_loss


class CountingLit(LightningModule):
    def __init__(self, cfg: dict[str, Any], model: nn.Module | None = None,
                 build: Any = None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=("model", "build"))
        self.model = model if model is not None else build(cfg)
        self.w_cnt = float(cfg.get("loss_count_weight", 0.4))
        self.loss_fn = cfg.get("loss_function", "mse")
        self.huber_delta = float(cfg.get("huber_delta", 5.0))
        self._train_mae = MeanMetric()
        self._val_mae = MeanMetric()
        self._val_rmse = MeanMetric()

    def configure_optimizers(self):
        params = (self.model.parameters
                  if not hasattr(self.model, "param_groups") else self.model.param_groups(
                      float(self.cfg.get("lr", 1e-3)), float(self.cfg.get("weight_decay", 0.08))))
        if hasattr(self.model, "param_groups"):
            optim = torch.optim.AdamW(params, betas=tuple(self.cfg.get("betas", (0.9, 0.999))))
        else:
            optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                      lr=float(self.cfg.get("lr", 1e-3)),
                                      weight_decay=float(self.cfg.get("weight_decay", 0.08)),
                                      betas=tuple(self.cfg.get("betas", (0.9, 0.999))))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.trainer.max_epochs, eta_min=float(self.cfg.get("eta_min", 1e-6)))
        return [optim], [{"scheduler": sched, "interval": "epoch", "frequency": 1}]

    def _call(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        imgs, bbox = batch["imgs"], batch["bboxes"]
        b3 = batch.get("bboxes3")
        if b3 is not None:
            try:
                return self.model(imgs, bbox, b3)
            except TypeError:
                return self.model(imgs, bbox)
        return self.model(imgs, bbox)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        gt_d, gt_c = batch["density"], batch["counts"]
        out = self._call(batch)
        dens = out["density"] if isinstance(out, dict) else out
        loss = counting_loss(dens, gt_d, gt_c, self.w_cnt, self.loss_fn, self.huber_delta)
        pred = dens.flatten(1).sum(1)
        self._train_mae((pred - gt_c).abs())
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/loss_epoch", loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log("train/mae", self._train_mae, on_epoch=True, on_step=False, prog_bar=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        gt_c = batch["counts"]
        out = self._call(batch)
        dens = out["density"] if isinstance(out, dict) else out
        pred = dens.flatten(1).sum(1)
        self._val_mae((pred - gt_c).abs())
        self._val_rmse((pred - gt_c) ** 2)

    def on_validation_epoch_end(self):
        mae = self._val_mae.compute()
        rmse = self._val_rmse.compute().sqrt()
        self.log("val/mae", mae, on_epoch=True, prog_bar=True)
        self.log("val/rmse", rmse, on_epoch=True, prog_bar=False)
        self._val_mae.reset()
        self._val_rmse.reset()
        self._train_mae.reset()
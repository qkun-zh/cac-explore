"""EMA callback — exponential moving average of trainable weights (optional).

Maintains a shadow copy of every trainable parameter, updated each training
step. During validation the shadows are swapped into the model so
BestCheckpoint records EMA-inference MAE, then restored immediately after.
"""
from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class EMACallback(Callback):
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self._initialized = False

    def _init(self, pl_module: LightningModule) -> None:
        for name, p in pl_module.model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name] = p.detach().clone().to(p.device)
        self._initialized = True

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule,
                             batch: Any, batch_idx: int) -> None:
        if not self._initialized:
            self._init(pl_module)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        if not self._initialized:
            return
        with torch.no_grad():
            d = self.decay
            for name, p in pl_module.model.named_parameters():
                if not p.requires_grad:
                    continue
                s = self.shadow[name]
                s.mul_(d).add_(p.detach(), alpha=1 - d)

    def _swap(self, pl_module: LightningModule, to_shadow: bool) -> None:
        with torch.no_grad():
            for name, p in pl_module.model.named_parameters():
                if not p.requires_grad:
                    continue
                s = self.shadow[name]
                if to_shadow:
                    self.backup[name] = p.detach().clone()
                    p.copy_(s)
                else:
                    p.copy_(self.backup[name])

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._initialized and not trainer.sanity_checking:
            self._swap(pl_module, to_shadow=True)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._initialized and not trainer.sanity_checking:
            self._swap(pl_module, to_shadow=False)
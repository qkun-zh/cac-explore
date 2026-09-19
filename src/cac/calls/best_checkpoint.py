"""Best checkpoint callback — writes best.pth on val/mae improvement."""
from __future__ import annotations

import os

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class BestCheckpoint(Callback):
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.best = float("inf")
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.sanity_checking:
            mae = trainer.callback_metrics.get("val/mae")
            if mae is not None:
                mae = float(mae)
                if mae < self.best:
                    self.best = mae
                    self.best_epoch = trainer.current_epoch + 1
                    path = os.path.join(self.run_dir, "best.pth")
                    torch.save({"epoch": self.best_epoch,
                                "model": pl_module.model.state_dict(),
                                "best_mae": mae}, path)
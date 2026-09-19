"""Best checkpoint sink — writes best.pth on val/mae improvement.

Not a Lightning Callback: CountingLit calls it from its own
on_validation_epoch_end with the freshly computed val MAE, so there is no
hook-ordering ambiguity (callback hooks run before the module hook and read
stale `callback_metrics`).
"""
from __future__ import annotations

import os

import torch


class BestCheckpoint:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.best = float("inf")
        self.best_epoch = 0

    def __call__(self, pl_module, mae: float) -> None:
        mae = float(mae)
        if mae < self.best:
            self.best = mae
            self.best_epoch = pl_module.current_epoch + 1
            path = os.path.join(self.run_dir, "best.pth")
            torch.save({"epoch": self.best_epoch,
                        "model": pl_module.model.state_dict(),
                        "best_mae": mae}, path)
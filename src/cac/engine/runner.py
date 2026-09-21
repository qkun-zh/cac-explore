"""Training runner — per-node training of FSC147 counting models.

Driven by a config file. ALWAYS smoke-tests one step first (fits in memory),
then runs the full budget (epochs or budget_seconds). The node's own sibling
`model.py` takes precedence over the registry builder.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger

from cac.calls import BestCheckpoint, EMACallback
from cac.config import load_config  # noqa: F401
from cac.data.pl_datamodule import FSC147DataModule
from cac.models.pl_module import CountingLit
import cac.hub as hub


def _model_from_cfg(cfg: dict[str, Any], node_dir: str | None = None,
                    built_model: Any = None):
    if built_model is not None:
        return built_model
    if node_dir:
        sibling = os.path.join(node_dir, "model.py")
        if os.path.exists(sibling):
            spec = importlib.util.spec_from_file_location("node_model", sibling)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["node_model"] = mod
            spec.loader.exec_module(mod)
            if hasattr(mod, "build_model"):
                return mod.build_model(cfg)
    from cac.models import make_model
    return make_model(cfg)


def _BudgetStop_hit(trainer: L.Trainer) -> bool:
    for c in trainer.callbacks:
        if isinstance(c, _BudgetStop) and c.hit:
            return True
    return False


def _FutilityStop_hit(trainer: L.Trainer) -> bool:
    for c in trainer.callbacks:
        if isinstance(c, _FutilityStop) and c.hit:
            return True
    return False


def run_train(cfg: dict[str, Any], log_dir: str, node_dir: str | None = None,
              smoke: bool = False, built_model: Any = None,
              epochs: int | None = None, budget_seconds: float | None = None,
              fast_dev_run: int | None = None, off: bool = False) -> dict[str, Any]:
    hub.setup_hf_env()
    cfg = dict(cfg)
    seed = int(cfg.get("seed", 20260830))
    L.seed_everything(seed, workers=True)
    if bool(cfg.get("deterministic", False)):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    use_ema = bool(cfg.get("use_ema", True))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    if off:
        print("--off passed; no training will run.")
        return {"off": True}

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    save_best = BestCheckpoint(log_dir)
    callbacks: list = [LearningRateMonitor("epoch")]
    if use_ema:
        callbacks.append(EMACallback(decay=float(cfg.get("ema_decay", 0.999))))

    model = _model_from_cfg(cfg, node_dir, built_model)

    def _mk_trainer(max_epochs: int, logger: Any, smoke_: bool) -> L.Trainer:
        cbs = [c for c in callbacks
               if not (logger is False and isinstance(c, LearningRateMonitor))]
        return L.Trainer(
            max_epochs=max_epochs, devices=1, accelerator="auto",
            precision="bf16-mixed" if smoke_ else cfg.get("precision", "16-mixed"),
            gradient_clip_val=grad_clip, callbacks=cbs, logger=logger,
            enable_progress_bar=not smoke_, enable_checkpointing=False,
            num_sanity_val_steps=0, enable_model_summary=not smoke_,
            default_root_dir=log_dir,
            limit_train_batches=4 if smoke_ else None,
            limit_val_batches=2 if smoke_ else None,
            fast_dev_run=int(fast_dev_run) if fast_dev_run else False,
        )

    smk = FSC147DataModule(cfg, smoke=True, img_size=384,
                           batch_size=min(int(cfg.get("batch_size", 16)), 4),
                           seed=int(cfg.get("seed", 20260830)))
    pl_smoke = CountingLit(cfg, model=model)
    _mk_trainer(max_epochs=2, logger=False, smoke_=True).fit(pl_smoke, datamodule=smk)
    print(f"[smoke] one step fits: ok (model={type(model).__name__})")
    if smoke:
        return {"smoke": True}

    max_epochs = epochs or int(cfg.get("epochs", 300))
    if budget_seconds:
        callbacks.append(_BudgetStop(budget_seconds))
    if bool(cfg.get("early_stop", False)):
        callbacks.append(EarlyStopping("val/mae", patience=8, mode="min", min_delta=1e-3))
    fbar = cfg.get("futility_bar", None)
    if fbar is not None:
        gates = [int(g) for g in str(cfg.get("futility_gates", "16,24")).split(",")]
        callbacks.append(_FutilityStop(save_best, fbar,
                                       ref_slope=float(cfg.get("futility_ref_slope", 0.06)),
                                       gates=gates,
                                       halt_from=int(cfg.get("futility_halt_from", 24)),
                                       margin=float(cfg.get("futility_margin", 2.0)),
                                       max_epochs=max_epochs))
    if int(cfg.get("val_every_n_epochs", 1)) != 1:
        callbacks.append(ModelCheckpoint(dirpath=os.path.join(log_dir, "ckpt/"),
                                         every_n_epochs=int(cfg.get("val_every_n_epochs", 1))))

    shutil.rmtree(os.path.join(log_dir, "tb"), ignore_errors=True)
    tb = TensorBoardLogger(log_dir, name="tb", version="t")
    pl = CountingLit(cfg, model=model)
    pl._best_sink = save_best
    dm = FSC147DataModule(cfg, smoke=False,
                          img_size=int(cfg.get("input_size", 384)),
                          batch_size=int(cfg.get("batch_size", 16)),
                          num_workers=int(cfg.get("num_workers", 4)),
                          augment=bool(cfg.get("augment", False)),
                          seed=int(cfg.get("seed", 20260830)))
    t_start = time.time()
    trainer = _mk_trainer(max_epochs=max_epochs, logger=tb, smoke_=False)
    trainer.fit(pl, datamodule=dm)
    return {"code": "ok", "log_dir": log_dir, "epochs": max_epochs,
            "best_mae": save_best.best, "best_epoch": save_best.best_epoch,
            "elapsed": time.time() - t_start,
            "n_epochs_done": trainer.current_epoch,
            "budget_hit": _BudgetStop_hit(trainer),
            "futility_hit": _FutilityStop_hit(trainer)}


class _FutilityStop(L.callbacks.Callback):
    """Bar-relative futility halt (AGENTS hard rule 16).

    Plateau early-stopping cannot catch hopeless-but-improving runs: H0011
    improved its val MAE every single epoch while running 4-5x short of the
    pace its bar needed, burning 8 dead epochs. This callback compares the
    REQUIRED per-epoch improvement against the best late slope ever observed
    in the lineage and halts when the booked bar is unreachable.
    Gate at ep16 WARNS only (mid-run jumps like H0012's ep15 discontinuity
    happen); halt authority from ep24. Installed ONLY when futility_bar is set
    — canonical baselines always run without it (they define the bars). A
    halted run keeps its best.pth, so eval_test + mechanism reads still work;
    the verdict false-shape is 'futility refutation'."""

    def __init__(self, tracker, bar, ref_slope=0.06, gates=(16, 24),
                 halt_from=24, margin=2.0, max_epochs=32):
        self.tracker = tracker
        self.bar = float(bar)
        self.ref = float(ref_slope)
        self.gates = tuple(int(g) for g in gates)
        self.halt_from = int(halt_from)
        self.margin = float(margin)
        self.max_epochs = int(max_epochs)
        self.hit = False

    def on_validation_epoch_end(self, trainer, pl_module):
        ep = int(trainer.current_epoch) + 1
        if ep not in self.gates or self.hit:
            return
        best = float(self.tracker.best)
        if not (best < float("inf")):
            return
        required = (best - self.bar) / max(1, self.max_epochs - ep)
        feasible = required <= self.margin * self.ref
        verdict = "KEEP" if feasible else ("HALT" if ep >= self.halt_from else "WARN (no halt before ep24)")
        print(f"[futility] ep{ep}: best {best:.4f} bar {self.bar:.4f} "
              f"need {required:.4f}/ep vs ref {self.ref:.4f}x{self.margin} -> {verdict}",
              flush=True)
        if not feasible and ep >= self.halt_from:
            trainer.should_stop = True
            self.hit = True


class _BudgetStop(L.callbacks.Callback):
    """Hard wall-clock ceiling — arguably the most important cipher of the paper:
    exploration never spends unbounded time on a branch."""
    def __init__(self, budget: float):
        self.budget = budget
        self.hit = False
        self._t0 = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if time.time() - self._t0 > self.budget and not trainer.sanity_checking:
            trainer.should_stop = True
            self.hit = True
            print(f"[budget] exceeded {self.budget:.0f}s — stopping early")


def checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
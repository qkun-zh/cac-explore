# N0010_h0009 — qualitative feedback (text-log training-dynamics read)

- **Node:** N0010_h0009 (H0009 `use_densexp` dense-tail expert, solo on `use_simprior=true`) · **Parent:** N0002_h0001 (22.5641 @ep30, bar 22.26).
- **Sources read:** `idea.md`; `result.json` (best 24.0500 @ep32, 32/32ep, `budget_hit=false`, `code=ok`, elapsed 1789.4/1794.9s); `info.json` (`done`); `model.py` gate wiring (CountingHead `use_densexp` → `dens + densexp(cond_map)`, `DenseTailExpert` zero-init out + detached GAP gate); server text log `/tmp/train-b4-n0010.log` (144378 B, 566 lines, single-grep reads); server TB scalars full 32-epoch series (`val/mae`, `val/rmse`, `train/loss_epoch`, `train/mae`). No stored feature dumps exist (AGENTS §9); this read is text-log + TB-scalar only, leaving the numeric verdict to quant and mechanism attribution to causal.

## 1. Trajectory shape: smooth healthy descent to a flat plateau — no pathology, no drama

- TB `val/mae` (32 pts): 270.56 → 138.13 → 94.79 → 70.80 → 57.45 → 47.21 → 39.83 → 34.74 → 31.48 → 29.67 → 27.98 → 27.06 → 26.53 → 26.08 → 25.72 → 25.30 → 24.97 → 24.70 → 24.51 → 24.37 → 24.27 → 24.20 → 24.16 → 24.12 → 24.10 → 24.08 → 24.07 → 24.07 → 24.06 → 24.06 → 24.05 → 24.05. Monotonic decrease every single epoch, zero spikes, zero NaN, zero U-turns. First ~10 epochs do the heavy lifting (270 → 29.7); epochs 11–24 grind (28.0 → 24.1); epochs 25–32 are a dead-flat plateau (24.10 → 24.05, −0.05 over 7 epochs).
- TB `train/loss_epoch`: 19.84 → 12.51 → 11.59 → 10.08 → 8.34 → 8.59 → 8.30 → 8.31 → 8.41 → 7.56 → 6.71 → 6.64 → 6.82 → 7.04 → 6.76 → 5.83 → 5.32, then pinned at ~5.32–5.35 from ep17 to ep32. Train and val agree: converged to a stable floor by ep~25, then idle.
- Best at ep32 (edge-best) therefore means **plateau-edge best, not selected minimum**: the curve never turned up, so the checkpoint is trustworthy, but the slope at the edge (~−0.005/ep) says extra epochs would not close the +1.49 gap to the parent — this is a finished run that lost, not an interrupted one.
- Family contrast (shape only): none of the known failure signatures appear — no N0007-style train stall (~14/val ~48 attractor), no N0008-style synchronized train+val NaN wall, no N0005/N0006-style budget timeout. This is the boring-healthy shape the refuted siblings never showed.

## 2. Clean vs dirty: CLEAN run, honest miss

- Clean by every text-log check: single-grep anomaly scan returns **0 hits** for `Traceback|Error|NaN|nan|BudgetStop|CUDA out of memory|Misconfiguration`; `code=ok`, `budget_hit=false`, all 32 epochs stepped, `Trainer.fit stopped: max_epochs=32 reached`, smoke `[smoke] one step fits: ok (model=Counter)`.
- Run hygiene verified in the header: `Seed set to 20260830` (canonical), RTX3060 + AMP (bfloat16/16bit-mixed), params `Total 31.4M / Trainable 3.5M / Non-trainable 27.8M` with 82 train / 151 eval modules — consistent with a ~4.3k-parameter appended expert, no envelope breach.
- The only log dirt is the standing benign warning set, identical to the family: non-writable-NumPy flood (~500 lines, dataloader env noise), `num_workers` bottleneck tip ×2, `151 module(s) in eval mode` (the frozen backbone, expected), `MeanMetric.compute before update`, float32-matmul-precision tip. Nothing node-specific, nothing actionable.
- Same runner-logging gap as the sibling: the text log carries no per-epoch scalars (rich progress bars overwrite; only the final `Epoch 31/31 … val/mae:` line survives, value truncated in capture). The entire trajectory above comes from TB scalars — from stdout alone this run would read as "smoke ok → done, 24.05", with no way to see the plateau.

## 3. Two shape tells worth logging (no mechanism verdict)

- **MAE flat, RMSE still falling:** `val/rmse` decreases ALL 32 epochs (281.76 → … → 85.24 → 84.15, still −1.1 over the last 8) while `val/mae` is frozen at 24.1→24.05. Late training was still fixing large errors while the median refused to move — the mirror image of N0008's pre-collapse RMSE-rises/MAE-falls decoupling. For a node whose entire pitch was "fix the catastrophic tail", the tail metric moving and the headline metric not is worth handing to causal, not solving here.
- **`train/mae` all-NaN (32/32):** repeats the pre-existing logging bug already noted on N0008/parent — deprives qual reads of a train-side counting signal again. Still unfixed, still not node-specific.
- **Near-ceiling elapsed, healthy speed:** 1789–1795s vs τ_max 1800s (+~40s over parent's ~1750s, final-epoch step 45s @ 5.03it/s). Close to the wall but `budget_hit=false` and the plateau says the time went to idle epochs, not to expert overhead distress. Not an anomaly, logged so the next reader does not misread "near-timeout" as "almost-failed".

## One-line assessment

- Clean, complete, honestly-lost run: smooth monotonic descent to a flat 24.05 @ep32 plateau (+1.49 over parent 22.56, −1.79 off the 22.26 bar) with no spikes, NaNs, stalls, or timeouts — the +1.49 is a quiet mechanism miss on a healthy curve, not a broken run, with MAE-plateau/RMSE-still-falling as the one shape thread for causal.

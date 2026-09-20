# Qualitative feedback — N0002_h0001 (`use_simprior`, child of N0001_champion)

Text-log analysis per AGENTS §9 (replaces VLM heatmap reads). Sources read
directly: `chain_run.log` (566 lines) on cac-server, `run_smoke.log`,
`info.json`, TensorBoard scalars from `run/latest/tb/t`, and `ls -la`
of `run/latest/`. Numbers below are characterisation only; MAE tables
belong to quant.

## Training character: smooth cold-start descent, no erratic phases

- The val curve from TB (`val/mae`, 32 points, one per epoch) is a smooth,
  essentially monotonic descent: a steep cold-start plunge over epochs 0–6
  (153 → 70 → 49 → 39 → 34 → 30 → 28) followed by a long gentle glide that
  bottoms at epoch 30 and ticks up only marginally at epochs 31–32. No
  spikes, no plateaus-then-cliffs, no mid-run regressions.
- `train/loss_epoch` mirrors this exactly: 38.0 → 8.7 → 8.4 → … → 2.62,
  monotonically decreasing every single epoch. Step-level `train/loss`
  (145 logged points) is noisy in the normal per-batch way (range ~1–30
  after the first step) with no sustained blowups.
- Early-epoch behavior is exactly what the zero-init residual predicts:
  epoch-0 val is far from the trained parent (cold head, random decoder —
  zero-init guarantees step-0 equivalence to the parent *architecture at
  init*, not to trained parent weights), and the new branch introduces no
  instability signature — no loss spike, no NaN, no stalled first epochs.
  The prior's `out` projection starts at zero and the optimizer picks it up
  gradually, consistent with the smooth rather than jumpy val trajectory.
- Late-run shape: `val/rmse` bottoms around epoch 25 (~88.9) and drifts up
  slightly to ~89.2 by epoch 31 while `val/mae` keeps improving to epoch 30
  — a mild, small-magnitude tail divergence (large-count images softening
  while typical images still gain). Benign at this magnitude; best-checkpoint
  selection at epoch 30 already handles it.

## Warnings: all benign, all pre-existing protocol noise

- `torchmetrics ... MeanMetric compute called before update` (once, at sanity
  check) — known benign wording warning, fires before any real update.
- Lightning `Found 151 module(s) in eval mode at start of training` — expected:
  the backbone is frozen by regime, so eval-mode modules are intentional.
- CUDA Tensor-Cores `float32_matmul_precision` tip, `num_workers` bottleneck
  tip, litlogger tip — stock Lightning/PyTorch advice, no action.
- `fsc147.py:70` NumPy non-writable-tensor `UserWarning` — pre-existing data
  loader warning, self-suppresses after first trigger; no data-path effect.
- Zero `traceback`, zero `error`, zero OOM/CUDA failures, zero `_BudgetStop`.

## Smoke-vs-train consistency: clean

- `run_smoke.log`: `[smoke] one step fits: ok (model=Counter)`, 31.3M total /
  3.5M trainable, `SMOKE_RESULT {'smoke': True}`. The full run then executed
  the same `Counter` build for all 32 epochs (`max_epochs=32 reached`,
  `budget_hit: false`, 1755 s wall under the 1800 s ceiling) with seed
  20260830 logged at the top. Smoke and train agree on model, seed, and AMP
  (bfloat16/16-mixed) configuration.

## Checkpointing evidence

- `run/latest/best.pth` present (125 MB, mtime 13:38, two minutes before
  `result.json` at 13:40 — written by the run, not hand-placed).
- `result.json`: `code ok`, `n_epochs_done 32`, `best_epoch 30`,
  `best_mae 22.564`; `info.json`: `status done`, `best_metric` matches
  `result.json`. Epoch-30-best with two further epochs completed rules out
  edge-of-training truncation.

## Anomalies: none material (one logging artifact noted)

- TB scalar `train/mae` logs NaN for all 32 epochs while `train/loss_epoch`
  and `val/mae` are healthy — a metric-logging artifact (consistent with the
  compute-before-update warning path), not an optimization pathology, since
  loss descends monotonically and validation improves throughout. Quant and
  causal should not lean on the `train/mae` TB series; it carries no signal.
- Otherwise: no NaN/inf in any loss or val series, no restarts, no budget
  interruption, no checkpoint anomalies.

## Qualitative confidence

This reads as a clean full-protocol pass: green smoke, 32/32 epochs inside
budget, smooth spike-free curves, benign-only warnings, and a properly
written epoch-30 best checkpoint.

Verdict in one sentence: best val MAE 22.564 at epoch 30 beats the live
parent 23.293 by ~0.73, clearing the 22.99 falsifier bar with smooth,
anomaly-free training character throughout.

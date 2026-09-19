# Quantitative feedback — N0004_h0010 (H0010 per-exemplar exemplar gate + H0009 image-global channel gate)

## 1. Verified numbers
| Quantity | Value | Source |
|---|---|---|
| best val MAE | **23.38243293762207** | result.json `best_mae` |
| best epoch | **32** (= max_epochs, best-at-last-epoch again) | result.json `best_epoch` |
| epochs done | 32 / 32 | result.json `epochs`, `n_epochs_done`; `Trainer.fit` stopped `max_epochs=32` (run_node.log:552) |
| budget_hit | **false** | result.json `budget_hit` |
| elapsed | **1707.76 s** train clock (result.json `elapsed`); 1713.2 s wall (`elapsed_s`) | result.json |
| budget ceiling | τ_max 1800 s → margin **≈ 87 s (4.8%)** of wall — run completed, not truncated | AGENTS.md §4 |
| params TOTAL | **31.4 M** (< 32 M gate OK) | run_node.log:35 |
| Trainable / frozen | **3.5 M / 27.8 M** | run_node.log:33-34 |
| gate params | channel gate **33,024** = Linear(128→256), w 128·256 + b 256 (model.py:160); exemplar gate **257** = Linear(256→1), zero-init weight, bias 3.0 (model.py:163-165) → **+33,281** over parent, does not move the rounded 31.4 M total | model.py + config.toml |
| result.json form | full record (`"code":"ok"`, real best_mae), **not** a `{"smoke":true}` overwrite | result.json |
| smoke | `[smoke] one step fits: ok (model=Counter)` | run_node.log:24 |
| checksums | config `644696b176c69c14`, model `9961590b5ab9b0f4` | result.json |

**Seeded parent baseline (N0001): best val MAE 21.458858489990234 @ ep32, 32/32** (tree/N0001_champion/result.json).

**Δ = 23.38243293762207 − 21.458858489990234 = +1.9235 → child-WORSE by 1.92 MAE** (ratio 1.0896 ≈ +9.0%). Both gates toggled on together produce a clearly negative joint result.

## 2. Per-epoch val/mae evidence
- The log flushes only the FINAL progress bar: `Epoch 31/31 … val/mae: 23.382` (run_node.log:553-554), matching result.json `best_mae` (23.3823… at step 0 — internally consistent).
- **No `tb/` directory is synced locally** for this node (`run/latest/` holds only `run_node.log`), so the per-epoch scalar series is **not recoverable from the local tree**; the series exists only server-side (`/data/cac/tree/N0001_champion/N0004_h0010/run/latest/tb`). No plateau/overfit claim can be made from local files.
- What is verifiable: `best_epoch == 32 == last epoch` ⇒ best-at-last-epoch again (same pattern as parent and sibling). Selection therefore carries zero robustness margin — best.pth is trivially the final checkpoint, chosen on val/mae only (best_checkpoint.py:21-29, pl_module.py:98-101).

## 3. Joint-test honesty — the Δ cannot be split
- One run deploys H0010 AND H0009; the observed +1.9235 is the **joint** effect (Δ_A + Δ_B + interaction), not attributable to either gate.
- Both falsifiers are the **same bar** (idea.md:7,10): support iff final val MAE ≥ 0.30 **lower** than parent ⇒ required ≤ 21.1589 (= 21.4589 − 0.30). Observed 23.3824 is **2.2236 above the bar** (~7.4× the bar width); the child must improve 2.2236 (≈9.5% of itself) just to touch it — far from the −0.30 the hypotheses claim.
- Attribution arithmetic: for EITHER gate to have met its −0.30 bar, the other gate plus the interaction would have to contribute ≥ +2.22 of net improvement to offset the observed joint sign. The single-observation design admits no such split.
- Cross-node context (same seed 20260830, same protocol — indicative, not a split test): H0009-only sibling N0003 landed at 22.6076, Δ = +1.1487 vs 21.4589 (itself 1.4487 above its own bar); the joint node is **+0.7749 worse than H0009-only** (23.3824 − 22.6076), so layering the per-exemplar gate on the channel gate looks net-harmful on top of an already-failing H0009.

## 4. Log notes vs sibling N0003
- Warning set is byte-similar to N0003: MeanMetric compute-before-update (run_node.log:15), repeated non-writable-numpy-array warnings (fsc147.py:70), 151 modules in eval, TF32/num_workers tips. Smoke + stop lines at analogous positions.
- **Key difference:** the sibling's `run/latest/` has a local `tb/` dir (tfevents) from which its quant.md recovered per-epoch scalars; **N0004's does not** — per-epoch verification here requires pulling the server tb dir before it is overwritten or pruned.
- Params identical at rounded precision (31.4 M / 3.5 M trainable) — the two toggles add 33,281 params but both nodes pass the <32 M gate.
- The final progress-bar flush matches result.json exactly; no divergence markers between log and ledger numbers (no budget-stop, no timeout lines).

**No confidence values computed; nothing committed or pushed.**
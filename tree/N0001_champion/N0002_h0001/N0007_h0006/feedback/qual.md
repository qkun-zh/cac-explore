# N0007_h0006 — qualitative feedback (text-log read, H0006 `use_h2pool`)

- **Node / parent:** N0007_h0006 (solo `use_h2pool=true`) under N0002_h0001 (22.5641 @ep30). **Result:** 47.8439 @ep32, status `done`, 32/32 epochs, `budget_hit=false`, elapsed ≈1804 s. Falsifier bar 22.26 missed by ≈25.6.
- **Sources read:** `idea.md` (§2–§4), `info.json`, `result.json` (local), server TB scalars `val/mae`, `val/rmse`, `train/loss_epoch` for child vs parent via `EventAccumulator` (PYTHONPATH=/data/cac/src, `setup_hf_env` first), `chain_run.log` grep, `/tmp/train-b3-n0007.log` grep (tail shows clean `max_epochs=32 reached`, no traceback).

## Trajectory narrative (shape, not table)

The child never visits the parent basin. From the very first validation point it is on a different, worse level and its learning curve has a different shape: a steep early fall that arrests into a long flat drift, while the parent keeps descending all the way to ep30.

- **Separation is immediate — epoch 1 (step 227):** child val/mae 219.64 vs parent 153.29 (gap ≈+66). By ep2 the gap is 104.46 vs 70.05, by ep3 76.00 vs 48.79, by ep5 53.33 vs 33.50, by ep8 48.54 vs 27.27. The ordering never flips at any of the 32 points — this is not a late divergence or a mid-run crossing, it is day-one separation.
- **The stall, not the fall, is the story:** after ep8–ep10 (≈48.3) the child barely moves — it drifts 48.28 → 47.84 over the remaining ~20 epochs with best=last epoch. The parent at the same stage is still improving (26.28 → 22.56 over ep9–ep30). So the child curve reads as *early partial descent then plateau*, the parent as *sustained descent*.
- **Train tells the underfit half:** child `train/loss_epoch` parks at ≈13.9–14.3 from ep2 to ep32 (ep2 14.29, ep8 14.00, ep32 13.93) while the parent descends 8.65 → 2.62 over the same span. Train-worse AND val-worse jointly, by large margins (≈5× on train, ≈2× on val at end). There is no overfit signature here (no falling train + rising val); it is an optimization stall at a bad level.
- **One odd first step:** at ep1 the child train loss is actually LOWER than the parent's (19.62 vs 38.02) while its val is much WORSE (219.64 vs 153.29). By ep2 the relation flips to train-worse too (14.29 vs 8.65) and stays there. Read plainly: the new branch pulls gradients somewhere else from the first updates, buys a small early train discount at the cost of generalization, then stalls above the parent on both.
- **Error-profile shape:** child val RMSE ends 128.20 vs parent 89.09; RMSE/MAE ≈2.68 vs ≈3.95. Both MAE and RMSE flat-line together — no joint MAE+RMSE descent of the kind `idea.md` §4 would accept as small-instance rescue. The claimed mechanism leaves no footprint in the curve shape.

## Divergence vs underfit verdict

**Underfit with early divergence — not a late-run divergence, not a clean null.** Rationale, kept to what the logs show: (a) the gap opens at epoch 1 and the train curve never tracks the parent, so the run did not "leave" a shared trajectory late; (b) train parked ≈5× above the parent rules out a clean null — a branch that stayed near its zero-init would reproduce roughly the parent train level (≈2.6), whereas 13.9 means the added path engaged and held optimization at a worse stationary point (whether via its own `out` moving off zero early or via shared-`qproj` gradient coupling is mechanism attribution for causal, not this note; the TB has no `h2pool.out`/temp tags to separate them). Either way the log shape is engaged-but-harmful stall, N0006-flavored in outcome though much larger in magnitude (+25.28 vs +1.02).

## Clean / dirty

**Clean.** Full 32/32 epochs, `budget_hit=false`, best=ep32 (plateau-trustworthy, not a frozen spike), no NaN in any `val/mae`, `val/rmse`, or `train/loss_epoch` point, no exception/traceback/OOM in the grepped logs, smoke passed (31.4 M total, ≤32 M cap). The only NaN tag is `train/mae`, which is NaN at every step in BOTH child and parent — a logging artifact, not a run anomaly. No restart/resume lines, no `_BudgetStop`, no dirty early-stop to discount.

## Anomalies (nothing load-bearing)

- The ep1 inverse signature above (train-better/val-worse for exactly one epoch) is the only unusual point; everything after is monotone-ish stall.
- Val/RMSE both creep DOWN very slowly to the last epoch (no uptick), so there is no overfitting confound to the verdict.
- Idea-predicted diagnostics (`h2pool.out` norm, temp trajectory, pool-attention entropy, §4) are not in the logged tag set (`epoch, hp_metric, lr-AdamW, train/loss, train/loss_epoch, train/mae, val/mae, val/rmse` only) — noted so synthesis does not claim them observed.

## Takeaway for synthesis

Refute read from the curve alone: the h2 branch did not quietly do nothing — it moved optimization to a worse basin on epoch 1 and kept it there for 32 epochs. Redirect per `idea.md` §2c (Direction 2 calibration), never a wider-bank retry; any future bank proposal must explain why it would not repeat this epoch-1 capture.

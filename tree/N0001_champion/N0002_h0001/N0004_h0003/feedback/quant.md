# feedback/quant.md — N0004_h0003 (H0003 `use_padapt`)

- **Role:** Quantitative Feedback subagent · **Node:** N0004_h0003 · **Parent:** N0002_h0001
- **Date:** 2026-09-20 · **Seed:** 20260830 (fixed) · **Protocol:** canonical 32ep / 1800s, augment=false, EMA eval
- **Verdict:** **REFUTE** (decisive).

## Sources (all numbers below were read from these; nothing recollected)

| Artifact | Path |
|---|---|
| Child result | `/data/cac/tree/N0001_champion/N0002_h0001/N0004_h0003/result.json` |
| Child info | `/data/cac/tree/N0001_champion/N0002_h0001/N0004_h0003/info.json` |
| Child TB events | `/data/cac/tree/N0001_champion/N0002_h0001/N0004_h0003/run/latest/tb/t/events.out.tfevents.1789885586.bwdqkrimegiarobw-snow-59d754d49d-f7svh.9960.0` |
| Child chain log | `/data/cac/tree/N0001_champion/N0002_h0001/N0004_h0003/chain_run.log` |
| Child best ckpt | `/data/cac/tree/N0001_champion/N0002_h0001/N0004_h0003/run/latest/best.pth` |
| Parent result | `/data/cac/tree/N0001_champion/N0002_h0001/result.json` |
| Parent TB events | `/data/cac/tree/N0001_champion/N0002_h0001/run/latest/tb/t/events.out.tfevents.1789881056.bwdqkrimegiarobw-snow-59d754d49d-f7svh.1670.0` |
| Parent best ckpt | `/data/cac/tree/N0001_champion/N0002_h0001/run/latest/best.pth` |
| Booking / bar | node `idea.md` (bar final val MAE ≤ 22.26 vs live parent 22.5641) |

Curves pulled with `EventAccumulator` directly on the two events files above (`val/mae`, `val/rmse`, `train/loss_epoch`, `lr-AdamW`; 32 points each, same tag set for both nodes). `scripts/curve.py` was **not** used for the nested path to avoid the known path ambiguity; the direct dump is the primary evidence.

## 1. Parent-vs-child summary

| Quantity | Parent N0002_h0001 | Child N0004_h0003 | Δ (child−parent) |
|---|---:|---:|---:|
| Best val MAE | **22.5641** | **25.5661** | **+3.0020** |
| Best epoch | 30 | 13 | −17 |
| Final val MAE (ep32) | 22.5705 | 26.1668 | +3.5963 |
| Epochs completed | 32 / 32 | 32 / 32 | 0 |
| `elapsed` (runner internal, s) | 1749.53 | 1767.63 | +18.10 |
| `elapsed_s` (wall, s) | 1755.0 | 1773.3 | +18.3 |
| `budget_hit` (1800 s cap) | false | false | — |
| `config_sha256` | 4d4c9baec459f657 | 769ddfa05d25dfa9 | differs |
| `model_sha256` | 611be2a9d70768b5 | 93c221209869fe69 | differs |
| Params (from `best.pth`, Σ numel) | 31,349,540 | 31,411,748 | +62,208 |
| `save_epoch` in best.pth | 30 | 13 | — |

Child best-MAE exact value: `25.566089630126953` (`result.json:5`, `info.json` best_metric). Parent best exact: `22.564146041870117`.

## 2. Full child curve, paired per-epoch vs parent

Δ = child val MAE − parent val MAE (positive = child worse). All 32 epochs shown; both curves are 32/32.

| ep | parent val MAE | child val MAE | Δ | parent val RMSE | child val RMSE | parent train loss | child train loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 153.2874 | 152.7835 | -0.5039 | 174.3924 | 175.7774 | 38.0236 | 41.5595 |
| 2 | 70.0534 | 69.2195 | -0.8339 | 116.8055 | 120.8460 | 8.6532 | 9.0033 |
| 3 | 48.7882 | 49.6918 | +0.9036 | 106.9041 | 112.9911 | 8.4087 | 8.3664 |
| 4 | 39.1216 | 40.8047 | +1.6831 | 102.9298 | 109.5301 | 7.7869 | 7.4904 |
| 5 | 33.5010 | 35.1977 | +1.6968 | 100.0118 | 105.9347 | 6.9885 | 7.4195 |
| 6 | 30.2952 | 31.3338 | +1.0386 | 98.0300 | 102.4284 | 6.8799 | 7.0246 |
| 7 | 28.4055 | 28.8931 | +0.4876 | 97.2657 | 99.8754 | 6.8469 | 6.5575 |
| 8 | 27.2659 | 27.4816 | +0.2157 | 96.8390 | 98.6117 | 6.5521 | 6.4535 |
| 9 | 26.2845 | 26.6618 | +0.3772 | 96.3401 | 97.7656 | 6.1468 | 6.0815 |
| 10 | 25.4092 | 25.9875 | +0.5783 | 95.3331 | 96.8147 | 5.4919 | 5.6277 |
| 11 | 24.6751 | 25.6881 | +1.0130 | 94.0627 | 96.3498 | 5.4525 | 5.7303 |
| 12 | 24.2563 | 25.5786 | +1.3223 | 93.2005 | 96.1416 | 5.5983 | 5.5633 |
| **13** | 23.9073 | **25.5661** | +1.6588 | 92.3794 | **96.0988** | 5.2261 | 5.5819 |
| 14 | 23.5598 | 25.5921 | +2.0323 | 91.4805 | 96.1379 | 5.0677 | 5.5878 |
| 15 | 23.4425 | 25.6438 | +2.2013 | 91.0531 | 96.1861 | 4.6221 | 5.5771 |
| 16 | 23.2482 | 25.7179 | +2.4698 | 90.5843 | 96.2776 | 4.5107 | 5.5691 |
| 17 | 23.0597 | 25.7898 | +2.7302 | 90.1668 | 96.3578 | 4.5179 | 5.5924 |
| 18 | 22.9224 | 25.8624 | +2.9400 | 89.9383 | 96.4479 | 4.1466 | 5.5732 |
| 19 | 22.9487 | 25.9220 | +2.9732 | 89.7513 | 96.5080 | 4.1724 | 5.5926 |
| 20 | 22.9341 | 25.9757 | +3.0417 | 89.6870 | 96.5558 | 3.9857 | 5.5790 |
| 21 | 22.9015 | 26.0180 | +3.1165 | 89.5291 | 96.6112 | 3.7517 | 5.5859 |
| 22 | 22.8883 | 26.0532 | +3.1650 | 89.4606 | 96.6604 | 3.7048 | 5.5840 |
| 23 | 22.7773 | 26.0818 | +3.3045 | 89.1640 | 96.6845 | 3.4890 | 5.5825 |
| 24 | 22.7200 | 26.1028 | +3.3829 | 89.1040 | 96.7128 | 3.5126 | 5.5878 |
| 25 | 22.6848 | 26.1212 | +3.4363 | 88.9863 | 96.7283 | 3.2047 | 5.5943 |
| 26 | 22.6515 | 26.1298 | +3.4783 | 88.9433 | 96.7504 | 3.1010 | 5.5757 |
| 27 | 22.6285 | 26.1365 | +3.5079 | 88.9668 | 96.7662 | 3.0211 | 5.5873 |
| 28 | 22.5923 | 26.1453 | +3.5530 | 88.9234 | 96.7755 | 2.8041 | 5.5896 |
| 29 | 22.5659 | 26.1528 | +3.5869 | 88.9828 | 96.7823 | 2.7641 | 5.5906 |
| 30 | **22.5641** | 26.1581 | +3.5939 | 89.0939 | 96.7887 | 2.7040 | 5.5911 |
| 31 | 22.5733 | 26.1628 | +3.5896 | 89.1886 | 96.7986 | 2.6526 | 5.5872 |
| 32 | 22.5705 | **26.1668** | +3.5963 | 89.2400 | 96.8000 | 2.6183 | 5.5889 |

## 3. Peak epoch and the monotone rise afterwards

- Child best val MAE = **25.5661 at ep13**; saved to `best.pth` (`epoch: 13`, `best_mae: 25.566089630126953`).
- From ep13 to ep32 the child val MAE is **strictly monotone non-decreasing** (checked pair-wise; no lower value anywhere after ep13): 25.5661 → 25.5921 → 25.6438 → 25.7179 → 25.7898 → 25.8624 → 25.9220 → 25.9757 → 26.0180 → 26.0532 → 26.0818 → 26.1028 → 26.1212 → 26.1298 → 26.1365 → 26.1453 → 26.1528 → 26.1581 → 26.1628 → **26.1668**, a rise of **+0.6007** over 19 epochs.
- The parent has the opposite shape: it keeps descending to **22.5641 @ep30** and stays flat at 22.57 to ep32. The child peaks 17 epochs earlier and at a level **+3.00** worse, then degrades — the adapter did not just fail to help, it made late-epoch val monotonically worse.

## 4. Paired per-epoch comparison (feasible; same seed 20260830, same epochs, both curves complete)

- Child beat parent on **2 / 32** epochs only (ep1 −0.5039, ep2 −0.8339, both within warm-up noise of the shared loss landscape).
- Child worse on **30 / 32** epochs; from ep3 onward always worse.
- Mean Δ = **+2.1668**; min Δ = −0.8339 (ep2); max Δ = **+3.5963** (ep32).
- Val RMSE at best epoch: child 96.0988 vs parent 89.0939 → **+7.0049**. Child val RMSE is essentially frozen near ~96 from ep10 (96.8147) to ep32 (96.8000) while parent descends to 89.09–89.24.

## 5. Verdict — REFUTE, exact margins

- **vs live parent (pre-registered paired contrast):** child best 25.5661 − parent best 22.5641 = **+3.0020 WORSE** (exact: 25.566089630126953 − 22.564146041870117 = 3.001943588256836). Sign is opposite to the claimed improvement and the magnitude is ~10× the 0.30 confirmation bar.
- **vs pre-registered bar ≤ 22.26:** child best 25.5661 − 22.26 = **+3.3061 ABOVE** the bar (exact: 25.566089630126953 − 22.26 = 3.3060896301269516). The child's best-ever epoch misses the bar by more than the entire expected effect.
- Also misses on the final-epoch reading: final 26.1668 vs parent best 22.5641 = **+3.6026**.
- Per AGENTS §6 / idea §6, this is a same-seed paired contrast against the live parent; the falsifier "final val MAE not at or below 22.26" is met by a wide margin. **H0003 is refuted for this variant.** No re-run / outcome shopping.

## 6. Numeric anomalies and use-evidence

1. **The adapter learned and actively hurt — it was not ignored.** `best.pth` at the best epoch (ep13) contains non-trivial `head.padapt.*` weights: `out.weight` norm **2.4946** (absmax 0.0620), `out.bias` norm **0.2905**; `attn.in_proj_weight` norm 9.1989, `eproj.weight` norm 5.0847, `fproj.weight` norm 4.5811. `out` (zero-init at t=0) has moved well off zero, so the adaptation was on and the optimizer chose a state that lowered val MAE — the failure is mechanistic, not "the switch was never exercised" (the idea §6 uninformative-null caveat does **not** apply). Parent `best.pth` has zero `padapt` keys.
2. **Train loss plateau after the peak.** Child `train/loss_epoch` falls 5.5819 (ep13) → then is ~flat for 19 epochs: 5.5878, 5.5771, 5.5691, 5.5924, 5.5732, 5.5926, 5.5790, 5.5859, 5.5840, 5.5825, 5.5878, 5.5943, 5.5757, 5.5873, 5.5896, 5.5906, 5.5911, 5.5872, 5.5889 (ep32). The parent, over the same window, drops 5.2261 → 2.6183. So the child's optimizer stopped reducing even the training objective while val MAE rose — consistent with the adapter saturating/destabilizing the head rather than fitting harder.
3. **Val RMSE decoupled from MAE.** Child best-ep RMSE 96.0988 is ~+7.0 above parent (89.0939); across the whole run the child RMSE sits ~96.0–96.8 and barely improves after the first ~10 epochs, whereas parent RMSE reaches 88.92–89.24. With MAE ≈ 26, the child ratio RMSE/MAE ≈ 3.7 vs parent ≈ 3.9 — the regression is broad, not a few catastrophic images.
4. **Param delta exactly as predicted, budget not hit.** Child ckpt total 31,411,748 vs parent 31,349,540 = +62,208 (idea §3 predicted 62,208 / 31,411,748; matches exactly). Child `elapsed_s` 1773.3 s ≤ 1800 s cap, `budget_hit=false`; 32/32 epochs, "Epoch 31/31 … 0:00:45, 5.06 it/s" in `chain_run.log`. The ~+18 s over the parent is not a timeout event.
5. **Harness logging artifact (not child-specific).** `train/mae` is `NaN` for all 32 epochs in **both** parent and child TB events; `train/loss_epoch` is finite for both. Pre-existing metric-logging behavior, worth noting only so it is not misread as a child-side NaN — no NaN/Inf appears in `chain_run.log` (grep count 0), and `val/mae`/`val/rmse`/`train/loss_epoch` are finite throughout.
6. **Config delta is exactly one switch.** Child `config.toml` = parent + `use_padapt = true` (line 28); `use_simprior=true`, `use_gca=true`, `use_xscale=true`, `seed=20260830`, `lr=1e-3`, `batch_size=16`, `epochs=32` identical. Cosine LR identical (1e-3 → 3e-6, ep13 = 6.92e-4 in both). The only architectural difference is the appended `PrototypeAdapter`, consistent with the shas differing.

## 7. Traceability / reproducibility

- Child best/final/epoch/seconds/budget/shas: `result.json` fields `best_mae`, `best_epoch`, `elapsed`, `elapsed_s`, `budget_hit`, `config_sha256`, `model_sha256` + `info.json`.
- Curve values: direct `EventAccumulator` dump of the two `events.out.tfevents.*` files above (child .9960.0, parent .1670.0), 32 `val/mae`, 32 `val/rmse`, 32 `train/loss_epoch`, 32 `lr-AdamW` each.
- Margins recomputed from the exact floats (IEEE double), not rounded values: +3.001943588256836 and +3.3060896301269516.
- Param counts and `padapt` norms: `torch.load(..., map_location="cpu")` on the two `run/latest/best.pth` files; child meta epoch 13 / best_mae 25.566089630126953, parent meta epoch 30 / best_mae 22.564146041870117.
- No file outside this `feedback/quant.md` was written; nothing committed.

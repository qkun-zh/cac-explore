# Quantitative feedback — N0022_h0021 (H0021 `use_exkern`)

Child endpoint: `result.json` best_mae 21.9284 @ep24, best_epoch 24, n_epochs_done 24/32, futility_hit true, budget_hit false, cfg_overrides `{augment: true, futility_bar: 19.0431}`.
Parent: N0015_h0014, seed 20260830, v2 protocol, EMA val MAE 19.3431 @ep32 (booked live parent).
Futility gate (ledger): best 21.9284@24, need 0.367/ep → HALT at ep24, saved ~8 epochs.

## 1. Verdict table — every booked bar

Bars from `idea.md` line 3 (triple conjunctive, bound to live parent 19.3431):

| Bar (booked) | Threshold | Child value | Parent value | Pass/Fail |
|---|---|---|---|---|
| val EMA MAE | ≤19.0431 | 21.9284 | 19.3431 | **FAIL** (+2.5853 vs parent, +2.8853 vs bar) |
| dense gt>500 MAE | <330.81 | 454.4514 | 330.8131 | **FAIL** (+123.6383) |
| sparse gt<50 MAE | ≤5.45 | 5.7913 | 5.8136 | **FAIL** (+0.3413 vs bar; −0.0223 vs parent = flat) |
| R2 tail transfer (8 ids) | tail ΣAE < parent | 4728.7 | 4238.5 | **FAIL** (+490.2, worse) |
| R1 mechanism `w_x` | >0 decisive at end (P1) | −0.0398 | 0 (init) | **FAIL** (F1 quiet-null: never engaged positive) |

Triple conjunctive gate: 0/4 numeric bars passed. Run futility-halted at ep24; all verdict reads taken from best.pth @ep24 (ckpt_epoch 24 in `val_result.json`).

Cross-checks (local): `val_result.json` mae 21.927526047589435 = recomputed from `val_perimage.json` exact; `result.json`/tb best 21.92840003967285 (0.0009 apart, EMA best-ckpt eval path). Parent MAE recomputed from `/home/qkun/cac_backup/N0015_val_perimage.json` = 19.3259 (booked live parent 19.3431; bars bind to the booked 19.3431).

Mechanism read R1: `w_x = −0.0398` is only recorded in `memory/hypotheses.jsonl` line 40 (ledger evidence note); no checkpoint is stored locally (`run/latest/` contains only tb), so the value is cited, not recomputed. `model.py` confirms the scalar is `nn.Parameter(zeros(()))` at init (line 359) and the fusion site is `model.py:248`.

## 2. Slice deltas (recomputed from both val_perimage dumps)

Definitions: sparse gt<50, mid 50≤gt≤500, dense gt>500 (matches `val_result.json` slice convention). Common ids: 1286 = 1286, `ids` and `gt` lists identical on all pairs — **n matches**.

| Slice | n | Parent MAE | Child MAE | Signed delta (child−parent) |
|---|---|---|---|---|
| dense gt>500 | 17 | 330.8131 | 454.4514 | **+123.6383** |
| mid 50–500 | 374 | 37.5029 | 40.8822 | **+3.3793** |
| sparse gt<50 | 895 | 5.8136 | 5.7913 | **−0.0223** |

Whole-val: n=1286, parent 19.3259, child 21.9275, delta **+2.6017**. Child slices exactly match `val_result.json` (sparse 5.79127146651625 / mid 40.88218260576381 / dense 454.4514258889591). Ledger "mid 40.9 vs 37.5" confirmed. Images improved 626 / worsened 660 / equal 0.

## 3. Trajectory (tb ep1..halt)

Source: `run/latest/tb/t/events.out.tfevents.1789997138.xxfposfbqysxuexl-snow-694b9f5f8c-6qwhn.30286.0`; tags `val/mae`, `val/rmse`, `train/loss_epoch`. 228 steps/epoch (first val at step 227); epoch = step//228+1. All 24 logged epochs (halt at 24):

| ep | val/mae | val/rmse | train/loss_epoch |
|---|---|---|---|
| 1 | 158.6372 | 180.402 | 32.1457 |
| 2 | 86.3657 | 127.319 | 9.0630 |
| 3 | 68.9574 | 115.078 | 8.0772 |
| 4 | 57.4553 | 106.986 | 7.5032 |
| 5 | 46.0363 | 100.382 | 7.0432 |
| 6 | 38.6764 | 96.262 | 6.9956 |
| 7 | 33.1632 | 94.142 | 6.5380 |
| 8 | 29.4660 | 92.163 | 6.5161 |
| 9 | 27.0828 | 90.780 | 6.1951 |
| 10 | 25.6136 | 89.795 | 6.0484 |
| 11 | 24.7849 | 89.083 | 5.8781 |
| 12 | 24.2458 | 88.365 | 5.5611 |
| 13 | 23.8438 | 87.711 | 5.6020 |
| 14 | 23.7229 | 87.423 | 5.3882 |
| 15 | 23.4811 | 86.254 | 5.4209 |
| 16 | 23.2879 | 85.532 | 5.1043 |
| 17 | 23.0154 | 84.287 | 5.0263 |
| 18 | 22.8548 | 83.717 | 4.9236 |
| 19 | 22.7380 | 83.128 | 4.5326 |
| 20 | 22.5003 | 82.148 | 4.5470 |
| 21 | 22.3803 | 81.696 | 4.3186 |
| 22 | 22.1673 | 81.177 | 4.1826 |
| 23 | 21.9796 | 80.763 | 4.2372 |
| 24 | **21.9284** (best, HALT) | 80.797 | 4.0232 |

Monotone decreasing through ep24; best = ep24 21.9284 = `result.json` best_mae/best_epoch. Late slope ep16→24 = −0.170/ep, ep21→24 = −0.151/ep — improving but ~2.2–2.4× short of the 0.367/ep needed to reach 19.0431 by ep32 (gap at halt: +2.8853). `train/mae` tag exists with 24 points, all NaN (not usable).

## 4. R2 tail read (the 8 pre-registered ids; idea.md names no wipe set)

idea.md §1/§4 names 8 catastrophic-tail ids (935, 865, 5059, 3481, 3482, 2850, 3483, 7656) but **no wipe set** (grep "wipe" in idea.md = 0) — the wipe bucket is therefore omitted from §5 below; these 8 are reported as the pre-registered R2 transfer read instead.

| id | gt | parent pred | child pred | parent AE | child AE | ΔAE |
|---|---|---|---|---|---|---|
| 935 | 2092 | 842.8 | 263.7 | 1249.2 | 1828.3 | **+579.1** |
| 865 | 1022 | 266.1 | 346.9 | 755.9 | 675.1 | −80.8 |
| 5059 | 286 | 28.8 | 35.0 | 257.2 | 251.0 | −6.2 |
| 3481 | 431 | 21.9 | 24.3 | 409.1 | 406.7 | −2.4 |
| 3482 | 315 | 28.1 | 31.1 | 286.9 | 283.9 | −3.0 |
| 2850 | 287 | 77.6 | 89.8 | 209.4 | 197.2 | −12.2 |
| 3483 | 261 | 40.7 | 37.6 | 220.3 | 223.4 | +3.1 |
| 7656 | 1231 | 380.6 | 368.0 | 850.4 | 863.0 | +12.6 |

ΣAE parent 4238.5 → child 4728.7 (**+490.2**): P2 transfer failed; 935 alone regresses +579.1 (pred 842.8 → 263.7 against gt 2092). 5/8 slightly improved (−2 to −81), 3/8 worse.

## 5. AE decomposition (common ids, buckets)

Common ids: 1286 = 1286 (child-only 0, parent-only 0; gt identical on all 1286 pairs). **n matches.** Bucket assigned from gt: gt>500 / 250–500 / 50–250 residual / sparse gt<50 (no wipe bucket — see §4).

| Bucket | n | Parent ΣAE | Child ΣAE | ΔAE | Share of ΔAE | Parent MAE | Child MAE | ΔMAE | Δcontribution to overall |
|---|---|---|---|---|---|---|---|---|---|
| gt>500 | 17 | 5623.82 | 7725.67 | **+2101.85** | **+62.8%** | 330.8131 | 454.4514 | +123.6383 | +1.6344 |
| 250–500 | 37 | 4899.17 | 5811.50 | **+912.34** | **+27.3%** | 132.4099 | 157.0676 | +24.6577 | +0.7094 |
| 50–250 (residual) | 337 | 9126.90 | 9478.43 | +351.53 | +10.5% | 27.0828 | 28.1259 | +1.0431 | +0.2734 |
| sparse gt<50 | 895 | 5203.17 | 5183.19 | −19.98 | −0.6% | 5.8136 | 5.7913 | −0.0223 | −0.0155 |
| **total** | **1286** | **24853.06** | **28198.80** | **+3345.74** | **100%** | **19.3259** | **21.9275** | **+2.6017** | **+2.6017** |

Dense gt>500 (62.8%) + 250–500 (27.3%) carry 90.1% of the AE regression; sparse is the only improving bucket (−19.98). Worst dense pair: 3425.jpg (gt 885, pred 795.1 → 132.4, ΔAE +662.7) followed by 935.jpg (+579.1).

## 6. Source citation per table

| Table | Source file(s) |
|---|---|
| 1. Verdict bars (child) | `NODE/result.json`, `NODE/val_result.json` |
| 1. Verdict bars (parent/thresholds) | `NODE/idea.md` line 3 + §4 (booked bars bound to live 19.3431 / 330.81 / 5.45); parent MAE cross-check `/home/qkun/cac_backup/N0015_val_perimage.json` |
| 1. Futility gate line | `NODE/result.json` (futility_hit, best 21.9284@24); need 0.367/ep from `memory/hypotheses.jsonl` line 40 (ledger evidence note) |
| 1. R1 `w_x=−0.0398` | `memory/hypotheses.jsonl` line 40 (ledger only; no local checkpoint — `run/latest/` holds tb only); zero-init confirmed `NODE/model.py:359`, fusion site `NODE/model.py:248` |
| 2. Slice deltas | `NODE/val_perimage.json` (child) + `/home/qkun/cac_backup/N0015_val_perimage.json` (parent), recomputed both sides; child slices also in `NODE/val_result.json` (exact match) |
| 3. Trajectory | `NODE/run/latest/tb/t/events.out.tfevents.1789997138.xxfposfbqysxuexl-snow-694b9f5f8c-6qwhn.30286.0` (tags `val/mae`, `val/rmse`, `train/loss_epoch`); endpoint `NODE/result.json` |
| 4. R2 tail per-image | `NODE/val_perimage.json` + `/home/qkun/cac_backup/N0015_val_perimage.json`; id list from `NODE/idea.md` §1 table + §4 P2/R2 |
| 5. AE decomposition | `NODE/val_perimage.json` + `/home/qkun/cac_backup/N0015_val_perimage.json` (paired on 1286 common ids, computed in this analysis) |

All numbers above were computed locally from the cited files; no remote access used.

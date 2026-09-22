# Quantitative feedback — N0023_h0022 (H0022 `use_margin`)

Child endpoint: `result.json` best_mae 22.1632 @ep24, futility_hit true, 24/32 epochs, budget_hit false.
Parent: N0015_h0014, seed 20260830, v2 protocol, EMA val MAE 19.3431 @ep32.

## 1. Verdict table — every booked bar

| Bar (booked) | Threshold | Child value | Parent value | Pass/Fail |
|---|---|---|---|---|
| val EMA MAE | ≤19.0431 | 22.1632 | 19.3431 | **FAIL** (+2.8201 vs parent, +3.1201 vs bar) |
| dense gt>500 MAE | <330.81 | 458.6918 | 330.8131 | **FAIL** (+127.8787) |
| sparse gt<50 MAE | ≤5.45 | 5.7026 | 5.8136 | **FAIL** (+0.2526 vs bar; −0.1110 vs parent) |
| R2 wipe rescue | ≥7 of 11 at pred/gt ≥0.65 | 0 of 11 | 0 of 11 | **FAIL** (0/11) |

Triple conjunctive gate: 0/4 bars passed. Run is futility-halted at ep24 (gate lines: ep16 WARN best 23.3236 need 0.2675/ep; ep24 HALT best 22.2686 need 0.4032/ep vs threshold 0.12).

Child val MAE cross-checks: `val_result.json` mae 22.163878457913317 (recomputed from `val_perimage.json` = 22.163878457913317, exact match); `result.json` best_mae 22.16316032409668 (EMA best-ckpt eval, 0.0007 apart). Parent MAE recomputed from `/home/qkun/cac_backup/N0015_val_perimage.json` = 19.3259 (booked live parent 19.3431; bars bind to the booked 19.3431).

## 2. Slice deltas (dense / mid / sparse)

Definitions: sparse gt<50, mid 50≤gt≤500, dense gt>500 (matches `val_result.json` slice convention; n per slice identical parent/child).

| Slice | n | Parent MAE | Child MAE | Signed delta (child−parent) |
|---|---|---|---|---|
| dense gt>500 | 17 | 330.8131 | 458.6918 | **+127.8787** |
| mid 50–500 | 374 | 37.5029 | 41.7143 | **+4.2114** |
| sparse gt<50 | 895 | 5.8136 | 5.7026 | **−0.1110** |

Whole-val: n=1286, parent 19.3259, child 22.1639, delta **+2.8380**.

## 3. Trajectory (tb readable — yes)

Source: `run/latest/tb/t/events.out.tfevents.1790049213.mrrxrmgwqeidrqee-snow-85d459784-pr9sl.5185.0`, tag `val/mae`. 228 steps/epoch; epoch = step//228+1. All 24 logged epochs:

| ep | val/mae | ep | val/mae |
|---|---|---|---|
| 1 | 146.0927 | 13 | 23.7119 |
| 2 | 72.7012 | 14 | 23.4832 |
| 3 | 53.1371 | 15 | 23.3236 |
| 4 | 43.6287 | 16 | 23.2461 |
| 5 | 37.4534 | 17 | 23.1767 |
| 6 | 33.4225 | 18 | 23.1015 |
| 7 | 30.3115 | 19 | 22.9844 |
| 8 | 27.9965 | 20 | 22.8132 |
| 9 | 26.5393 | 21 | 22.6339 |
| 10 | 25.4323 | 22 | 22.4277 |
| 11 | 24.6355 | 23 | 22.2686 |
| 12 | 24.1748 | 24 | 22.1632 |

Monotone decreasing through ep24; best = ep24 22.1632 = `result.json` best_mae/best_epoch.

Gate-line cross-check: quoted values 23.3236 and 22.2686 both appear in this curve (steps 3419 and 5243 → ep15/ep23 under the mapping above; gate labels quoted as ep16/ep24 as given).

`train/mae` tag exists with 24 points — **all values are NaN** (not usable). `train/loss_epoch` points exist (16.8072 @ep1 → 4.0367 @ep24) but were not requested as a bar.

## 4. R2 per-image table (wipe set)

Wiped set = 11 booked ids with parent pred/gt <0.5. Ratios recomputed from both per-image dumps (agreement with card-quoted ratios to ±0.01).

| id | gt | parent ratio | child ratio | rescued (≥0.65)? |
|---|---|---|---|---|
| 935 | 2092 | 0.4029 | 0.1484 | no |
| 7656 | 1231 | 0.3092 | 0.2977 | no |
| 865 | 1022 | 0.2603 | 0.2779 | no |
| 3481 | 431 | 0.0508 | 0.0473 | no |
| 3482 | 315 | 0.0891 | 0.0715 | no |
| 840 | 637 | 0.2811 | 0.2851 | no |
| 3484 | 356 | 0.0755 | 0.0630 | no |
| 851 | 320 | 0.4532 | 0.4275 | no |
| 3428 | 458 | 0.2267 | 0.0914 | no |
| 3487 | 321 | 0.3098 | 0.2240 | no |
| 3488 | 431 | 0.1918 | 0.0921 | no |

**Rescued: 0/11 (needs ≥7).** Mean ratio: parent 0.2410 → child 0.1842 (worsened, not ≥0.60). 6/11 ratios decreased; none reached 0.65.

## 5. AE decomposition (common ids)

Common ids: 1286 = 1286 = 1286 (child-only 0, parent-only 0; gt identical on all 1286 pairs, 0 mismatches). **n matches.**

| Quantity | Parent | Child | Delta |
|---|---|---|---|
| Σ abs error (total AE) | 24853.06 | 28502.75 | **+3649.68** |
| MAE (AE/1286) | 19.3259 | 22.1639 | +2.8380 |

AE delta share by bucket (bucket assigned from parent gt; shares of total delta +3649.68):

| Bucket | n | Parent AE | Child AE | ΔAE | Share of total ΔAE | Parent AE share |
|---|---|---|---|---|---|---|
| wipes (11 ids) | 11 | 5437.64 | 6115.93 | +678.29 | +18.6% | 21.9% |
| mid 250–500 non-wipe | 30 | 2775.00 | 3716.19 | +941.19 | +25.8% | 11.2% |
| other dense (gt>500 non-wipe) | 13 | 2310.34 | 3958.20 | +1647.85 | +45.2% | 9.3% |
| sparse (gt<50) | 895 | 5203.17 | 5103.82 | −99.34 | −2.7% | 20.9% |
| residual (50≤gt<250 non-wipe) | 337 | 9126.90 | 9608.60 | +481.70 | +13.2% | 36.7% |
| **total** | **1286** | **24853.06** | **28502.75** | **+3649.68** | **100%** | 100% |

Largest single contributors to the AE increase: other dense +1647.85 (45.2%) and mid 250–500 non-wipe +941.19 (25.8%); wipes +678.29 (18.6%). Sparse is the only improving bucket (−99.34, −2.7%).

## 6. Source citation per table

| Table | Source file(s) |
|---|---|
| 1. Verdict bars (child) | `NODE/result.json`, `NODE/val_result.json` |
| 1. Verdict bars (parent values/thresholds) | card `NODE/idea.md` §line 3 (booked bars bound to live parent 19.3431 / 330.81 / 5.45); parent MAE cross-check `/home/qkun/cac_backup/N0015_val_perimage.json` |
| 1. MAE cross-checks | `NODE/val_perimage.json`, `NODE/val_result.json`, `NODE/result.json`, `/home/qkun/cac_backup/N0015_val_perimage.json` |
| 1. Futility gate lines | quoted server log (given; values also present in tb curve) |
| 2. Slice deltas | `NODE/val_perimage.json` (child) + `/home/qkun/cac_backup/N0015_val_perimage.json` (parent); child slice n/MAE also in `NODE/val_result.json` (exact match) |
| 3. Trajectory | `NODE/run/latest/tb/t/events.out.tfevents.1790049213.mrrxrmgwqeidrqee-snow-85d459784-pr9sl.5185.0` (tags `val/mae`, `train/mae`); endpoint `NODE/result.json` |
| 4. R2 per-image | `NODE/val_perimage.json` + `/home/qkun/cac_backup/N0015_val_perimage.json` (both dumps; wipe-id list from `NODE/idea.md` §1/§2) |
| 5. AE decomposition | `NODE/val_perimage.json` + `/home/qkun/cac_backup/N0015_val_perimage.json` (paired on 1286 common ids, computed in this analysis) |

All numbers above were computed locally from the cited files; no remote access used.

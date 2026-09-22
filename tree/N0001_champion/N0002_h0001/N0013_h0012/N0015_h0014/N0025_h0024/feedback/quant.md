# Quantitative feedback — N0025_h0024 (H0024 `use_canchor`, CountAnchor)

**Verdict: REFUTED via futility-HALT (val bar decisive on the only available child metric).** Every number below is recomputed this session from files actually present under the node dir. Dense/slice/per-image child numbers are **blocked by missing artifacts** (listed §0) — no placeholder values are invented.

## 0. Artifact inventory (checked this session)

| File | Status |
|---|---|
| `result.json` | **present** |
| `info.json` | present (`status: "timeout"`, `best_metric` 22.22052001953125, `epochs` 24) |
| `val_result.json` | **MISSING** |
| `val_perimage.json` | **MISSING** |
| `val_attr.json` / any alpha dump | **MISSING** |
| `test_result.json` / `test_perimage.json` | **MISSING** (eval_test not run/pulled) |
| `run/latest/tb/t/events.out.tfevents.1790054936…10646.0` | **present** (only artifact under `run/latest/` besides `hparams.yaml`; no local `best.pth`, no local train log) |
| local `n0025_run.log` (journal:64) | **not present** on disk |

Child endpoint (`result.json`): `code` ok, best_mae **22.22052001953125 @ep24**, `futility_hit` **true**, `n_epochs_done` **24**/32, `budget_hit` false, elapsed_s **1333.6**, `config_sha256` `f30d9a561f34090d`, `model_sha256` `8599b71f9a5644dd`, `cfg_overrides` `{augment: true, futility_bar: 19.0431}`.

Parent N0015_h0014: seed 20260830, v2, live booked EMA val MAE **19.3431** @ep32; local dump `/home/qkun/cac_backup/N0015_val_perimage.json` recompute **19.325865507496644** (= `N0015_val_result.json` `mae` exactly), dense `mae_gt_gt_thr` **330.8131462545956** (n=17), sparse gt<50 recompute **5.813596044572372** (n=895), mid 50–500 recompute **37.502863894171895** (n=374). Booked bars bind to live 19.3431 / 330.81 / 5.45 (`idea.md:7,124`).

Config delta vs parent: filtered diff is exactly `use_canchor = true` (one added line); `use_simprior`/`use_cellcal`/`use_peakcal` remain true.

## 1. Triple-bar table — parent / child / delta

| Slice | n | Parent MAE | Child MAE | Δ (child−parent) | Booked bar | Verdict |
|---|---|---|---|---|---|---|
| val EMA MAE | 1286 | 19.3259 (dump) / 19.3431 (live) | **22.2205** (`result.json` best; tb `val/mae` @ep24 **exact match** 22.22052001953125) | **+2.8947** vs dump / **+2.8774** vs live | ≤19.0431 (−0.30 vs 19.3431) | **FAIL** (+3.1774 vs bar) |
| dense gt>500 | 17 (parent) | 330.8131 | **not in dump** — no child `val_result.json`/`val_perimage.json` | — | <330.81 | **UNMEASURABLE** (artifact missing) |
| sparse gt<50 | 895 (parent) | 5.8136 | **not in dump** | — | ≤5.45 | **UNMEASURABLE** (artifact missing) |
| mid 50–500 *(context)* | 374 (parent) | 37.5029 | **not in dump** | — | — | — |

Triple conjunctive gate (val ∧ dense ∧ sparse): **0/3 measurable-pass** — val fails outright; dense and sparse cannot be scored without the missing child dumps. Booked val alone refutes the card (AGENTS §6: missing any one bar refutes).

Cross-checks: tb `val/mae` @ep24 = 22.22052001953125 = `result.json` `best_mae` **exactly** (best = last; monotone curve). No `val_result.json`/`val_perimage.json` recompute path exists this session.

## 2. Booked mechanism bars — R1 / R2 / R3

All three need `val_perimage.json` and/or an alpha dump (`val_attr.json` equivalent). **None present.**

| Bar (booked, `idea.md:121–124`) | Threshold | Child value | Verdict |
|---|---|---|---|
| **R1a** val `alpha` coefficient of variation | ≥0.05 | **not in dump** (no alpha / per-image field anywhere under the node) | **UNMEASURABLE** |
| **R1b** median alpha on booked-11 wipes ≥1.3× median alpha on 27 KEEP (gt≥300) | ≥1.3× | **not in dump** | **UNMEASURABLE** |
| **R1c** residual L2 on wipes ≥20% above parent residual | ≥+20% | **not in dump** (no residual norms, no child per-image) | **UNMEASURABLE** |
| **R2** wipe mean pred/gt 0.241 → ≥0.40; ≥4/11 at ≥0.50 | ≥0.40 / ≥4 of 11 | **not in dump** (no child ratios) | **UNMEASURABLE** |
| **R3** sparse ≤5.45; no KEEP ratio≥0.8 falls below 0.6 | see booked | **not in dump** | **UNMEASURABLE** |
| **R4 val** | ≤19.0431 | **22.2205** | **FAIL** |
| **R4 dense** | <330.81 | **not in dump** | **UNMEASURABLE** |
| **R4 sparse** | ≤5.45 | **not in dump** | **UNMEASURABLE** |

Engagement reads (R1) are the card's mechanism falsifiers (`idea.md:121`); with alpha absent from every local artifact they cannot PASS — and under the booked DISPROVED-IF they also cannot be claimed. Verdict rests on the measurable val bar + futility line; alpha stats must be reported as **not in dump**.

## 3. Per-image AE split — blocked

AE partition (wipes / KEEP-27 / sparse / mid / dense) requires paired `val_perimage.json` child dump vs `/home/qkun/cac_backup/N0015_val_perimage.json`. Child dump **MISSING** → **no AE split this session**. Parent total AE from its dump = **24853.06**-class figure as in sibling quant files (parent MAE 19.3259 × 1286); child total AE unknown.

## 4. Trajectory, sibling crossover, futility line (exact)

Source: `run/latest/tb/t/events.out.tfevents.1790054936.mrrxrmgwqeidrqee-snow-85d459784-pr9sl.10646.0`, tag `val/mae`, 24 points, 228 steps/epoch (epoch = step//228+1; steps 227…5471). Parent tb **not local** (parent dir has no `run/`); sibling curves from `../N0023_h0022/...events.out.tfevents.1790049213…5185.0` and `../N0024_h0023/...events.out.tfevents.1790051003…7088.0`.

| ep | N0025 | N0023 | N0024 | Δ(25−23) | Δ(25−24) |
|---|---|---|---|---|---|
| 1 | 141.6922 | 146.0927 | 148.7946 | −4.4005 | −7.1024 |
| 2 | 70.4375 | 72.7012 | 71.3562 | −2.2638 | −0.9188 |
| 3 | 52.3612 | 53.1371 | 51.5911 | −0.7759 | +0.7702 |
| 4 | 43.3718 | 43.6287 | 42.1085 | −0.2569 | +1.2633 |
| 5 | 37.5553 | 37.4534 | 36.2074 | **+0.1020** | +1.3479 |
| 6 | 33.7801 | 33.4225 | 32.4409 | +0.3576 | +1.3393 |
| 7 | 31.1598 | 30.3115 | 29.9337 | +0.8484 | +1.2261 |
| 8 | 29.4210 | 27.9965 | 28.3963 | +1.4245 | +1.0247 |
| 9 | 28.0208 | 26.5393 | 27.3313 | +1.4815 | +0.6895 |
| 10 | 27.1377 | 25.4323 | 26.5895 | +1.7054 | +0.5482 |
| 11 | 26.5180 | 24.6355 | 26.0415 | +1.8826 | +0.4766 |
| 12 | 25.9915 | 24.1748 | 25.7221 | +1.8167 | +0.2694 |
| 13 | 25.4032 | 23.7119 | 25.8116 | +1.6913 | −0.4084 |
| 14 | 24.9677 | 23.4832 | 25.7801 | +1.4844 | −0.8125 |
| 15 | 24.6253 | 23.3236 | 25.6014 | +1.3017 | −0.9760 |
| 16 | 24.4384 | 23.2461 | 25.2893 | +1.1923 | −0.8509 |
| 17 | 24.2264 | 23.1767 | 24.9069 | +1.0497 | −0.6805 |
| 18 | 23.9438 | 23.1015 | 24.6170 | +0.8423 | −0.6732 |
| 19 | 23.7472 | 22.9844 | 24.3379 | +0.7628 | −0.5907 |
| 20 | 23.3338 | 22.8132 | 24.0510 | +0.5206 | −0.7173 |
| 21 | 23.1160 | 22.6339 | 23.8082 | +0.4821 | −0.6922 |
| 22 | 22.8288 | 22.4277 | 23.5111 | +0.4011 | −0.6822 |
| 23 | 22.4523 | 22.2686 | 23.2437 | +0.1838 | −0.7914 |
| 24 | **22.2205** | 22.1632 | 23.0618 | **+0.0574** | **−0.8413** |

**Crossover:** N0025 ahead of N0023 at eps 1–4 (max lead −4.4005 at ep1), first behind at **ep5** (+0.1020), behind every epoch thereafter; ends **+0.0574** vs N0023 (essentially parity with the MarginCal sibling) and **−0.8413** vs the worse N0024. Monotone decrease through ep24; best = ep24. `train/mae` tag exists (24 pts) but **all values NaN — not usable**; `train/loss_epoch` 15.9699 @ep1 → 4.0819 @ep24 (not a booked bar). `val/rmse` 166.867 → 80.454.

**Futility (exact arithmetic from tb + `result.json`; rule `src/cac/engine/runner.py:_FutilityStop`, threshold `margin×ref = 2.0×0.06 = 0.12`):**

- **ep16 WARN** (gates 16,24): best-through-ep15 **24.6253** → need `(24.6253−19.0431)/16 = 0.3489`/ep > 0.12 → **WARN (no halt before ep24)**. (Post-update convention with best-through-ep16 24.4384 → need 0.3372, also WARN.)
- **ep24 HALT:** ceiling = `19.0431 + 0.12×(32−24)` = **20.0031**. Final best **22.2205** → need `(22.2205−19.0431)/8 = 0.3972`/ep > 0.12 → **HALT**; best sits **+2.2174 above the ep24 ceiling**. Pre-update convention (best-through-ep23 22.4523 → need 0.4262) also HALT under either tracker ordering.
- `result.json` confirms: `futility_hit: true`, `n_epochs_done: 24`, `best_epoch: 24`.
- Booked gate line (`idea.md:132,139`): **ep24 HALT iff best > 20.00** → 22.2205 > 20.00. HALT band for "draw-contaminated" is (20.0, 21.6); **22.2205 is outside that band** (above 21.6) → **not draw-contaminated by the card's own arbitration rule**; R1/R2 cannot arbitrate (not in dump), so the verdict is futility + val-bar refutation as booked.
- Gap vs live parent: **+2.8774**; vs bar: **+3.1774**. Same-seed draw spread context: {19.34, 20.70, 21.59} sd≈1.13 — 22.22 is above the worst historical draw but mechanism reads are unavailable to re-weight the level.

## 5. Source citation per table

| Table | Source file(s) |
|---|---|
| 0. Artifact inventory | directory listing of `NODE/` and `NODE/run/latest/` this session |
| 1. Triple bars (child) | `NODE/result.json`; tb `val/mae` (exact cross-check); dense/sparse/mid **absent** |
| 1. Triple bars (parent) | `/home/qkun/cac_backup/N0015_val_result.json` (`mae`, `mae_gt_gt_thr`); sparse/mid recompute from `/home/qkun/cac_backup/N0015_val_perimage.json`; booked thresholds `NODE/idea.md:7,124` |
| 2. R1–R4 mechanism rows | bars `NODE/idea.md:121–124`; child values **not in dump** |
| 3. AE split | requires child `val_perimage.json` — **MISSING**; parent dump only |
| 4. Trajectory + crossover | `NODE/run/latest/tb/t/events.out.tfevents.1790054936…10646.0`; siblings `../N0023_h0022/...5185.0`, `../N0024_h0023/...7088.0` (tag `val/mae`) |
| 4. Futility | rule `src/cac/engine/runner.py:151–191`; endpoint `NODE/result.json` (`futility_hit`, `best_mae`, `n_epochs_done`); ceiling arithmetic this analysis; booked gate `NODE/idea.md:139` |

All numbers above were computed locally from the cited files this session; no remote access used. **Verdict: REFUTED via futility-HALT** (val bar FAIL +2.8774/+3.1774; dense/sparse/AE/R1–R3 blocked on missing `val_result.json`, `val_perimage.json`, and any alpha dump).

## 6. Slices now available

Addendum (appended after the body above was written; inventory in §0 superseded on these rows only): `val_result.json` and `test_result.json` are now present locally under the node.

- **val** (`val_result.json`, n=1286, ckpt_epoch 24): mae **22.2127**, rmse 80.45, bias −15.57.
- **Slices:** dense gt>500 **454.05** (n=17, +123.2 vs parent 330.81) · mid 50–500 **41.91** (n=374, +4.41 vs 37.50) · sparse gt<50 **5.779** (n=895, −0.035 vs 5.814, still over bar 5.45).
- **test** (`test_result.json`, n=1190): mae **19.437** (ckpt_epoch 24).
- **Triple-bar gate: all three bars FAILED** — val 22.2127 > 19.0431 · dense 454.05 > 330.81 · sparse 5.779 > 5.45. Verdict unchanged: REFUTED via futility-HALT.
- Still missing: `val_perimage.json`, any alpha/attr dump → §2 R1/R2/R3 and §3 AE split remain unmeasurable (stated honestly; nothing inferred).

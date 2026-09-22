# Quantitative feedback — N0024_h0023 (H0023 `use_antimatch`)

**Verdict: REFUTED (as booked; unchanged).** Every number below is recomputed this
session from local files now present in the node dir (`result.json`,
`val_result.json`, `val_perimage.json`, `val_attr.json`,
`run/latest/tb/t/events.out.tfevents.1790051003…7088.0`) plus the parent dump
`/home/qkun/cac_backup/N0015_val_perimage.json` (+ `N0015_val_result.json`,
parent attribution `/tmp/opencode/val_attr.json`). **No placeholder holes remain —
every table cell below is file-derived.**

Child endpoint (`result.json`): best_mae **23.06184959411621 @ep24**,
`futility_hit=true`, `n_epochs_done=24`/32, `budget_hit=false`, elapsed_s 1314.8,
`cfg_overrides {augment: true, futility_bar: 19.0431}`.
Parent N0015_h0014: seed 20260830, v2 protocol, live booked EMA val MAE **19.3431**
@ep32; local parent dump recompute 19.325865507496644 = `N0015_val_result.json`
`mae` exactly. Booked bars bind to the live 19.3431 / 330.81 / 5.45
(`idea.md:9`); paired per-image math uses the dumps (n=1286, ids and gt identical
across parent/child, 0 mismatches).

## 1. Triple-bar table — parent / child / delta (`val_result.json` + parent dumps)

| Slice | n | Parent MAE | Child MAE | Δ (child−parent) | Booked bar | Verdict |
|---|---|---|---|---|---|---|
| val EMA MAE | 1286 | 19.3259 (dump) / 19.3431 (live) | **23.0572** (`val_result.json` `mae`); best **23.0618** (`result.json`) | **+3.7314** vs dump / **+3.7187** vs live (best basis) = **+19.2%** | ≤19.0431 (−0.30 vs 19.3431) | **FAIL** (+4.0187 vs bar) |
| dense gt>500 | 17 | 330.8131 (`mae_gt_gt_thr`) | **491.8111** (`mae_gt_gt_thr`) | **+160.9980 (+48.7%)** | <330.81 | **FAIL** |
| sparse gt<50 | 895 | 5.8136 (recompute; no `slices` in parent dump) | **5.9717** (`slices.sparse.mae`) | **+0.1581** | ≤5.45 | **FAIL** (+0.5217 vs bar; bar was a standing parent miss: parent 5.8136 already +0.3636 over) |
| mid 50–500 *(context, not a booked bar)* | 374 | 37.5029 (recompute) | **42.6369** (`slices.mid.mae`) | **+5.1340 (+13.7%)** | — | — |

Triple conjunctive gate (val ∧ dense ∧ sparse): **0/3 pass**.

Cross-checks: child `val_perimage.json` recompute = 23.057240564011117 =
`val_result.json` `mae` (exact); tb `val/mae` @ep24 = 23.06184959411621 =
`result.json` `best_mae` (exact). The 0.0046 gap between the two evals is the
same best-ckpt-vs-val_result distinction seen in sibling N0023 (0.0007).
Parent sparse/mid recomputed from `N0015_val_perimage.json` (its `val_result.json`
carries no `slices` block; dense and `mae` come from that file directly).

## 2. Booked mechanism bars — R1 / R2 / R3

| Bar (booked) | Threshold | Child value | Verdict |
|---|---|---|---|
| **R1a** routed post-swap `evg_sum` non-negative on ≥90% of routed (`draft:118`) | ≥90% | **531/531 = 100%** (routed `evg_sum` min +18796.83, mean +18804.23) | **PASS** |
| **R1b** route fires on ≥90% of the booked 11 wipes (`draft:118`) | ≥90% | **7/11 = 63.6%** (unrouted 840, 935, 3482, 3487 — all `top1_mean>0`) | **FAIL** |
| **R2 wipe rescue** — `idea.md` DISPROVED IF | ≥7 of 11 at pred/gt ≥0.65 | **0/11** | **FAIL** |
| **R2 wipe read** — dispatch-quoted | mean pred/gt rises from 0.241; **≥4 of 11 at ≥0.50** | mean **0.20021** (from **0.24095**, i.e. **fell** −0.0407); **0/11 ≥0.50** (max child ratio 0.4780) | **FAIL** (0 vs ≥4; mean fell, did not rise) |
| **R2 wipe read** — full card (`draft:119`) | mean rises 0.241 → ≥0.55 | **0.20021** | **FAIL** |
| **R3** no new wipes + dense degradation ≤3% (`draft:35`) | 0 new wipes; dense within 3% | **19 child wipes vs parent 11 → 8 new**; dense **+48.7%** | **FAIL** (both) |

Overall incl. R1/R2/R3: **1 of 8 bars pass (R1a only)**. Triple bars alone 0/3.
The decisive signature: the swap fired construction-true (R1a) and every count
bar still failed — verdict **REFUTED as booked**.

## 3. Per-image AE split — paired parent vs child (real AE from both dumps)

Disjoint, exhaustive partition of all 1286 ids (buckets from parent gt = child gt
everywhere): wipes (booked 11) ∪ KEEP (gt≥300, parent ratio ≥0.5, n=27) = all 38
gt≥300 images; sparse gt<50; mid residual 50≤gt<300. AE = |pred−gt| per dump.

| Bucket | n | Parent AE | Child AE | ΔAE | Share of total ΔAE | Parent AE share |
|---|---|---|---|---|---|---|
| wipes (booked 11 ids) | 11 | 5437.64 | 6042.93 | **+605.29** | **+12.6%** | 21.9% |
| KEEP gt≥300 (parent ratio ≥0.5) | 27 | 3328.81 | 6604.23 | **+3275.42** | **+68.3%** | 13.4% |
| sparse gt<50 | 895 | 5203.17 | 5344.63 | **+141.46** | **+2.9%** | 20.9% |
| mid residual 50≤gt<300 | 353 | 10883.44 | 11659.82 | **+776.38** | **+16.2%** | 43.8% |
| **total** | **1286** | **24853.06** | **29651.61** | **+4798.55** | **100%** | 100% |

Whole-val on this pairing: 19.3259 → 23.0572, Δ **+3.7314** (= 4798.55/1286).
Slice cross-check: sparse + mid(50–500) + dense ΔAE = 141.46 + 1920.12 +
2736.97 = **4798.55** (exact); the val_result mid slice (n=374) = mid-residual
353 + the 21 wipe/keep images with 300≤gt≤500.

Findings: the **KEEP-27 block carries 68.3% of all damage and 0/27 images
improved**; the 8 new wipes (ids §6) sit inside KEEP-27 and alone contribute
**+1674.81 ΔAE (34.9% of total ΔAE, 51.1% of KEEP-27 ΔAE)**; the other 19 keeps
add +1600.61. Wipes only +12.6% — the rescue never happened but was not the
main damage; sparse +2.9% is the smallest share yet still regressed in absolute
MAE (unlike sibling N0023, whose sparse improved −0.1110).

## 4. Trajectory, crossover vs N0023, futility line

Source: `run/latest/tb/t/events.out.tfevents.1790051003.mrrxrmgwqeidrqee-snow-85d459784-pr9sl.7088.0`,
tag `val/mae`, 24 points, 228 steps/epoch (epoch = step//228+1; steps 227…5471).
Sibling curve from `../N0023_h0022/run/latest/tb/t/events.out.tfevents.1790049213…5185.0`.

| ep | N0024 | N0023 | Δ(24−23) | ep | N0024 | N0023 | Δ(24−23) |
|---|---|---|---|---|---|---|---|
| 1 | 148.7946 | 146.0927 | +2.7019 | 13 | 25.8116 | 23.7119 | +2.0997 |
| 2 | 71.3562 | 72.7012 | −1.3450 | 14 | 25.7801 | 23.4832 | +2.2969 |
| 3 | 51.5911 | 53.1371 | −1.5460 | 15 | 25.6014 | 23.3236 | +2.2777 |
| 4 | 42.1085 | 43.6287 | −1.5202 | 16 | 25.2893 | 23.2461 | +2.0433 |
| 5 | 36.2074 | 37.4534 | −1.2459 | 17 | 24.9069 | 23.1767 | +1.7302 |
| 6 | 32.4409 | 33.4225 | −0.9816 | 18 | 24.6170 | 23.1015 | +1.5155 |
| 7 | 29.9337 | 30.3115 | −0.3777 | 19 | 24.3379 | 22.9844 | +1.3535 |
| 8 | 28.3963 | 27.9965 | **+0.3998** | 20 | 24.0510 | 22.8132 | +1.2379 |
| 9 | 27.3313 | 26.5393 | +0.7920 | 21 | 23.8082 | 22.6339 | +1.1743 |
| 10 | 26.5895 | 25.4323 | +1.1572 | 22 | 23.5111 | 22.4277 | +1.0833 |
| 11 | 26.0415 | 24.6355 | +1.4060 | 23 | 23.2437 | 22.2686 | +0.9751 |
| 12 | 25.7221 | 24.1748 | +1.5473 | 24 | **23.0618** | 22.1632 | **+0.8987** |

**Crossover:** N0024 ahead of N0023 at eps 2–7 (max lead −1.5460 at ep3), first
behind at **ep8** (+0.3998) and behind every epoch thereafter; gap +0.8987 at
ep24. Both runs futility-halted at ep24. `train/mae` tag exists (24 points) but
**all values are NaN — not usable**; `train/loss_epoch` exists (36.6454 @ep1 →
4.0871 @ep24) but was not a booked bar.

**Futility:** rule = `need = (best − bar)/(32 − ep)` vs `margin × ref_slope` =
2.0 × 0.06 = **0.12** (`src/cac/engine/runner.py:_FutilityStop`; threshold
confirmed `journal/events.jsonl:33`). **HALT at ep24: final best 23.0618 vs the
ep24 ceiling 20.0031 = 19.0431 + 0.12×(32−24) → +3.0587 above ceiling.**
Booked gate line (`memory/hypotheses.jsonl:44`, `diagnostic.md:9`): **ep24
need 0.5251/ep > 0.12 → HALT**, gate-line best 23.2437 (pre-update tracker:
(23.2437−19.0431)/8 = 0.5251 exactly; with final best 23.0618 need is still
0.5023 > 0.12 — HALT under either convention). ep16 WARN line recomputed from
the tb curve under the same convention: best-through-ep15 25.6014 → need
0.4099/ep > 0.12 → WARN (no halt before ep24).

## 5. R2 per-image table (booked 11 wipes, paired)

Ratios = pred/gt (`val_perimage.json` parent cols; child `val_attr.json` `ratio`
field — equal to child pred/gt exactly, max deviation 0.0). AE = |pred−gt|
paired. `routed` ⇔ `top1_mean<0` holds on all 1286 rows (0 mismatches).
Ordered by gt desc.

| id | gt | parent AE | child AE | ΔAE | parent ratio | child ratio | Δratio | routed | top1_mean | ev_sum_pswap | evg_sum | ≥0.50? | ≥0.65? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 935 | 2092 | 1249.21 | 1800.94 | +551.73 | 0.4029 | 0.1391 | −0.2637 | 0 | +0.1032 | +1719.3 | +1800.6 | no | no |
| 7656 | 1231 | 850.37 | 877.56 | +27.19 | 0.3092 | 0.2871 | −0.0221 | 1 | −0.6855 | −12843.3 | +18839.2 | no | no |
| 865 | 1022 | 755.94 | 697.40 | −58.54 | 0.2603 | 0.3176 | +0.0573 | 1 | −0.2635 | −4923.7 | +18843.7 | no | no |
| 840 | 637 | 457.96 | 436.51 | −21.45 | 0.2811 | 0.3147 | +0.0337 | 0 | +0.0349 | +402.4 | +449.4 | no | no |
| 3428 | 458 | 354.15 | 410.07 | +55.91 | 0.2267 | 0.1047 | −0.1221 | 1 | −0.3434 | −6524.0 | +18812.1 | no | no |
| 3481 | 431 | 409.12 | 408.80 | −0.32 | 0.0508 | 0.0515 | +0.0007 | 1 | −0.1644 | −3112.2 | +18815.8 | no | no |
| 3488 | 431 | 348.34 | 378.18 | +29.83 | 0.1918 | 0.1226 | −0.0692 | 1 | −0.0184 | −457.8 | +18826.9 | no | no |
| 3484 | 356 | 329.12 | 337.03 | +7.91 | 0.0755 | 0.0533 | −0.0222 | 1 | −0.2436 | −4700.8 | +18808.1 | no | no |
| 3487 | 321 | 221.54 | 242.69 | +21.15 | 0.3098 | 0.2439 | −0.0659 | 0 | +0.0967 | +1626.9 | +1691.1 | no | no |
| 851 | 320 | 174.97 | 167.03 | −7.93 | 0.4532 | 0.4780 | +0.0248 | 1 | −0.3054 | −5718.3 | +18816.7 | no | no |
| 3482 | 315 | 286.92 | 286.73 | −0.20 | 0.0891 | 0.0898 | +0.0006 | 0 | +0.0600 | +833.8 | +880.4 | no | no |
| **Σ / mean** | — | **5437.64** | **6042.93** | **+605.29** | **0.24095** | **0.20021** | **−0.0407** | 7/11 routed | — | — | — | **0/11** | **0/11** |

**R2 actuals:** mean pred/gt **0.24095 → 0.20021** (booked "from 0.241" — it
**fell** by 0.0407, and is far below the card's ≥0.55); **0/11 ≥ 0.50** (booked
≥4/11; max child ratio 0.4780); **0/11 ≥ 0.65** (booked ≥7/11). Routing: 7/11
fired (unrouted 840, 935, 3482, 3487 — all `top1_mean>0`); of those 7 routed,
**4 ratios decreased** (7656, 3428, 3484, 3488) and 3 rose slightly (865, 3481,
851) — none reached 0.50 despite `evg_sum` +18.8k on every routed row. Wipe-set
AE worsened **+605.29** (12.6% of total ΔAE).

## 6. Quantitative slices from `val_attr.json` (exact fields)

**Field inventory (1286 rows, all present):** `id, gt, ratio, pred, routed,
top1_mean, ev_sum_pswap, evg_sum, neg_ev_frac, gain_on_neg, mhat_mean, w_c,
w_p, temp`. **Not in dump:** any per-image or per-epoch `w_c`/`w_p`/`temp`
series (each is a single constant per row, constant across all 1286), a
gate-fire flag other than `routed`, per-epoch dumps, and the parent-attribution-
only fields `sim_m, cons_mean, ev_sum, m_min, mhat_p99, gain_p99, cond_absmean,
cond_std, dec_std, dens_std` (those live in `/tmp/opencode/val_attr.json`, the
parent N0015 dump — which itself has **no** `routed`, `w_p`, or `ev_sum_pswap`).
No field is quoted below unless it exists.

### R1 — routed post-swap evidence

| Quantity | Value (`val_attr.json`) |
|---|---|
| routed images (`routed`==1) | **531 / 1286 = 41.29%** |
| `evg_sum` ≥ 0 among routed | **531/531 = 100%** (bar ≥90% → R1a PASS) |
| routed `evg_sum` min / max / mean / Σ | +18796.83 / +18903.85 / **+18804.23** / **+9,985,043.6** |
| routed `ev_sum_pswap` (pre-swap) | **531/531 < 0**, mean −2774.02, Σ −1,473,002.5 → swap flipped the sign as constructed |
| booked-11 routed-7 `evg_sum` | +18808.1 … +18843.7 each (Σ +131,762.5); their `ev_sum_pswap` −457.8 … −12843.3 (all <0) |
| same 7 ids at parent weights (`/tmp/opencode/val_attr.json`) | `evg_sum` Σ **−62,236.3** (mean −8890.9); all-11 parent `evg_sum` Σ −58,316.9 (mean **−5301.5** = the `idea.md:14` motivation number) |

### R3 — no-new-wipes (child ratio <0.5 at gt≥300)

| Quantity | Value |
|---|---|
| parent wipes (gt≥300 ∧ parent ratio<0.5) | **11** — exactly the booked set (verified) |
| child wipes (gt≥300 ∧ child ratio<0.5) | **19** (child keeps: 19) |
| **new wipes** | **8**: 1919, 1956, 3425, 3433, 3434, 3436, 3437, 5744 (all parent keeps) |
| new-wipe AE | parent 1416.76 → child 3091.57, **ΔAE +1674.81** (34.9% of total ΔAE) |
| dense gt>500 | 330.8131 → 491.8111 = **+48.7%** (bar: degradation ≤3%) → **FAIL** |

### Gate-fire by group (`routed` ⇔ `top1_mean<0`, exact on 1286/1286 rows)

| Group | n | routed | routed_frac | mean `ratio` | mean `ev_sum_pswap` | mean `evg_sum` | mean `gain_on_neg` |
|---|---|---|---|---|---|---|---|
| booked wipes | 11 | 7 | 0.6364 | 0.2002 | −3063.4 | +12416.7 | 1.0172 |
| KEEP gt≥300 (parent keeps) | 27 | 18 | 0.6667 | 0.5928 | −2195.2 | +13523.4 | 1.0144 |
| mid residual 50≤gt<300 | 353 | 194 | 0.5496 | 0.8326 | −454.0 | +11704.4 | 1.0177 |
| sparse gt<50 | 895 | 312 | 0.3486 | 1.0734 | +834.6 | +8228.7 | 1.0191 |
| val_result mid slice 50–500 | 374 | 209 | 0.5588 | 0.8121 | −572.5 | +11839.9 | 1.0177 |
| child-ratio KEEP (gt≥300, ratio≥0.5) | 19 | 12 | 0.6316 | 0.7044 | −1783.0 | +13002.8 | 1.0138 |
| child-ratio WIPE (gt≥300, ratio<0.5) | 19 | 13 | 0.6842 | 0.2540 | −3110.0 | +13403.3 | 1.0167 |
| **all val** | **1286** | **531** | **0.4129** | 0.9897 | +383.9 | +9329.7 | 1.0186 |

### Gain-stack scalars (present fields, exact values)

- `w_c` = **0.028223907575011253** (0.02822391), constant on all 1286 rows;
  parent dump `w_c` = 0.14764755964279175 → child actuator **5.23× weaker**.
- `w_p` = **0.006430869922041893** (0.00643087) < 0.01 ⇒ peakcal skipped at dump.
- `temp` = **0.07000000029802322** (0.07), pinned, constant on all rows.
- `gain_on_neg`: all-mean **1.01865**, routed-mean **1.01880**, min 1.00634,
  max 1.02080; wipe-group child 1.0167 / keep-group 1.0138 vs parent wipe
  **1.0979** / keep **1.0758** (`/tmp/opencode/val_attr.json`) — the excess over
  1 is ~5–6× smaller than parent (the "gain stack ran ~5× weaker" claim).
- `neg_ev_frac` all-mean **0.55303**; `mhat_mean` routed-mean **0.999997**
  (mean-1 identity of the swapped substrate, exact to dump precision).

## 7. Source citation per table

| Table | Source file(s) |
|---|---|
| 1. Triple bars (child) | `NODE/val_result.json` (`mae`, `mae_gt_gt_thr`, `slices`), `NODE/result.json` (best) |
| 1. Triple bars (parent) | `/home/qkun/cac_backup/N0015_val_result.json` (`mae`, `mae_gt_gt_thr`); sparse/mid recomputed from `/home/qkun/cac_backup/N0015_val_perimage.json` (parent dump has no `slices`); booked thresholds `NODE/idea.md:9` |
| 1. MAE cross-checks | `NODE/val_perimage.json`, `NODE/val_result.json`, `NODE/result.json`, tb `val/mae`, `N0015_val_perimage.json` |
| 2. R1a/R1b/R2/R3 bars | `NODE/val_attr.json` (`evg_sum`, `routed`, `ratio`, `top1_mean`); bars: `NODE/idea.md:9`, `local/research/h0023a_idea_DRAFT.md:118-119,35`, dispatch quote (mean-from-0.241 / ≥4-of-11-at-0.50) |
| 3. AE split | `NODE/val_perimage.json` paired with `/home/qkun/cac_backup/N0015_val_perimage.json` (computed in this analysis; wipes/keeps from parent gt & ratio; 1286 common ids, 0 gt mismatches) |
| 4. Trajectory + crossover | `NODE/run/latest/tb/t/events.out.tfevents.1790051003…7088.0` and `../N0023_h0022/run/latest/tb/t/events.out.tfevents.1790049213…5185.0` (tag `val/mae`) |
| 4. Futility | rule `src/cac/engine/runner.py:_FutilityStop`; booked gate line `memory/hypotheses.jsonl:44` + `feedback/diagnostic.md:9`; ceiling arithmetic in this analysis; threshold 0.12 = 2.0×0.06 (`journal/events.jsonl:33`) |
| 5. R2 per-image | `NODE/val_perimage.json` (parent cols) + `NODE/val_attr.json` (child cols: `ratio`, `routed`, `top1_mean`, `ev_sum_pswap`, `evg_sum`); wipe-id list `NODE/idea.md:9` |
| 6. R1/R3/gate/gain | `NODE/val_attr.json` (exact field names listed above); parent contrast `/tmp/opencode/val_attr.json` (parent N0015 attribution dump) |

All numbers above were computed locally from the cited files in this session; no
remote access used. **Verdict: REFUTED** (as booked; unchanged).

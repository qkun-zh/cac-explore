# Quantitative feedback — N0027_h0026 (H0026 `use_geme`, GemeCal geometry ME scale on SimPrior ev)

**Verdict: REFUTED via futility-HALT** — all three booked bars FAIL on pulled eval files. Every number below recomputed this session from local files under the node. **No `w_g` / per-image `f` dump exists anywhere → R0/R1/R2 mechanism reads are UNMEASURABLE**; nothing is invented.

## 0. Artifact inventory (checked this session)

| File | Status |
|---|---|
| `result.json` | **present** — best **22.087779998779297 @ep24**, `futility_hit: true`, `n_epochs_done` 24/32, `budget_hit` false, elapsed_s 1316.5, shas config `c48cbec8d36450f4` / model `f8b0470a4542fd92`, overrides `{augment: true, futility_bar: 19.0431}` |
| `val_result.json` | **present** (n=1286, ckpt_epoch 24) — mae **22.0730057964058**, dense **445.0527** (n=17), mid **41.6080** (n=374), sparse **5.8755** (n=895), bias −15.36, rmse 77.53 |
| `test_result.json` | **present** (n=1190) — mae **19.804404152942304**, dense 760.44 (n=8), sparse 8.577 (n=736) |
| `val_perimage.json` | **present** (paired-AE possible; no mechanism scalar field) |
| `w_g` value / per-image `f` histogram / corr(log me, gt) | **MISSING** — never dumped by the run → R0/R1/R2 **UNMEASURABLE** |
| `run/latest/tb` (events file) | **MISSING locally** — no `run/` dir under the node → no per-epoch series this session |
| `info.json` | present but `status: "proposed"`, `best_metric: null`, `tested_hypotheses: []` (server-run sync gap, as N0026 — do not hand-edit, rule 3) |

Config delta vs parent: filtered diff is exactly `use_geme = true` (one added line); `use_simprior/cellcal/peakcal` remain true; file `augment=false` vs launch `--set augment=true` (config-vs-runtime gap, as siblings).

Parent N0015_h0014: live booked EMA val **19.3431** @ep32 v2; dump recompute 19.3259, dense **330.8131** (n=17), sparse **5.8136** (n=895), mid **37.5029** (n=374). Bars bind to **19.0431 / 330.81 / 5.45** (`idea.md:7`).

## 1. Triple-bar table — parent / child / delta / verdict

| Slice | n | Parent MAE | Child MAE | Δ (child−parent) | Booked bar | Verdict |
|---|---|---|---|---|---|---|
| val EMA MAE @best | 1286 | 19.3431 (live) | **22.0730** (`val_result.json`) / best 22.0878 @ep24 (`result.json`) | **+2.73** | ≤19.0431 (−0.30 vs 19.3431) | **FAIL** (+3.03 vs bar) |
| dense gt>500 | 17 | 330.8131 | **445.0527** | **+114.24** | <330.81 | **FAIL** |
| sparse gt<50 | 895 | 5.8136 | **5.8755** | +0.062 | ≤5.45 | **FAIL** (standing parent-infeasible bar) |
| mid 50–500 *(context)* | 374 | 37.5029 | **41.6080** | +4.11 | — | — |
| test | 1190 | 19.066 (STATE) | **19.8044** | +0.74 | — | corroborating only |

Triple conjunctive gate: **0/3 PASS** — every bar measurable and every bar failed. Per `idea.md:7` ("missing any one bar refutes"), the card is **REFUTED as booked**.

## 2. Booked mechanism bars — R0 / R1 / R2 / R3 (`idea.md:156–159`)

All engagement reads need a `w_g`/`f` dump and/or child per-image mechanism fields. **None present.**

| Bar (booked) | Threshold | Child value | Verdict |
|---|---|---|---|
| **R0** corr(log me, log(1+gt)) on val | ≥0.20 | **not in dump** (no me/f field in `val_perimage.json`) | **UNMEASURABLE** |
| **R1** CV of per-image `f` on val (+ `w_g` value, clamp histogram) | ≥0.05 | **not in dump** | **UNMEASURABLE** |
| **R2** mean f\|gt≥300 ≤0.90× mean f\|gt<50 (attenuate-on-dense loading) | ≤0.90× | **not in dump** | **UNMEASURABLE** |
| **R3 val** | ≤19.0431 | **22.0730** | **FAIL** (+3.030) |
| **R3 dense** | <330.81 | **445.05** | **FAIL** (+114.24) |
| **R3 sparse** | ≤5.45 | **5.8755** | **FAIL** (+0.426) |
| Futility (rule 16) | ep16 WARN / ep24 need ≤0.12/ep | ep16 WARN @**24.0491**; ep24 need **0.3975**/ep | **HALT CONFIRMED** (`result.json`) |

R0/R1/R2 are the card's mechanism falsifiers (F1 quiet-null / F2 dead-covariate / F3 wrong-direction detectors). With `w_g`/`f` absent from every local artifact they can neither PASS nor be claimed — the engagement disjuncts in DISPROVED-IF are **undecidable this session**; verdict rests on R3 (0/3) + futility alone.

## 3. Trajectory, corridor, futility (exact)

**Trajectory:** no local tb → no per-epoch `val/mae` series, no sibling crossover table. Only the Lead-logged ep16 WARN point (**24.0491**, need 0.313/ep) and the ep24 endpoint are available.

**Futility (rule `src/cac/engine/runner.py:151–191`; threshold = 2.0×0.06 = 0.12):**
- **ep16 WARN:** best-through-ep16 **24.0491** → need (24.0491−19.0431)/16 = **0.3129/ep > 0.12** → WARN (no halt before ep24).
- **ep24 HALT:** ceiling = 19.0431 + 0.12×8 = **20.0031**. Required slope **0.3975/ep ≫ 0.12** → **HALT**; best 22.0878 sits **+2.085 above the ceiling**. (Post-update convention from final best gives 0.3806/ep — HALT under either tracker ordering.)
- Card's own arbitration band (`idea.md:176`): HALT in (20.0, 21.6) is draw-contaminated — **22.0878 is above 21.6 by +0.49** → **outside the band by the card's own rule** → not draw-contaminated; R0/R1/R2 tiebreakers unavailable (§2), so verdict = futility + triple-bar as booked.
- Draw context (`perimage_diagnostic.md:229`): same-seed draws {19.3431, 20.697, 21.5928}, mean 20.544, sd≈1.13. Child 22.073 is **+0.48 above the worst draw**, ≈1.35σ above draw-mean — the **smallest gap of the corridor** but still outside every observed draw.
- `result.json` confirms: `futility_hit: true`, `n_epochs_done: 24`, `best_epoch: 24`.

**Sibling corridor (all four futility-HALT ep24, bar 19.0431, live parent 19.3431):**

| node | hyp | best/val | Δ vs live | dense Δ | verdict |
|---|---|---|---|---|---|
| N0023_h0022 | H0022 margin | 22.1632 | **+2.82** | +127.9 | REFUTED |
| N0024_h0023 | H0023 antimatch | 23.0618 | **+3.72** | +161.0 | REFUTED |
| N0025_h0024 | H0024 canchor | 22.2127 | **+2.88** | +123.2 | REFUTED |
| N0026_h0025 | H0025 capfilm | 23.232 | **+3.89** | (pending then) | REFUTED |
| **N0027_h0026** | **H0026 geme** | **22.0878** | **+2.73** | **+114.24** | **REFUTED** |

N0027 is the **best of the five-card corridor** (+2.73, dense +114 the smallest dense blowup) yet still misses every bar by mechanism-level margins — the corridor is five loci (sign / substrate / residual-scale / decoder-FiLM / similarity-amplitude), one consistent +2.7…+3.9 failure envelope.

## 4. Source citation per table

| Table | Source file(s) |
|---|---|
| 0. Inventory | directory listing of `NODE/` this session (no `run/`, no w_g/f dump) |
| 1. Triple bars (child) | `NODE/val_result.json`, `NODE/test_result.json`, `NODE/result.json` |
| 1. Triple bars (parent) | `/home/qkun/cac_backup/N0015_val_result.json` + `N0015_val_perimage.json`; bars `NODE/idea.md:7,159` |
| 2. R0–R3 rows | bars `NODE/idea.md:156–159`; child mechanism values **not in dump**; futility `result.json` + rule `runner.py:151–191` |
| 3. Corridor | sibling `result.json`/`val_result.json` under `../N002{3,4,5,6}*/`; ep16 WARN 24.0491 from Lead launch/log record; draws `local/research/perimage_diagnostic.md:229` |

*All numbers computed from the cited local files this session; no ssh, no invented per-epoch values, no w_g/CV statistics. **Verdict: REFUTED via futility-HALT** (triple bars 0/3; R0/R1/R2 UNMEASURABLE on missing dump).*

*Files written: this document only — `feedback/quant.md`. No commit, no ledger/`info.json` edits.*

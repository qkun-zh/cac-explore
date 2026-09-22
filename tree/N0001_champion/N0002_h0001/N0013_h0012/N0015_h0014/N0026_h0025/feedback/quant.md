# Quantitative feedback — N0026_h0025 (H0025 `use_capfilm`, CapFilm decoder-side FiLM)

**Verdict: REFUTED via futility-HALT** (the only measurable child bar, val level, fails by +3.89 vs live parent / +4.19 vs booked bar). Every number below was recomputed this session from files actually present on local disk. Dense/sparse/test child numbers are **UNMEASURABLE** (eval launched server-side, not pulled) — no placeholder values are invented, and no per-epoch trajectory numbers are fabricated.

## 0. Artifact inventory (checked this session, directory listing of `NODE/` and `NODE/feedback/`)

| File | Status |
|---|---|
| `result.json` | **MISSING locally** — Lead-cited endpoint (best 23.232019424438477@24, elapsed_s 1339.6) is not on disk; see §0.1 |
| `info.json` | present, but `status: "proposed"`, `best_metric: null`, `epochs: null`, `tested_hypotheses: []` — server run not synced (do not misread as "never ran": `journal/events.jsonl:68,70` record launch + verdict; do not hand-edit, rule 3) |
| `val_result.json` / `val_perimage.json` | **MISSING** (full val eval launched server-side per `journal:70`) |
| `test_result.json` / `test_perimage.json` | **MISSING** (eval_test pending server) |
| γ dump (R1 attr: histogram / per-image mean_c(γ)) | **MISSING** (never produced by the run) |
| `run/latest/tb` (events file) | **MISSING locally — no `run/` directory exists under the node; tb lives only on the server copy. No per-epoch numbers this session.** |
| `idea.md` / `model.py` / `config.toml` | present |
| `feedback/qual.md`, `feedback/diagnostic.md` | present (sibling roles this session) |
| local `n0026_run.log` | **not present** on disk (journal:68 cites `/data/repro/logs/n0026_run.log` — server-side) |

### 0.1 Child endpoint — what is actually file-backed

- **`journal/events.jsonl:70` (verdict, read this session):** "H0025 (N0026 use_capfilm CapFilm) futility-HALT **ep24: best 23.232** (need ~0.52/ep vs 0.12 -> HALT), val **+3.89 vs live 19.3431** … Full val/test eval launched; evidence note pending slice numbers."
- **`journal/events.jsonl:68` (launch, read):** green smoke, `config_sha 4b56e8e656961e0c`, `model_sha 917596dd5b2df04e`, `32ep/1800s v2 seed 20260830 augment=true futility_bar=19.0431`, log `/data/repro/logs/n0026_run.log`.
- **Not verifiable locally:** the full-precision `23.232019424438477` and `elapsed_s 1339.6` from the Lead's `result.json` claim appear in **no local file** (grep across `cac/` + `cac_backup/` this session) — `result.json` itself is missing. Tables below therefore use **23.232 @ep24** (the journal-logged value, rule 5); precision beyond 3 decimals, `futility_hit`/`n_epochs_done` booleans, and elapsed_s are carried as **Lead-provided, locally unverifiable** until `result.json` is pulled. The journal verdict independently records HALT @ep24, which fixes `n_epochs_done = 24` semantics (halt at the ep24 gate).
- Config delta vs parent: filtered diff is exactly `use_capfilm = true` (one added line, `diff` exit confirms `30d29`); `use_simprior/cellcal/peakcal` remain true. Config file says `augment = false` while launch carried `--set augment=true` (journal:68) — the usual config-vs-runtime gap siblings flagged.

Parent N0015_h0014: seed 20260830, v2, live booked EMA val MAE **19.3431** @ep32 (STATE.md:164, `idea.md:7`); local dump `/home/qkun/cac_backup/N0015_val_result.json` recompute **19.325865507496644** (= its `mae` exactly), dense `mae_gt_gt_thr` **330.8131462545956** (n=17), sparse gt<50 recompute **5.813596044572372** (n=895), mid 50–500 recompute **37.502863894171895** (n=374) from `/home/qkun/cac_backup/N0015_val_perimage.json`. Live bars bind to 19.0431 / 330.81 / 5.45 (`idea.md:7,168`).

## 1. Triple-bar table — parent / child / delta / verdict

| Slice | n | Parent MAE | Child MAE | Δ (child−parent) | Booked bar | Verdict |
|---|---|---|---|---|---|---|
| val EMA MAE @best | 1286 | 19.3431 (live) / 19.3259 (dump) | **23.232** (`journal:70` best @ep24; no local `result.json`/tb to cross-check) | **+3.8889** vs live / **+3.9061** vs dump | ≤19.0431 (−0.30 vs 19.3431) | **FAIL** (+4.1889 above bar) |
| val EMA MAE @full eval | 1286 | 19.3431 | **UNMEASURABLE** (`val_result.json` MISSING; eval running/queued on server) | — | ≤19.0431 | **UNMEASURABLE** (best@HALT already refutes) |
| dense gt>500 | 17 (parent) | 330.8131 | **UNMEASURABLE** (no child `val_result.json`) | — | <330.81 | **UNMEASURABLE** |
| sparse gt<50 | 895 (parent) | 5.8136 | **UNMEASURABLE** | — | ≤5.45 (standing parent-infeasible: parent 5.814 > 5.45) | **UNMEASURABLE** |
| mid 50–500 *(context)* | 374 (parent) | 37.5029 | **UNMEASURABLE** | — | — | — |
| test | 1190 | 19.066 (STATE.md:174 / journal:67) | **UNMEASURABLE** (`test_result.json` MISSING) | — | — | — |

Triple conjunctive gate (val ∧ dense ∧ sparse): **0/3 measurable-pass** — val fails outright by mechanism-level margin; dense/sparse cannot be scored without the child dumps. Per `idea.md:7` ("missing any one bar refutes") and the decisive val FAIL alone, the card is **REFUTED as booked**; the missing slices bound the *failure-mode* analysis (quant cannot split F2 sparse-blowup vs F3 dense-blowup), not the verdict.

## 2. Booked mechanism bars — R1 / R2 / R3 / R4 (`idea.md:165–168`)

All engagement reads need a γ dump and/or child `val_perimage.json`. **None present.**

| Bar (booked) | Threshold | Child value | Verdict |
|---|---|---|---|
| **R1a** CV of per-image `mean_c(γ)` on val | ≥0.05 | **not in dump** (no γ anywhere under the node) | **UNMEASURABLE** |
| **R1b** mean `mean_c(γ)` gt≥300 ≥1.5× mean gt<50 (capacity loading) | ≥1.5× | **not in dump** | **UNMEASURABLE** |
| **R1c** ≥25% channels positive mean γ on gt≥300 | ≥25% | **not in dump** | **UNMEASURABLE** |
| **R2** gt≥300 pred/gt ~0.685 → ≥0.75; mid50–300 0.86 → ≥0.90; wipes no >0.02 regression | see booked | **not in dump** (no child per-image) | **UNMEASURABLE** |
| **R3** cond×2 count-ratio closer to 1.0 than parent 0.824, or fine-sens ≥15.2×1.1 | see booked | **not in dump** (requires trained-best.pth probe, not run locally) | **UNMEASURABLE** |
| **R4 val** | ≤19.0431 | **23.232** | **FAIL** (+4.1889) |
| **R4 dense** | <330.81 | **not in dump** | **UNMEASURABLE** |
| **R4 sparse** | ≤5.45 | **not in dump** | **UNMEASURABLE** |
| Futility (rule 16) | ep24 need ≤0.12/ep | need **0.5236**/ep | **HALT CONFIRMED** (`journal:70`) |

R1 is the card's mechanism falsifier (F1 quiet-null detector). With γ absent from every local artifact it can neither PASS nor be claimed — the DISPROVED-IF engagement clause is **undecidable this session**; verdict rests on R4-val + futility alone (as with N0025).

## 3. Per-image AE split — blocked

AE partition (wipes / KEEP-27 / sparse / mid / dense / S-curve buckets) requires paired child `val_perimage.json` vs `/home/qkun/cac_backup/N0015_val_perimage.json`. Child dump **MISSING** → **no AE split, no bucket-ratio movement, no wipe table this session.** Parent-only anchors for context: parent gt≥300 pred/gt **0.6228** (n=38), mid50–300 **0.8225** (n=353), sparse **0.984** (n=895), bias **−12.4056** — recomputed this session from the parent dump (R2 baselines; child side unmeasurable).

## 4. Trajectory and futility arithmetic (exact)

**Trajectory:** `run/latest/tb` is **not on local disk** (no `run/` directory under the node; see §0) — the tfevents file exists only on the server copy. Therefore: **no per-epoch `val/mae` series, no ep16 WARN-point value, no sibling crossover table** are reported here. Sibling *endpoints* (from their local `result.json`, for corridor context only): N0023 **22.1632**@24, N0024 **23.0618**@24, N0025 **22.2205**@24 — all futility-HALT at ep24 with `futility_bar=19.0431`.

**Futility (rule `src/cac/engine/runner.py:151–191`, read this session; threshold = margin × ref = 2.0 × 0.06 = 0.12):**

- **ep24 ceiling** = `19.0431 + 0.12×(32−24)` = 19.0431 + 0.96 = **20.0031**.
- **Required slope at ep24** = `(23.232 − 19.0431)/8` = 4.1889/8 = **0.5236/ep ≫ 0.12** → **HALT**. (Matches `journal:70` "need ~0.52/ep vs 0.12 -> HALT".)
- **Gaps:** best sits **+3.2289 above the ep24 ceiling 20.0031**; **+3.8889 vs live parent 19.3431**; **+4.1889 vs booked bar 19.0431**.
- **Card's own arbitration band** (`idea.md:184`): HALT in (20.0, 21.6) is draw-contaminated → **23.232 is above 21.6 by +1.632**, i.e. outside the band by the card's own rule → not draw-contaminated; R1/R2 tiebreakers unavailable (§2), so verdict = futility + val-level refutation as booked.
- **Draw context** (file-backed: `local/research/perimage_diagnostic.md:229`, sibling diagnostics §1a): same-seed draws of identical N0015 config {19.3431 live, 20.697 N0021 retrain, 21.5928 exact repro}, mean 20.544, sd ≈ 1.13. 23.232 is **+1.639 above the worst draw** and ≈(23.232−20.544)/1.13 ≈ **2.4σ above the draw-mean** — the level is outside every observed draw; refutation still rests on the pre-registered bar + futility line, not on the draw comparison.
- ep16 WARN value: **not measurable** (no local tb; journal logs only the ep24 verdict).

## 5. Source citation per table

| Table | Source file(s) |
|---|---|
| 0. Artifact inventory | directory listing of `NODE/` and `NODE/feedback/` this session; grep for `23.232019424438477`/`1339.6` across `cac/`+`cac_backup/` (no hits) |
| 0.1 Child endpoint | `journal/events.jsonl:70` (verdict), `:68` (launch shas/overrides/smoke); local `result.json` MISSING |
| 1. Triple bars (child) | `journal:70` best 23.232; slices **absent** (`val_result.json`/`test_result.json` MISSING) |
| 1. Triple bars (parent) | `/home/qkun/cac_backup/N0015_val_result.json` (`mae`, `mae_gt_gt_thr`); sparse/mid recompute from `/home/qkun/cac_backup/N0015_val_perimage.json`; booked thresholds `NODE/idea.md:7,168`; parent test `STATE.md:174` |
| 2. R1–R4 rows | bars `NODE/idea.md:165–168`; child γ/per-image values **not in dump**; futility from `journal:70` + rule `runner.py:151–191` |
| 3. AE split / bucket ratios | parent-only recompute from `N0015_val_perimage.json`; child MISSING |
| 4. Trajectory | **absent locally** (no `run/` dir); sibling endpoints `../N0023_h0022/result.json`, `../N0024_h0023/result.json`, `../N0025_h0024/result.json` |
| 4. Futility | rule `src/cac/engine/runner.py:151–191`; endpoint `journal:70`; ceiling/need arithmetic this analysis; band `NODE/idea.md:184`; draws `local/research/perimage_diagnostic.md:229` |
| Config delta | `diff NODE/config.toml ../config.toml` → exactly `use_capfilm = true` |

## 6. Slices addendum (pending)

`val_result.json` / `test_result.json` are **not present** at write time (eval launched server-side, `journal:70`). When pulled, append: val mae + dense/mid/sparse slices vs the §1 parent column, and test mae vs 19.066. Nothing in §1–4 is estimated or back-filled.

*All numbers above were computed from the cited files this session; no ssh, no remote reads, no invented per-epoch values. **Verdict: REFUTED via futility-HALT** (val bar FAIL +3.8889/+4.1889; dense/sparse/test/AE/R1–R3 UNMEASURABLE on missing artifacts).*

*Files written: this document only — `feedback/quant.md`. No commit, no ledger/`info.json` edits.*

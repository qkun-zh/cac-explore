# Constraints & Goal (authoritative) — N0029_tf_champion

Updated: 2026-09-23 (user directive: rapid-iteration goal binds the **subset**; run the full TF works autonomously).  
Prior locks: 2026-09-22 (subset286 ultra-fine 89-bin rebuild).  
This file overrides looser notes elsewhere when they conflict.

## Goal

| item | value |
|---|---|
| **Primary metric (rapid iteration)** | FSC147 **subset286 val MAE** (`/data/repro/val_subset286.json`, locked) |
| **Target (success bar)** | **subset286 MAE ≤ 20.0** (or better) — **this is the goal that must be reached** |
| Champion / external claim | still **full val 1286 only** (subset MAE is never a champion row); full val runs **only on explicit user request** (iron rule 2026-09-23) |
| Secondary | full-val MAE trend (only if user requests a full-val run); multi-dim per-image stats on subset |
| Style | autonomous, fast, minimal dead ablations; GPU busy → single-ssh remote wait (IRON RULE #0) |

**Rationale (user, 2026-09-23):** subset286 ≈ full val in absolute level (Δ≈0.15 on best config) and is ~4× faster (~1.5–2 min vs ~6.5 min). Hitting **≤20 on subset286** is the working success criterion for rapid structural iteration; full val remains the only number allowed on champion/board rows.

## Hard constraints

### 1. Training-free only
- No supervised head training, no gradient updates, no fit-on-val-GT for main board.
- Val-GT bin/scale calibration = oracle diagnostic only, **never** champion row.
- Train-only post-hoc scale on frozen outputs: not main TF champion unless user re-opens (see `calibration_trainfit.md`).
- Building the eval subset with GT **once** for stratification is OK; **tuning** fs/thresholds/readouts **to val GT** is forbidden for board numbers.

### 2. Required model / backbone
- **MUST include in the final TF system:** `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` (~28M) — https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
  - **Role is free:** main backbone, aux branch, ensemble member, density refiner, prototype encoder, etc. Whether it is "the backbone" **does not matter**.
  - It must remain in the inference graph at val time (not dropped after ablation).
- **Default experimental home base:** this ConvNeXt-T as primary substrate (Wave1 baseline).
- **Also allowed (TF, total stack ≤64M, required model still present):**
  - a second ConvNeXt-T (same weights, different views/flags; ~56M pair)
  - other training-free pretrained encoders as *additional* modules
  - ConvNeXt-S (~50M) only if required T is still included and **total ≤64M** (e.g. T as aux + S, or swap with T retained in graph)
- Do not silently remove the required HF model from the system.

### 3. Model size
- **Prefer ≤ 64M parameters** for the active inference stack.
  - Single backbone or combined (e.g. two ConvNeXt-T ≈ 56M) must stay ≤ 64M.
  - Count auxiliary heads/readouts: pure post-process on density is ~0 params (OK); learned adapters count if any appear.
- ViT-B (~86M) and larger ViTs: **out of budget** unless user lifts the cap.

### 4. Hardware / data
- RTX 3060 12GB; FSC147 official splits; offline HF cache `/data/asset/hf`.

### 5. Logging (standing)
- Rolling MAE every 25 images; EXTENDED_METRICS + density bins + per-image CSV every full val (and final line on subset runs).
- Evidence via `discovery evidence`; new H-ids only; journal + `baseline_board.md` after each meaningful run.

### 5b. Documentation language (hard, user 2026-09-23)
- **All project documentation MUST be written in precise, accurate English. Chinese (or any non-English prose) is forbidden** in constraints, board, handoffs, STATE, ideas, feedback, synthesis, journal notes, and README files.
- Numerals, identifiers, paths, and quoted code/flags are exempt only when they are not prose.
- Violations are doc-drift: fix in the same session (AGENTS Hard Rule 8).
### 6. Evaluation protocol — IRON RULE (user 2026-09-23): subset only
- **IRON RULE (hard):** every experiment runs **only on subset286**. Full val 1286
  runs **only when the user explicitly requests it**. No auto-promotion, no
  near-tie promotion, no end-of-wave full-val batches — the agent never launches
  a full-val run on its own initiative.
  **Success criterion (2026-09-23):** subset286 MAE **≤ 20.0** ⇒ goal met (champion-row
  full val only if/when the user asks for it).
- **Subset (locked, rebuild #2 2026-09-22 — ultra-fine bins):** `/data/repro/val_subset286.json`
  - **n=286**, source = **val only** (1286 → 286)
  - **89 ultra-fine density bins** (step1 on 0–40, step2 on 40–80, step5 on 80–160, step10 on 160–280, step20 on 280–500, step50+ on 500+), merge only pop<2
  - **strict population-proportional** (largest remainder), floor 1 per nonempty bin, exact N=286
  - max |pick%−pop%| = **0.195pp**, mean 0.081pp; coarse 5-bucket shares match val (8.0/44.4/28.0/13.3/6.3 vs 8.1/43.9/28.5/13.9/5.7)
  - GT mean subset **64.81** vs val **63.80** (Δ+1.01)
  - **within-bin:** sort by (GT, key), systematic evenly spaced take
  - **seed = 20260922**; order = annotation order
  - meta: `/data/repro/val_subset286_meta.json`
  - supersedes `/data/repro/val_subset200.json` and broken fine-v1 (do not use for promotion)
- **Same subset file** for every comparison inside a wave; never re-sample per config.
- **Promotion rule (superseded 2026-09-23):** full val 1286 **only on explicit
  user request** (iron rule above). Subset results go to the journal only.
- **Champion / board claim:** **full val 1286 only**. Subset MAE is **never** a champion row.  
  **MAE-20 rapid goal:** binds **subset286 ≤ 20.0** (user 2026-09-23); full-val 20.0 remains the stretch/external target, not the rapid-iteration gate.
- CLI: `--subset_file /data/repro/val_subset286.json` (implemented).
- **Concordance note (coarse12-bin subset, pre-rebuild):** 4 anchors → Pearson 0.05 / Spearman 0.0, subset MAE ≈ full−1.8; ranking not trustworthy alone → always send near-ties to full. Re-measure after ultra-fine rebuild.

### 7. Iteration speed rules
- Prefer subset286 sweep for any new flag/idea (~1–2 min/config order).
- Full val only for promoted winners (~6.5 min).
- One GPU job at a time; chain multiple subset configs in one remote script; single-ssh wait loop.
- Skip-existing `exist_match_df` can false-positive when CSV lacks new columns → use `--no_skip true` for intentional re-runs.

## Working baseline (as of update)

| row | setting | split | MAE |
|---|---|---|---|
| Paper-best CountingDINO flags, ConvNeXt-T | DEI+DEI2+filter+ellipse, no thr, **fs=1.0** | full val | **27.628** |
| + `filter_thresh_scale=0.5` density readout | same + fs0.5 | full val | **26.485** ← current best local TF |
| CountingDINO paper anchor | DINOv2-L | val* | 25.48 (external) |
| Subset286 baseline (fs=1.0 density) | ConvNeXt-T | subset286 | TBD (Wave2 ref run) |

Gap to goal (2026-09-23):  
- **Rapid (primary):** subset286 best **25.818 → ≤20.0 = −5.82 MAE** (subset286; ADEI ladder floor t1.5d1m8 = 25.871, **+#26 MWEx = 25.818**; baseline 26.339; prior best 25.871 #9c).  
- Champion path (secondary): full val 26.485 → 20.0 = −6.48 MAE (only after subset gate).

## Attack stack (ordered, all TF, ≤64M, required T always present)

**Full closed-family table + open queue: `ATTEMPTS.md` (authoritative; update after every run).**

1. ~~Wave2 / fs sweep~~ locked at 0.5. ~~P2 / TTA / scale-TTA / in768 / fusion / density_warp~~ closed (see ATTEMPTS).
2. ~~ADEI constants~~ exhausted at **t1.5 d1 m8 = subset 25.871** (stacked with #26 MWEx → **25.818**).
3. ~~Gate+scalar filter (#11/#12), gated HR (#13), multi-scale exemplar (#14), feature modulation (#15), structure-routed dense dual-path (#16), TFCounter context-aware similarity (#17), CountSE multi-res CEF soft exemplar (#18), same-checkpoint dual ConvNeXt-T mean (#19 null/degenerate), cross-arch dual density mean T+ViT-S/16 (#20 34.236), multi-level feature fusion mid+final mean (#21 209.275), Otsu hard-filter form (#22 32.259), pre-norm hard-filter order (#23 28.872), per-exemplar ROI-norm before mean (#24 30.218), per-exemplar pre-mean abs hard-filter (#25 27.114)~~ **closed** (user: no more param tuning / gate+scalar; AdaCount both axes exhausted; spatial dual-path wall-null; bg-fusion mass wipe; CEF 25.921 lose outside near-tie; same-ckpt dual mean exact identity to champion; multi-level plain mean catastrophic overcount ratio_sum 3.999; Otsu undercount 200+ 243; pre-norm sparse overcount 200+ 204; per-exemplar norm sparse better but mid/dense worse 200+ 222 ratio_sum 0.650; pre-mean filter 27.114 lose, 200+ 208 worse than old wall, pre-cut near-noop).
4. **Open structural (autonomous, no permission prompts):** **#26 mass-weighted exemplar aggregation WIN (25.818, 200+ 196.47) locked as default champion component** — design NEW **#27** stacked orthogonally on MWEx. Must not be any closed family in `ATTEMPTS.md` (#1–#25); do not retune MWEx weights as free hyper-parameters.
5. Heavier TF density prior / diffusion-style only if structural stack still short of 20.

**Standing user directives (2026-09-23):** autonomous execution without asking to continue; English-only docs; durable lessons → `ATTEMPTS.md` immediately; one journal result per batch.

## Explicit non-goals (unless user changes lock)
- Chasing MAE 20 with only flag tweaks on a single T forward (insufficient).
- Exceeding 64M “just because”.
- Dropping `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` from the final TF system.
- Quietly using val GT to place champion numbers.
- Reporting subset286 MAE as the official/**champion** number (subset **is** the rapid-goal metric; it is **not** a champion row).

## Confidence (honest)
- ≤26: high (already 26.48 full).
- ≤24: medium-low with component stack + dense work.
- **=20: stretch goal** — needs dense problem largely solved and/or second-T / S under 64M; not guaranteed. Re-assess after subset Wave2 + P2 + dense first pass.

## Quick refs
- Board: `baseline_board.md`
- Wave1 code: `/data/cdino_run/src/readout.py`, flags on `convolutional_counting.py`
- Subset runner pattern: `--subset_file /data/repro/val_subset286.json --log_file results/results_wave2_sub.csv --no_skip true`

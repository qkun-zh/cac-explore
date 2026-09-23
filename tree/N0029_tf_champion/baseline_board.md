# Baseline board — training-free only (root N0029)

**Scope:** training-free / inference-side only. Supervised heads and post-hoc scale calibration on supervised outputs are NOT training-free and are excluded from champion ranking.

**Eval gate (IRON RULE, user 2026-09-23):** all experiments run on subset286 only; **full val 1286 only when the user explicitly requests it** (no auto-promotion). Fixed stratified subset `/data/repro/val_subset286.json` (n=286, **89 ultra-fine density bins**, proportional, seed=20260922). **Rapid goal (2026-09-23): subset286 MAE ≤ 20.0.** Champion numbers = full 1286 only. See `constraints_and_goal.md` §6.

**Authoritative constraints/goal:** see `constraints_and_goal.md` — **rapid goal (user 2026-09-23): subset286 MAE ≤ 20.0**; champion/board rows remain **full val 1286 only**; **must include** `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` (role free: backbone or aux); other TF modules allowed; **stack ≤64M params**.

## Local training-free results

| method | backbone | split | MAE | status |
|---|---|---|---|---|
| CountingDINO full flags (no DEI2) | DINOv3-ViT-L/16 | val | **32.531** | DONE (H0028 SUPPORTED) |
| CountingDINO paper-best + DEI2 (no thr) | DINOv3-ViT-S/16 | val | **40.354** | DONE; DEI2 hurts small ViT |
| CountingDINO paper-best + cosine (no DEI2) | DINOv3-ViT-S/16 | val | **38.808** | DONE (H0030) |
| CountingDINO paper-best (no thr) | DINOv3-ConvNeXt-T | val | **27.628** | DONE; best local TF |
| CountingDINO paper-best no-DEI2 control | DINOv3-ViT-S/16 | val | **39.696** | DONE; DEI2 cost vs 40.354 |
| CountingDINO full flags | DINOv3-ViT-B/16 | val | queued | optional after control |
| CountingDINO DEI-only (bad flags) | DINOv3-ViT-L/16 | val | 41.062 | invalid ablation, not a champion |
| CountingDINO paper-best + fs=0.5 density readout | DINOv3-ConvNeXt-T | val | **26.485** | DONE; best local TF (Wave1) |
| CountingDINO paper-best + hybrid mass_restore pw0.25 | DINOv3-ConvNeXt-T | val | 27.399 | DONE; weaker than fs0.5 |
| CountingDINO paper-best + fs=0.4 density | DINOv3-ConvNeXt-T | val | 27.184 | DONE; fs0.5 better |

## External anchors (reference only)

| method | source | split | MAE | note |
|---|---|---|---|---|
| CountingDINO (paper) | DINOv2-L | val* | 25.48 | paper; *re-verify split before equal compare |
| CountingDINO (paper) | DINOv2-L | test | 20.93 | paper |
| seed local full flags | DINOv3-ViT-S/16 | val | 39.696 | prior local baseline, superseded by 32.531 |

## Excluded (NOT training-free)

- supervised N0015 head raw / +scale calibration — external historical only; not a N0029 champion candidate.
- TFCounter / TF-CAC full val — cancelled (too slow on 12GB); subset later if needed.

## Queue (locked constraints: **subset286 ≤ 20 rapid goal**, ≤64M, required ConvNeXt-T in graph; **full val only on explicit user request**)
- Full closed list: **`ATTEMPTS.md`**.
- Best subset286: **25.818** (#26 mass-weighted exemplar + ADEI t1.5d1m8; prior best 25.871 #9c). Baseline subset 26.339. Full board best **26.485**. **200+ wall = 196.47** (#26; prior 198.68).
- **Closed:** fs fine · P2 · TTA flip · scale-TTA · in768 · resolution-fusion · density_warp · ADEI constants · gate+scalar filters (#11/#12) · gated HR (#13) · multi-scale exemplar (#14) · feature modulation (#15 27.617) · structure-routed dense dual-path (#16 26.232) · TFCounter context-aware similarity (#17 55.935) · CountSE multi-res CEF soft exemplar (#18 25.921) · same-checkpoint dual ConvNeXt-T mean (#19 25.8706 null/degenerate) · cross-arch dual density mean T+ViT-S/16 (#20 34.236) · multi-level feature fusion mid+final mean (#21 209.275) · Otsu hard-filter form (#22 32.259) · pre-norm hard-filter order (#23 28.872) · per-exemplar ROI-norm before mean (#24 30.218) · per-exemplar pre-mean abs hard-filter (#25 27.114) · hybrid readout full-val secondary.

- **Open (structural only, autonomous):** **#26 mass-weighted exemplar WIN (25.818, 200+ 196.47) is now default champion component** — design NEW **#27** stacked orthogonally on top of MWEx (#1–#25 closed; do not retune MWEx weights as free hyper-params).
## Notes (2026-09-22)
- DEI2 on vits16: 40.354 vs no-DEI2 39.696 → DEI2 costs only ~+0.66 MAE (not the main gap).
- cosine on vits16 no-DEI2: 38.808 (best ViT-S), still >> vitl16 32.531.
- Best local TF was ConvNeXt-T paper-best 27.628; superseded by **26.485** (fs=0.5) — see Wave1 notes.

## Notes (2026-09-22 Wave1)
- **filter_thresh_scale=0.5** on ConvNeXt-T paper-best: val **26.485** (from 27.628); bias −9.6; ratio_sum 0.85; dense gt≥200 bias −139 (was −212).
- Root cause of dense undercount: fixed `filter_background` threshold wiped 85–95% of mass on crowded images.
- Readout-only (hybrid/mass_restore/peaks) without fs fix: ~27.40 best — secondary.

## Notes (2026-09-23 structural session)
- Best subset286 **25.818** (#26 MWEx + ADEI t1.5d1m8; prior 25.871 #9c); **200+ wall 196.47** first beaten (prior 198.68).
- Closed this day: exemplar max, P2, TTA, scale-TTA, in768×2, resolution-fusion, density_warp, gate+scalar filters, gated HR, multi-scale exemplar (#14 28.795), feature modulation (#15 27.617; 200+ **216.78** worse than old wall 198.68), structure-routed dense dual-path (#16 26.232; 200+ **198.64** ≈ old wall null, sparse inflated), TFCounter context-aware similarity (#17 **55.935** catastrophic undercount, ratio_sum 0.138), CountSE multi-res CEF soft exemplar (#18 **25.921** lose, 200+ **198.89** worse than old wall), same-ckpt dual mean (#19 **25.8706** null), cross-arch density mean (#20 **34.236**), multi-level feature fusion (#21 **209.275** overcount ratio_sum 3.999), Otsu hard-filter form (#22 **32.259** undercount ratio_sum 0.563, 200+ **243.18** worse than old wall), per-exemplar ROI-norm (#24 **30.218** sparse better 0-10 2.51/10-30 5.66 but mid/dense worse 200+ **221.73** ratio_sum 0.650), per-exemplar pre-mean abs filter (#25 **27.114** lose, 200+ **208.18** worse than old wall, pre-cut near-noop). **Kept: #26 MWEx 25.818 WIN** (200+ 196.47).
- **User:** no more parameter tuning; structural only; **act autonomously**; English-only docs; durable facts → `ATTEMPTS.md` immediately; prefer full code refactor for clear SE structure when patching repeatedly.
- Next queue lives in `ATTEMPTS.md` §Open structural queue.

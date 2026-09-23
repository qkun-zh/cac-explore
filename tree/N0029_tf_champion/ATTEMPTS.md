# Attempt ledger (subset286) — N0029_tf_champion

**Language:** English only (constraints §5b).  
**Eval:** subset286 only unless user explicitly requests full val 1286 (iron rule 2026-09-23).  
**Best subset286:** **25.818** (attempt #26 mass-weighted exemplar, ADEI t1.5d1m8, fs=0.5, density; prior best 25.871 #9c).  
**Baseline subset286:** 26.339 (fs=0.5 density, no ADEI).  
**Best full val (board):** 26.485 (fs=0.5); ADEI t3.0 full was 26.358 (#8, recorded before subset-only iron rule).  
**Rapid goal:** subset286 ≤ 20.0.  
**200+ wall:** bin MAE **196.47** on best (#26) — first attempt to beat prior wall 198.68.

## User standing directives (memorize)

1. **No more parameter tuning** before major progress; **structural innovation** only (2026-09-23). Gate+scalar-constant combos are banned as “小改动”.
2. **subset286 only**; full val only on explicit user request. No auto-promotion, no board row from subset.
3. **Autonomous operation** — do not ask permission to continue; act, then document.
4. **Docs language:** English only for all project documentation.
5. **Journal:** exactly one result entry per batch via `scripts/journal.py`; never double-call; append-only.
6. **fs / readout flags locked** at paper-best + `filter_thresh_scale=0.5` + `count_readout=density` (do not re-tune globally).
7. Required backbone always in graph: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`; stack ≤64M.
8. Training-free only; no val-GT fitting / outcome-shopping.

## Closed / refuted families (do not retry the same mechanism)

| Family | Attempts | Verdict / numbers | Lesson |
|---|---|---|---|
| exemplar aggregation max | H | 27.597 refuted | in-box max/mean gap re-couples ROI-norm wrong way |
| P2 transductive harvested kernels | P2 R=4 h=3 | **37.117** refuted | crowd kernels fire too broadly → global overcount |
| flip TTA | tta_flip=1 | 26.358 ≈ 26.339 null | views too correlated |
| scale-TTA | 0.8,1.25 | 26.441 ≈ baseline null | grid fixed at 64; scene scale does not move filter semantics |
| input_size 768 global | fs0.5 / fs1.125 | 29.525 / 28.862 refuted | resolution axis closed (no third fs point) |
| resolution ensemble | offline α=0.5 | all ≥26.659 lose | error correlation ~0.88–0.96 |
| density_warp (AdaCount) | #10 g1 s2 | 28.880 refuted | warp hurts dense under ROI-norm pipeline |
| ADEI thresh ladder | #9a–#9f | floor **t=1.5** d1 m8 = **25.871**; t1.0 overshoot 26.851; d2 26.446; m4 25.932 | ADEI constants **exhausted** |
| gate+scalar filter soft_sum | #11 g150 fs0.25 | 27.111 refuted | soft anti-correlated with norm (r≈−0.51) |
| gate+scalar filter norm_coeff | #12 g50 fs0.25 | 25.899 near-tie **loss**; 200+ still 198.68 | fired n=32/413; did not move 200+ wall |
| gated high-res re-inference | #13 g50 in768 fs1.125 | 29.722 refuted | HR always lowers dense preds; family closed |
| multi-scale annotation-exemplar | #14 ms 0.75/1.0/1.25 max | 28.795 refuted | mid/sparse inflated more than dense gained |
| residual feature modulation (AdaCount FM) | #15 a0.9 g2.0 | **27.617 refuted** | sparse slightly better (0-10 3.20, 10-30 5.74) but 80-200 45.60 and 200+ **216.78** (wall 198.68) worse; ratio_sum 0.679; feature-modulation family CLOSED (no alpha/gamma retune) |
| structure-routed dense second stage (spatial dual-path) | #16 dss1.5 k5 | **26.232 refuted** | sparse inflated (0-10 4.92, 10-30 7.35); 200+ **198.64** ≈ wall 198.68 null; dense_frac on true 200+ images often ~0.01–0.03 so open path barely engaged; family CLOSED (no thresh/kernel retune) |
| context-aware similarity (TFCounter bg fusion) | #17 cx0.5 t1.3 | **55.935 refuted hard** | ratio_sum **0.138**, bias −55.9, 285/286 under; CTXSIM always fired (fg_ratio≪0.5) and range-aligned bsim wiped mass; family CLOSED (no fusion_ratio/T_div retune) |
| CountSE multi-res CEF soft exemplar selection | #18 cef | **25.921 refuted** | MAE +0.0503 (outside near-tie 0.05) and 200+ **198.89** worse than wall 198.68; sparse slightly better (0-10 4.07, 10-30 6.89) but mid/dense worse; CEF always fires (n_cand~15 n_keep 8–14 from 5 stages); family CLOSED (no CEF/clustering form retune) |
| same-checkpoint 2×ConvNeXt-T feature mean (#19 dual_convnext) | #19 dual | **25.8706 DEGENERATE/NULL** | third run exact identity to champion (per-image same_pred 286/286 maxdiff 0.0; weights identical max_abs_diff 0.0; mean1==mean2 all 286); first launch 31.874 and second 35.98 invalid (ADEI-deep bug, forced L2 with cosine/normalize False) journaled as repairs; formal near-tie vacuous: f(x,x) fusion form cannot add signal; family CLOSED (no same-ckpt dual mean / no fusion-form retune) |
| cross-arch dual density mean T + DINOv3-ViT-S/16 (#20 dual_density) | #20 xd | **34.236 refuted hard** | mean of T and ViT-S counts; sparse inflated (0-10 12.52, 10-30 15.88) by ViT overcount; 200+ **192.91** slightly under wall 198.68 but useless alone; shared dense undercount (200+ bias −171.6 ≈ champion −171.3); error corr T vs ViT-S high enough that mean cannot beat either on sparse; family CLOSED (no fusion-weight / aux-backbone / img_size retune) |
| multi-level feature fusion mid-stage + final mean (#21 mlvl_fuse) | #21 mlv | **209.275 refuted hard** | same-forward mid (1,384,128,128) bilinear+channel-repeat to 64-grid, plain 0.5 mean before ROI/ADEI; ratio_sum **3.999** (255/286 over); every bin inflated (0-10 170.17, 10-30 147.80, 200+ **404.03** worse than wall); family CLOSED (no stage-index / weight / align retune) |
| Otsu valley hard-filter form (#22 filter_otsu) | #22 otsu | **32.259 refuted hard** | parameter-free Otsu tau on clamp_min(0) replaces (1/area)*fs; sparse improved (0-10 2.15, 10-30 5.92 vs champ 4.08/6.89) but 30-80 23.05, 80-200 59.91, 200+ **243.18** worse than wall 198.68; ratio_sum 0.563 undercount (221/286 under); family CLOSED (no Otsu bins/threshold-variant/percentile/hysteresis form retune) |
| pre-norm hard-filter order (#23 filter_prenorm) | #23 pn | **28.872 refuted** | same abs thresh BEFORE /norm, post-cut skipped; raw scale >> thresh so almost no wipe then divide → sparse overcount (0-10 7.32, 10-30 9.64 vs champ 4.08/6.89); 200+ **204.27** still worse than wall 198.68; ratio_sum 0.890 (176/286 over); family CLOSED (no thresh-order/norm-order retune; post-norm fs=0.5 cut is load-bearing for sparse) |
| per-exemplar ROI-norm before mean (#24 roi_norm_per_exemplar) | #24 pex | **30.218 refuted** | stacked split before mean; each map channel-collapse+minmax+ellipse ROI /n_i (n_i=ROI/scaling), mean, norm_coeff=1, locked post-norm fs=0.5; sparse improved (0-10 **2.51**, 10-30 **5.66**) but mid/dense worse (30-80 23.61, 80-200 53.93, 200+ **221.73** vs wall 198.68, bias −206); ratio_sum **0.650** undercount; family CLOSED (no median/min/max-box per-exemplar norm variants, no other ROI-norm formula retune) |
| per-exemplar pre-mean abs hard-filter (#25 per_exemplar_filter) | #25 pef | **27.114 refuted** | each minmax map cut with (1/area)*fs=0.5 then mean, then champion ROI-norm+post-cut; sparse slightly worse (0-10 4.64, 10-30 7.57), mid/dense mixed (30-80 18.26 better, 80-200 40.15), 200+ **208.18** worse than prior wall 198.68; ratio_sum 0.842; thresh on [0,1] near-noop (kept_per ~85–90%); family CLOSED (no cut-placement / pre-mean filter threshold-form retune) |
| mass-weighted exemplar aggregation (#26 mass_weight_exemplar) | #26 mwex | **25.818 WIN** | weight each minmax exemplar map by in-box ROI mass then sum, then champion ROI-norm+post-cut; **MAE 25.818 < 25.871** and **200+ 196.47 < 198.68** (first wall beat); bins 0-10 4.16, 10-30 7.02, 30-80 17.52, 80-200 39.62, 200+ 196.47; ratio_sum 0.776; new best + new wall; MWEx locked as champion component (no weight-form retune as free param — next attempts may stack on top) |
| hybrid/mass_restore/peaks readout (full, no ADEI) | Wave1 | ~27.40 best | secondary vs fs0.5 density |
| offline base+ADEI blend | w=0.5 | 26.063 > 25.871 | highly correlated errors; blend dead |
| oracle per-image min of 13 configs | offline | 18.009 | diagnostic only — no TF selection signal |

**Gate+scalar family (soft/norm gates × fs) is permanently closed.**  
**ADEI mechanism constants closed** (thresh floor 1.5, depth 1, max 8).  
**Resolution axis closed** (512 locked; no gated HR).  
**Multi-scale exemplar scales closed** (no scale retune).  
**Feature-modulation family closed** (#15: alpha/gamma fixed from paper, both criteria failed).  
**Image/box density_warp closed** (#10); **feature-domain FM closed** (#15) — AdaCount both axes exhausted.  
**Structure-routed dual-path spatial join closed** (#16: dense_struct_thresh/kernel fixed; sparse inflated, 200+ wall null).  
**Context-aware similarity / TFCounter background-prototype fusion closed** (#17: fusion_ratio=0.5, T_div=1.3 paper-locked; catastrophic undercount 55.935; no λ/T retune).  

**CountSE multi-res CEF soft exemplar selection closed** (#18: multi-res candidates + spectral keep-largest + soft ROI kernels on final grid; 25.921 lose, 200+ worse; no candidate-mining/clustering form retune).

**Same-checkpoint dual ConvNeXt-T feature mean closed** (#19: two identical frozen T backbones, mean of final-grid feats before ROI/ADEI/post; null-check run reproduces champion exactly; earlier launches invalid for mechanism — ADEI-deep rebuilt from primary blocks only, then unconditional L2 while cosine/normalize both False; no same-ckpt dual and no fusion-form retune).

**Cross-arch dual density mean closed** (#20: ConvNeXt-T + DINOv3-ViT-S/16 aux img_size=256 → 64-grid, plain 0.5 mean of counts; 34.236 lose, sparse massively inflated, dense undercount shared; no weight/backbone/img retune).

**Multi-level feature fusion closed** (#21: same-forward penultimate stage DEI-merged, bilinear to 64-grid, integer channel-repeat, plain 0.5 mean with final feats before ROI/ADEI/post; 209.275 catastrophic overcount ratio_sum 3.999; no stage-index/weight/align-form retune).

**Otsu hard-filter form closed** (#22: parameter-free Otsu valley (256-bin between-class on clamp_min(0)) replaces absolute (1/area)*fs when --filter_otsu True; 32.259 lose, 200+ 243.18 worse than wall, ratio_sum 0.563 undercount; no bins/threshold-variant/percentile/hysteresis/Bradley form retune).

**Pre-norm hard-filter order closed** (#23: abs (1/area)*fs applied before /norm_coeff with post-cut skipped; 28.872 lose, sparse overcount from near-no-filter, 200+ 204.27 still worse than wall; no filter/norm order retune).

**Per-exemplar ROI-norm before mean closed** (#24: split stacked maps, per-exemplar channel-collapse+minmax+ellipse-ROI divide by n_i=ROI/scaling then mean with norm_coeff=1 and locked post-norm fs=0.5 cut; 30.218 lose, sparse bins improved but mid/dense worse and 200+ 221.73 vs wall 198.68; no median/min/max-box norm variants and no other ROI-norm formula retune).

**Per-exemplar pre-mean abs hard-filter closed** (#25: split stacked maps, channel-collapse+minmax each, cut each with (1/max_box_area)*fs=0.5 on the [0,1] map, mean, then champion global ROI-norm + locked post-mean cut; 27.114 lose vs 25.871, every bin at or worse than champion, 200+ 208.18 worse than prior wall 198.68, pre-cut near-noop on minmax scale; no other cut-placement / pre-filter threshold-form / order retune).

**Mass-weighted exemplar aggregation #26 WIN (kept):** ROI-mass weights on minmax maps before sum (not plain mean, not max); new best subset286 **25.818**, new 200+ wall **196.47**; mechanism locked as default champion component for follow-ups (do not retune weights as free hyper-parameters; only stack a new orthogonal mechanism on top).

## Open structural queue (next experiments, not yet run)

1. ~~Context-aware similarity (TFCounter background prototype)~~ **closed** (#17 55.935).
2. ~~Soft exemplar selection (CountSE CEF)~~ **closed** (#18 25.921; no CEF/clustering retune).
3. ~~2×ConvNeXt-T (~56M) same-ckpt dual mean~~ **closed** (#19 25.8706 null/degenerate; exact identity to champion).
4. ~~Cross-arch dual density mean (T + ViT-S/16)~~ **closed** (#20 34.236; no weight/backbone retune).
5. ~~Multi-level feature fusion mid+final mean~~ **closed** (#21 209.275; no stage/weight retune).
6. Low-dense overcount (bias +5 on 10–30) only if free / does not trade away 200+.
7. ~~Dense-specific second stage (spatial dual-path)~~ **closed** (#16 26.232).
8. ~~Otsu valley hard-filter form~~ **closed** (#22 32.259; no threshold-variant retune).
9. ~~Pre-norm hard-filter order~~ **closed** (#23 28.872; no order retune).
10. ~~Per-exemplar ROI-norm before mean~~ **closed** (#24 30.218; no per-exemplar norm-variant / ROI-norm formula retune).
11. ~~Per-exemplar pre-mean abs hard-filter~~ **closed** (#25 27.114; no cut-placement / pre-filter threshold-form retune).
12. **Mass-weighted exemplar aggregation (#26) — WIN, now default** — keep; next structural #27 stacks on top (still not fs/gate+scalar/ADEI retune/multi-scale box/FM/DSS/CTXSIM/CEF/same-ckpt dual/cross-arch density mean/multi-level mean/Otsu threshold form/pre-norm order/per-exemplar ROI-norm/per-exemplar pre-mean filter; #1–#25 closed).

## Code / runner quick reference

| Item | Path / value |
|---|---|
| Main script (thin entry) | `/tmp/opencode/cdino/convolutional_counting.py` → `tf_pipeline.entry.main` |
| Pipeline package (local) | `/tmp/opencode/cdino/tf_pipeline/{args,backbones,features,pipeline,postprocess,metrics,tags,state,entry}.py` |
| Pipeline package (server) | `/data/cdino_run/tf_pipeline/` |
| Readout | `src/readout.py` (`readout_count`: density= pure sum) |
| Adapter | `/data/cdino_run/dinov3_convnext_adapter.py` (img_size reshape only) |
| Best subset runner pattern | `run_adei_sub286.sh` args `1.5 1 8` |
| #14 runner | `/data/repro/run_msx_sub286.sh` (and local `/tmp/opencode/run_msx_sub286.sh`) |
| #15 runner | `/data/repro/run_fm_sub286.sh` (local `/tmp/opencode/run_fm_sub286.sh`), args `1.5 0.9 2.0` |
| #16 runner | `/data/repro/run_dss_sub286.sh` (local `/tmp/opencode/run_dss_sub286.sh`), args `1.5 1.5 5` |
| #17 runner | `/data/repro/run_cx_sub286.sh` (local `/tmp/opencode/run_cx_sub286.sh`), args `1.5 0.5 1.3` |
| #18 runner | `/data/repro/run_cef_sub286.sh` (local `/tmp/opencode/run_cef_sub286.sh`), args `1.5` |
| #19 runner | `/data/repro/run_dual_sub286.sh` (local `/tmp/opencode/run_dual_sub286.sh`), args `1.5` |
| #20 runner | `/data/repro/run_xd_sub286.sh` (local `/tmp/opencode/run_xd_sub286.sh`), args `1.5` |
| #21 runner | `/data/repro/run_mlv_sub286.sh` (local `/tmp/opencode/run_mlv_sub286.sh`), args `1.5` |
| #22 runner | `/data/repro/run_otsu_sub286.sh` (local `/tmp/opencode/run_otsu_sub286.sh`), args `1.5` |
| Subset JSON | `/data/repro/val_subset286.json` (locked, seed 20260922) |
| Journal CLI | `python3 scripts/journal.py --kind <kind> --detail "..."` workdir `/home/qkun/cac` |
| Conformance | `python3 scripts/conformance.py` must exit 0 before any commit |
| Per-image CSVs (analysis) | `/tmp/opencode/perimg/*.csv` + server `results/per_image_*.csv` |
| Known local md5 after #14 | `f4ccb2a419b736eee833dd84e632ce6a` (both sides) |
| Known local md5 after #15 | `43bf9f95af5bec72c798a5af3d87c87f` (both sides) |
| Known local md5 after #16 | `a55db47da879126a73568bed3c697b76` (both sides) |
| Known local md5 after #17 | `af8e18367cafef105e9d720ad675fedd` (both sides) |
| Known local md5 after #18 | `5871906da53e1c44d0e3a70ce9987053` (both sides) |
| Known local md5 after #19 | `4dc1a641d570023e5fe3789288a0f4a7` (both sides; dual_convnext; repaired null-check) |
| Known local md5 after #20 code | `8d68623a602520a8f63de42f5d5afd2f` (both sides; dual_density + dual_convnext) |
| Known local md5 after #21 code | `74fd01adfe8c2d4776cba929fda81ce5` (both sides; mlvl_fuse + dual_* wires) |
| Known local md5 after #22 code | `ee6adaa97d43dbc438fc83c894c6108c` (both sides; filter_otsu Otsu form) |
| #19 runner md5 | `90f6a03221ffc39e6aca3c6a83f40c6e` (both sides) |
| #20 runner md5 | `e88c762ddb73ab0a86372291a8b56c1b` (both sides) |
| #21 runner md5 | `ada90bba17d82afcabc8f84e886586c7` (both sides) |
| #22 runner md5 | `4fff07fac4712efa3410029a617dc9a9` (both sides) |
| #23 code md5 | `c1c9e5662bd6c25fd9b399b69f6f80fe` (both sides; filter_prenorm) |
| #23 runner md5 | `0d636d49bb2471679ae947a6938e6b6d` (both sides; `/data/repro/run_pn_sub286.sh`) |
| #23 runner | `/data/repro/run_pn_sub286.sh` (local `/tmp/opencode/run_pn_sub286.sh`), args `1.5` |
| #24 code md5 | `4fa9c1505acd5c26b1b5686c5be229cf` (both sides; roi_norm_per_exemplar mean-order fix) |
| #24 runner md5 | `be2b3e4e95fcb8c49703293bb9ef3a91` (both sides; `/data/repro/run_pex_sub286.sh`) |
| #24 runner | `/data/repro/run_pex_sub286.sh` (local `/tmp/opencode/run_pex_sub286.sh`), args `1.5` |
| #25 code md5 (pre-refactor) | `aaf439ba43fffe324372bc5cf9358db1` (both sides; per_exemplar_filter) |
| #25 runner md5 | `79f19dfed3cb4f05a8d539d7f2d1025d` (both sides; `/data/repro/run_pef_sub286.sh`) |
| #25 runner | `/data/repro/run_pef_sub286.sh` (local `/tmp/opencode/run_pef_sub286.sh`), args `1.5` |
| #26 code md5 (refactored post_process + mass_weight) | `9e2bbad211bfb28da0b9647ed25cd44b` (monolith, superseded by package) |
| SE package entry md5 | `6467702029594489bdd79ffd1d3ac5c9` (both sides) |
| SE package pipeline md5 | `5e79c0475835de222e4583d2dafc4bc2` (both sides) |
| SE package tags md5 | `f758197ed25f334b435b6d6f04d691d5` (both sides) |
| SE null-check | full subset286 EXTENDED_METRICS identical to #26 after package split |
| #26 runner md5 | `203fa8ba2d6302e9b48c9d39eb236a79` (both sides; `/data/repro/run_mwex_sub286.sh`) |
| #26 runner | `/data/repro/run_mwex_sub286.sh` (local `/tmp/opencode/run_mwex_sub286.sh`), args `1.5` |
| Adapter md5 after #18 | `315c5e653f3976e585392413b5defc33` (both sides) |
| #18 runner md5 | `b21b7be011dbff61f3a23197c649a615` (both sides) |

## Pipeline facts (stable)

- ConvNeXt-T 512 → stride 32 → 16×16 per crop → DEI twice → merged **64×64** grid.
- ADEI: pass-1 crop mass ≥ `thresh × median` (annotation-overlapping crops excluded) → re-split selected crops depth `d`, merge back to 64-grid; max `m` crops.
- `post_process_density_map`: mean (or max) exemplar maps → minmax → `/ norm_coeff` (ROI mean of annotation boxes / scaling_coeff) → hard filter `thresh=(1/max_box_area)*filter_thresh_scale` → density integral.
- DFSCALL ~413 calls / 286 images (ADEI re-entry).
- subset286 ≈ 90–110 s at ~3 it/s; one GPU job at a time; RTX 3060 12GB.
- Launch: `ssh -f cac-server 'cd /data/repro && setsid nohup bash script args > outer.log 2>&1 < /dev/null &'`; wait with `while pgrep -f "[c]onvolutional_counting"; do sleep 6; done` (blocking wait allowed; never `sleep N; ssh` status polls).
- Process count: bracketed grep only.

## Error structure on best (subset286 #9c)

| GT bin | n | MAE | bias |
|---|---:|---:|---:|
| 0–10 | 28 | 4.16 | +4.16 |
| 10–30 | 123 | 7.02 | +4.72 |
| 30–80 | 80 | 17.52 | −9.48 |
| 80–200 | 37 | 39.62 | −28.64 |
| **200+** | 18 | **196.47** | **−169** |

200+ ≈ 48% of total absolute error. Halving 200+ → overall ≈19.6 (goal math).  
Prior wall 198.68 broken by #26 (196.47); **new wall = 196.47**. Still ~11× above sparse bins — dense remains the gap.

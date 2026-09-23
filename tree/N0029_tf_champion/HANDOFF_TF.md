# HANDOFF — Training-free (N0029)

**Date:** 2026-09-23 (post #18)  
**Status:** Active · subset locked · **rapid goal = subset286 ≤ 20** · **autonomous structural work only** (user: do not ask to continue)  
**Best subset286:** **25.818** (#26 mass-weighted exemplar + ADEI t1.5d1m8; prior best 25.871 #9c) · **full val board:** 26.485 (no new full val without user request)  
**#25 per-exemplar pre-mean abs hard-filter:** **27.114 refuted** — family closed (no cut-placement / pre-filter threshold-form retune).  
**#26 mass-weighted exemplar aggregation (`--mass_weight_exemplar`):** **25.818 WIN** — MAE < 25.871 **and** 200+ **196.47** < prior wall 198.68 (first wall beat); mechanism locked as default champion component for follow-ups.  
**#15 AdaCount feature modulation:** **27.617 refuted** — family closed (no alpha/gamma retune).  
**#16 structure-routed dense second stage:** **26.232 refuted** — family closed (no dense_struct_thresh/kernel retune; 200+ wall was 198.68).  
**#17 TFCounter context-aware similarity:** **55.935 refuted hard** — family closed (no fusion_ratio/T_div retune; bg-prototype subtract wiped mass).  

**#18 CountSE multi-res CEF soft exemplar:** **25.921 refuted** — family closed (no CEF/clustering retune; MAE +0.0503 outside near-tie, 200+ 198.89 worse than wall).

**#19 same-checkpoint 2×ConvNeXt-T dual mean:** **25.8706 DEGENERATE/NULL closed** — null-check run exact identity to champion (per-image same_pred 286/286, maxdiff 0.0); launches 31.874/35.98 were mechanism-invalid (repair entries); no same-ckpt dual, no fusion-form retune.

**#20 cross-arch dual density (T + ViT-S/16):** **34.236 refuted hard** — family closed (no fusion-weight/backbone/img_size retune; sparse inflated, shared dense undercount; 200+ 192.91 alone insufficient).

**#21 multi-level feature fusion (`--mlvl_fuse`):** **209.275 refuted hard** — family closed (no stage-index/weight/align retune). Mid stage (384ch, 128-grid) + final mean blew up response: ratio_sum 3.999, 255/286 over, 200+ 404.03.

**#22 Otsu hard-filter form (`--filter_otsu`):** **32.259 refuted hard** — family closed (no Otsu bins/threshold-variant/percentile/hysteresis retune). Sparse improved (0-10 2.15, 10-30 5.92) but mid/dense undercount: 30-80 23.05, 80-200 59.91, 200+ **243.18** worse than wall 198.68, ratio_sum 0.563 (221/286 under).

**#23 pre-norm hard-filter order (`--filter_prenorm`):** **28.872 refuted** — family closed (no filter/norm order retune). Abs cut before /norm kept almost all pixels → sparse overcount (0-10 7.32, 10-30 9.64); 200+ **204.27** still worse than wall; post-norm fs=0.5 is load-bearing.

**#24 per-exemplar ROI-norm before mean (`--roi_norm_per_exemplar`):** **30.218 refuted** — family closed (no median/min/max-box per-exemplar norm variants, no other ROI-norm formula retune). Sparse improved (0-10 **2.51**, 10-30 **5.66**) but mid/dense worse (30-80 23.61, 80-200 53.93, 200+ **221.73** vs prior wall 198.68); ratio_sum **0.650** undercount. Mean-order bug fixed pre-launch (split stacked before mean); single launch PID 30337, rc=0.

**#25 per-exemplar pre-mean abs hard-filter (`--per_exemplar_filter`):** **27.114 refuted** — family closed (no cut-placement / pre-filter threshold-form retune). Bins 0-10 4.64, 10-30 7.57, 30-80 18.26, 80-200 40.15, 200+ **208.18** (worse than prior wall); ratio_sum 0.842; pre-cut near-noop on minmax scale (kept_per ~85–90%). Single launch rc=0, journaled once.

**#26 mass-weighted exemplar aggregation (`--mass_weight_exemplar`):** **25.818 WIN** — ROI-mass weights on minmax exemplar maps then sum, then champion ROI-norm+post-cut; MAE 25.818 < 25.871, 200+ **196.47** < prior wall 198.68 (first beat), bins 0-10 4.16, 10-30 7.02, 30-80 17.52, 80-200 39.62, 200+ 196.47, ratio_sum 0.776. Single launch rc=0. Keep MWEx as default; next #27 stacks orthogonally.

**SE refactor (hygiene, same session):** monolith → `tf_pipeline/` package (`args`, `backbones`, `features`, `pipeline`, `postprocess`, `metrics`, `tags`, `state`, `entry`); `convolutional_counting.py` is a thin entry (runners unchanged). Full subset286 null-check after refactor: **EXTENDED_METRICS bit-identical** to #26 (MAE `25.817842750282555`). Layout is now software-engineering standard; keep new flags modular (one module concern each).

---

## 0. Read this first

| Item | Path |
|---|---|
| **Authoritative constraints** | `constraints_and_goal.md` |
| **Full attempt ledger (CLOSED families)** | **`ATTEMPTS.md`** ← read before designing #15+ |
| Scoreboard / queue | `baseline_board.md` |
| This handoff | `HANDOFF_TF.md` |
| Archive note | `/home/qkun/cac/archive_local/README_ARCHIVED.md` |

**Mission:** training-free FSC147 **subset286 val MAE ≤ 20.0** · stack **≤64M** · must keep `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` in the inference graph · champion/board rows = **full 1286 only**.

**User directives (2026-09-23, non-negotiable):**
1. **No parameter tuning** / no gate+scalar combos before major progress; **structural innovation**.
2. **subset286 only**; full val **only on explicit user request**.
3. **Act autonomously** — do not ask whether to continue; execute, then document.
4. **All docs in English** (§5b constraints).
5. Any durable lesson → write into `ATTEMPTS.md` / constraints / this handoff **immediately** (same session).

---

## 1. Environment (locked)

```
ssh cac-server   # port 46315; creds: local/address_and_password.md
PYTHONPATH=/data/cdino_run:/data/cdino_run/src:/data/cac/src:/data/cac/scripts
HF_HOME=/data/asset/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=/data/miniconda/envs/cac/bin/python
cd /data/cdino_run
```

- GPU: RTX 3060 12GB · **one job at a time**
- **IRON RULE #0:** never `sleep N; ssh` status polls; one bracketed `pgrep`/`grep` per poll; blocking remote wait `while pgrep -f "[c]onvolutional_counting"; do sleep 6; done` is allowed
- Sync: no rsync → `scp` or `tar | ssh tar`; serialize scp vs concurrent file reads
- Speed: subset286 ≈ **1.5–2 min** (~3 it/s) · full 1286 ≈ 6.5 min
- Complex remote scripts: write local → `cat f | ssh "cat > dest"` → separate launch ssh

---

## 2. Locked evaluation protocol (constraints §6)

- **Subset (LOCKED):** `/data/repro/val_subset286.json` (n=286, 89 fine bins, seed=20260922) — **never resample**
- **IRON RULE:** experiments **subset286 only**; full val **only if user asks**
- **Success:** subset286 **≤ 20.0**
- **Champion / board:** full 1286 only; subset MAE is never a board row
- CLI: `--subset_file /data/repro/val_subset286.json --log_file results/... --no_skip true`

Concordance (best config): full 26.485 vs subset 25.871 (Δ≈0.15) — absolute levels agree; do not promote without user request.

---

## 3. Best config (LOCKED — do not re-tune fs / readout)

```
--model_name dinov3_convnext_tiny
--divide_et_impera True --divide_et_impera_twice True
--filter_background True --filter_thresh_scale 0.5
--ellipse_normalization True --ellipse_kernel_cleaning True
--count_readout density
--adaptive_dei True --adaptive_dei_thresh 1.5 --adaptive_dei_depth 1 --adaptive_dei_max 8
--input_size 512
# NO use_threshold; NO val-GT calibration; NO gate+scalar; NO dense_hr; NO multi_scale_ex; NO feature_modulation; NO dense_struct_stage; NO context_aware_sim
```

| Split | MAE | notes |
|---|---:|---|
| **full val 1286** | **26.485** | champion row (fs0.5, no ADEI in board row unless user requests refresh) |
| full val ADEI t3.0 (#8) | 26.358 | recorded before subset-only iron rule |
| **subset286 best (#26 MWEx)** | **25.818** | MWEx + ADEI t1.5d1m8; baseline 26.339; prior best 25.871 #9c |
| subset ADEI constants | exhausted | t floor 1.5, d1, m8 |
| #15 feature modulation | **27.617 REFUTED** | sparse better, dense 200+ **216.78**; family closed |
| #16 dense struct dual-path | **26.232 REFUTED** | sparse inflated; 200+ **198.64** ≈ wall; family closed |
| #17 TFCounter context-aware sim | **55.935 REFUTED** | ratio_sum 0.138, mass wiped; family closed |
| #18 CEF soft exemplar | **25.921 REFUTED** | MAE +0.0503 outside near-tie; 200+ 198.89 worse; family closed |
| #19 same-ckpt dual mean | **25.8706 NULL** | exact identity to champion; degenerate; family closed |
| #20 cross-arch density mean | **34.236 REFUTED** | sparse inflated, shared dense undercount; family closed |
| #21 multi-level feature fusion | **209.275 REFUTED** | ratio_sum 3.999 overcount; 200+ 404; family closed |
| #22 Otsu hard-filter form | **32.259 REFUTED** | ratio_sum 0.563 undercount; 200+ 243; family closed |
| #23 pre-norm hard-filter order | **28.872 REFUTED** | sparse overcount (near no-filter); 200+ 204; family closed |
| #24 per-exemplar ROI-norm before mean | **30.218 REFUTED** | sparse better, mid/dense worse; 200+ **221.73**; ratio_sum 0.650; family closed |
| #25 per-exemplar pre-mean abs filter | **27.114 REFUTED** | 200+ **208.18** worse than old wall; pre-cut near-noop; family closed |
| #26 mass-weighted exemplar | **25.818 WIN** | 200+ **196.47** first wall beat; default champion component |

---

## 4. Error structure (why structural dense work)

Best subset286 bins: 0–10 **4.08**/+4 · 10–30 **6.90**/+5 · 30–80 **17.34**/−8 · 80–200 **39.82**/−30 · **200+ 198.68/−171**.  
200+ ≈ **48% of AE**. Root cause: `filter_background` + ROI-norm wipe spread-thin dense mass (fs=0.5 partially mitigates; dense still broken). **Not a subset bug.**

---

## 5. Code & artifacts

| What | Where |
|---|---|
| Main script | `/data/cdino_run/convolutional_counting.py` (sync from `/tmp/opencode/cdino/…`) |
| Readout | `/data/cdino_run/src/readout.py` |
| Adapter | `/data/cdino_run/dinov3_convnext_adapter.py` (69 lines; img_size only reshapes tokens) |
| Runners | `/tmp/opencode/run_*.sh` → `/data/repro/run_*.sh` |
| Results | `/data/cdino_run/results/results_*.csv`, `per_image_*.csv` |
| Analysis copies | `/tmp/opencode/perimg/*.csv` |
| Constraints / board / ledger | `tree/N0029_tf_champion/{constraints_and_goal,baseline_board,ATTEMPTS}.md` |
| Journal | `/home/qkun/cac/journal/events.jsonl` via `scripts/journal.py` only |

---

## 6. Queue (do in order; autonomous — no permission prompts)

1. **Read `ATTEMPTS.md` closed table** — never retry a closed family.
2. **Next structural attempt (#27):** design after **#26 WIN (25.818, 200+ 196.47)**. MWEx is now the default champion component — stack a **new orthogonal mechanism** on top. Must not retune MWEx weights as free hyper-params; not fs, gate+scalar, ADEI retune, multi-scale box kernels, gated HR, AdaCount FM, structure-routed dual-path spatial join, TFCounter bg-prototype fusion, CountSE CEF soft exemplar, same-ckpt dual mean, cross-arch density mean, multi-level mid+final mean, Otsu/threshold-form filter, pre-norm filter order, per-exemplar ROI-norm / any ROI-norm formula variant, per-exemplar pre-mean filter cut-placement (all #1–#25 closed).
3. Pre-register with `journal.py --kind protocol_decision` **before** code/run (**#26 pre-reg+run+close done**; also #25, #24, #23, #22, #21, #20).
4. One launch, one result journal, `conformance.py` OK.
5. subset286 only; full val only on user request.

**Explicit non-goals:** `use_threshold` · val-GT scale on board · sleep-polling · re-sampling subset · champion claim from subset · asking user to continue.

---

## 7. Cleanup (2026-09-22, historical)

Local `/home/qkun/cac` ~220M→46M; server `/data` 36G→28G; N0001 weights purged; old subsets deleted. Keep: miniconda, asset/hf, dataset, N0029, subset286.

---

## 8. Next session first actions

1. Read `ATTEMPTS.md` + `constraints_and_goal.md`.
2. Confirm best still **25.818** / full **26.485** / **200+ wall 196.47**.
3. #20–#25 closed as listed. **#26 WIN (25.818) default.** → design/pre-register **#27** (orthogonal stack on MWEx) → implement → launch ONCE → journal once → close/update docs. Full SE modular refactor of `convolutional_counting.py` in progress (see ATTEMPTS code md5 row).
4. **Do not ask the user whether to proceed.**

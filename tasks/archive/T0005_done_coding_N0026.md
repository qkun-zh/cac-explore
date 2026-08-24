# T0005_coding_N0026_res_sweep

- status: done             # pending -> claimed_<agent> -> done | cancelled
- evidence: scripts/eval_res_sweep.py green on server (smoke: extreme grids 224+518, 20 imgs each, SMOKE_OK rows=40 nonfinite=0; 518 ran bs4 no OOM). Full sweep 5 sizes x 1286 val imgs: 392 baseline reproduced MAE=20.4408 (champion 20.4376, AMP drift). VERDICT H0035: PASS — S=448 improves RMSE -7.97 (79.856→71.883) with dMAE -0.089 (bar: dRMSE<=-3 AND dMAE<=+0.5); S=518 RMSE -14.53 (65.326) but MAE +5.20 fails the MAE guard alone; 224/308 degrade both. KEY tail [500,inf) (17 imgs): monotone res gain — MAE 486.7@392 → 405.8@448 → 318.8@518; RMSE 604.9 → 531.7 → 440.3; SSE share 75.9% → 72.3% → 60.1% (cell quantization mechanism CONFIRMED in tail). Tercile edges fixed once from full val GT distribution: [17.000, 46.000]. Results: tree/nodes/N0026_res_sweep/{res_results.json (per-image x per-res),res_sweep_log.txt}; server copies at /data/repo/tree/nodes/N0026_res_sweep/. No engine/model/data files touched; nothing committed.
- created: 2026-08-24T09:17:26+0800
- role: coding
- node: tree/nodes/N0026_res_sweep
- inputs: tree/nodes/N0026_res_sweep/idea.md + memory/failure_modes.md + code/engine/train.py (evaluate(), lines 145-170) + code/data/fsc147.py + tree/nodes/N0021_dino_partialft/{model.py,config.py}
- outputs: scripts/eval_res_sweep.py (parameterized by input_size) + stratified MAE/RMSE/SSE tables + smoke green on server + flip card done
- notes: EVAL-ONLY — do NOT modify engine, data code, or champion node files. dynamic_img_size=True already VERIFIED in champion model.py:38; no model edits needed. Same checkpoint/env as T0004.

SPEC:
1. `scripts/eval_res_sweep.py --input_size {224|308|448|518}` (+392 baseline pass): rebuild val
   loader per size via FSC147Density(root, size, "val") (counts are size-invariant,
   sum-conserving resize fsc147.py:53-54); load /data/runs/N0021_dino_partialft/best.pth;
   one forward pass per image (model consumes imgs+bboxes at any S; ps=S//14 exact for all sizes).
2. Log per resolution: overall MAE/RMSE AND tercile-stratified MAE/RMSE — tercile edges fixed
   ONCE from the val GT distribution and shared across resolutions.
3. FREE RIDER (on the 392 pass): SSE-share decomposition by GT bucket [0,25)/[25,75)/[75,200)/[200,500)/[500,inf).
4. Final table prints ALL five configs + H0035 verdict with numbers (pass: some non-392 res has
   RMSE improvement >=3 AND MAE regression <=0.5 vs 392 baseline; disproof: every non-392 res
   degrades MAE >2.0).
5. Smoke FIRST on server (ssh cac-server): env HF_HOME=/data/asset/hf HF_ENDPOINT=https://hf-mirror.com
   /data/miniconda/envs/cac/bin/python scripts/eval_res_sweep.py --smoke (few batches @224 and @518,
   the extreme grids), then full passes in tmux; batch_size<=4 for 448/518 to avoid OOM.

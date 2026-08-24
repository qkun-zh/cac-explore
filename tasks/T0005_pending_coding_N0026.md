# T0005_coding_N0026_res_sweep

- status: pending          # pending -> claimed_<agent> -> done | cancelled
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

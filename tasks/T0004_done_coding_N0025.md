# T0004_coding_N0025_eval_readout

- status: done             # pending -> claimed_<agent> -> done | cancelled
- evidence: scripts/eval_readout_lab.py green on server (smoke 20 imgs SMOKE_OK rows=20 nonfinite=0; box scaling cross-checked vs Dataset). Full 1286-img val pass MAE_raw=20.441 (= champion 20.4376 within AMP drift). Verdicts: H0033 FAIL — ttnorm MAE 108.261 (+87.8 vs raw), median g clamped at 5.0, exemplar-box integrals ~0.03-0.19 (not ~1); H0034 FAIL — split-half isotonic cross-fit hurts both directions (A->B -2.160, B->A -1.555); trims all worse (trim0.5% +0.241, trim1% +0.756, trim2% +2.238); deployment gate BLOCKED (ship raw counts). Results: tree/nodes/N0025_eval_readout/{lab_results.txt,dump_val.jsonl}
- created: 2026-08-24T09:17:26+0800
- role: coding
- node: tree/nodes/N0025_eval_readout
- inputs: tree/nodes/N0025_eval_readout/idea.md + memory/failure_modes.md + code/engine/train.py (evaluate(), lines 145-170) + code/data/fsc147.py + tree/nodes/N0021_dino_partialft/{model.py,config.py}
- outputs: scripts/eval_readout_lab.py (standalone; supersedes any engine --dump_preds edit) + per-image JSONL dump + --analyze mode printing H0033/H0034 verdicts with numbers + smoke green on server + flip card done
- notes: EVAL-ONLY — do NOT modify engine, data code, or champion node files. Build model via load_module of champion config.py/model.py exactly as train.py does, load `/data/runs/N0021_dino_partialft/best.pth` state_dict, eval mode.

SPEC:
1. `scripts/eval_readout_lab.py --dump` — one val pass @392 bs8 AMP, writes per-image JSONL rows:
   {"img_id", "N_gt", "N_hat_raw", "N_hat_ttnorm", "N_hat_trimmed_05", "N_hat_trimmed_10",
    "N_hat_trimmed_20", "box_gains": [g1,g2,g3], "box_integrals": [i1,i2,i3]}
   - Read ALL 3 exemplar boxes from annotation_FSC147_384.json box_examples_coordinates,
     rescaled to S-space like fsc147.py:57-61 (Dataset exposes only exemplar[0] — read JSON directly).
   - TT-Norm per idea.md: integral_k = sum of rho over cells whose centers fall in box_k;
     if all integrals < 1e-6 → g=null and N_hat_ttnorm=N_hat_raw (skip); else g=median(1/max(i_k,1e-6))
     clamped to [0.2,5].
   - Trimmed arms: drop top {0.5%,1%,2%} hottest cells of raw 28×28 rho, then sum.
2. `scripts/eval_readout_lab.py --analyze <dump.jsonl>` — OFFLINE only:
   split-half isotonic (sklearn IsotonicRegression(out_of_bounds="clip"), fit half → apply other,
   BOTH directions MAE pre/post; regularize to identity outside p10–p90 of fit-half predictions),
   trimmed-arm MAE/RMSE table, then PRINT H0033 verdict (bar: ttnorm MAE <=19.5 pass; >20.64 or
   >50% images |g−1|<0.01 fail) and H0034 verdict (held-out-half improvement >=0.6 pass; <0.3 or
   curve within ±5% of identity fail), each with explicit numbers.
3. Smoke FIRST on server (ssh cac-server): env HF_HOME=/data/asset/hf HF_ENDPOINT=https://hf-mirror.com
   /data/miniconda/envs/cac/bin/python scripts/eval_readout_lab.py --smoke  (few batches, verify
   JSONL schema + no NaN), then full val pass (tmux if >1min). Checkpoint: /data/runs/N0021_dino_partialft/best.pth.
4. Leak rules from idea.md are binding: isotonic cross-fit numbers ONLY for the verdict;
   test-set full-val refit is a deployment decision gated on split-half stability (state it in output).

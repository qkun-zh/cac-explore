# STATE — Session 2026-08-29 (Lead=qkun-local, temply direct)

**Mode**: Free-Research (autonomous, temply Lead does all roles directly per user directive). Frozen backbone hard constraint; no unfreeze.
**Preflight**: SERVER_OK (port 44387). Server engine has +5-line periodic-ckpt hotfix (save_every, default off).

## Champion (frozen regime, UNCHANGED)
**N0054_xscale_exemplar** (GCA + XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED. AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384.

## Closed negative evidence (30ep @384) — exemplar + aggregation + count/readout FULLY MAPPED
| Node | Axis | Delta |
|---|---|---|
| N0054 | — | **19.647 CHAMPION** |
| N0055 | info-add 2K keys | +1.19 |
| N0056 | info-add extra fine scale (exemplar agg) | +3.06 |
| N0057 | consumer swap | +1.43 |
| N0058 | producer swap (part-pool+−42%cap) | +3.50 floor |
| N0059 | producer swap (matched, full-token) | +1.31 floor |
| N0060 | 2nd coarse MAX on SAME ROI | +3.24 |
| N0037 | N·p factorization readout | +0.5 |
| N0061 | multiplicative count autoscale | +3.855 |

**Laws**: H0085 single-slider (exemplar coarse-summary slot=1 only); H0089 readout-multiplier premise error (gt_d raw non-unit-mass, output multiplier never count-weights); H0081 producer+consumer attention load-bearing 0.59.

## Active: N0062_fine_decoder (H0090) — NEW receiver-resolution axis
**Design**: frozen native-1/4 h1 (96ch @1/4, hs[1] currently dead) injected into DensityDecoder INPUT only: 1x1 96->8 + GN(2,8) concat -> in_ch 192->200 (+19.2k params, total 31.34M). Exemplar/condenser/GCA/attention untouched; use_fine_decoder=False bit-identical (verified forward diff 0.0, param delta 0, smoke PASS). Targets tail cell-quantization mass loss (75.86% SSE in 17 imgs N>=500, N0026 tail error falls with res). Distinct from N0056 (exemplar agg) and RGA (output bias). Gates: CONFIRM <19.45 (2nd seed if <19.40), WEAK-KEEP 19.45-20.0, FAIL >20.0. H0090 DISPROVED IF >20.4.
**Smoke**: True 31.34M OK, False 31.32M identical, shapes (2,1,96,96), finite, n_aux retained.
**Status**: code+smoke done, hypothesis booked (H0090), index 25 hyps, launching 30ep on server (tmux node_N0062_fine_decoder).

## Server gotchas
- run_node.sh `git pull` hangs: launch tmux directly (export PATH=/data/miniconda/bin). python /data/miniconda/envs/cac/bin/python; HF /data/asset/hf + hf-mirror. Sync via scp.

## Queue
1. N0062 running 30ep — poll single ssh, early-stop ep16+ >=+1.5 vs parent best.
2. Next: feedback×4 + synthesis after result; if N0062 FAILs, receiver-resolution also mapped → remaining frozen levers exhausted per hardening LOS.

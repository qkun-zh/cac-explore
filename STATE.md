# STATE — Session 2026-08-29 (Lead=qkun-local)

**Mode**: Free-Research (autonomous). Directive: combine PoM (2604.06129) + ParTY (2603.09611) modules autonomously, no questions.
**Preflight**: creds ROTATED (port 44387) → install_key.py → SERVER_OK. Server engine has +5-line periodic-ckpt hotfix (save_every, default off).

## Champion (frozen regime, UNCHANGED)
**N0054_xscale_exemplar** (GCA + XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED. AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384.

## Active node: N0061_countnorm (RUNNING, 30ep)
**H0088 test — P1 count-normalized readout (variant ii).** CountNormHead: read GAP(fine)+e_mean → z=clamp(W2·GELU(W1[·]),−2,2), W2 zero-init → identity-at-init (f=1, forward==champion at ep0); `out.density = champion_density * exp(z)`. Count-weighted MSE steers gradient OUT of the RMSE-tail (17/83 N≥500 = 76% SSE). Single switch use_countnorm (False = exact champion bit-identical; verified 0.000 diff). +24,705 params → head 3.53M / total 31.35M.
**Research brief (2026 SOTA)**: FSC147 now CoDi 5.81 (diffusion) / GeCo2 9.38 (detection) — unportable (huge fine-tuned backbones). Only portable frozen-lever = readout/target shape + tail + kernel σ.
**H0088** IF density re-scaled by count-consistency factor density·n_hat/sum(density) with n_hat from shared-interface count trunk IN frozen N0054 THEN MAE < 19.647 BECAUSE count-weighted MSE shrinks the RMSE-tail without spatial/ROI/operator change. DISPROVED IF > 20.4.
**Gates**: CONFIRM <19.45 (2nd seed if <19.40) · WEAK-KEEP 19.45–20.0 · FAIL >20.0; early-stop ep16+ ≥+1.5.

## Frozen-regime exemplar axis table (30ep @384) — FULLY MAPPED
| Node | Axis | Delta |
|---|---|---|
| N0054 GCA+XScale | — | **19.647 CHAMPION** |
| N0055 XScale-Key | info-add 2K keys | +1.19 |
| N0056 XFine | info-add extra fine scale | +3.06 |
| N0057 cond-matcher | consumer swap | +1.43 |
| N0058 PMOM | producer swap (part-pool+−42%cap) | +2.17 same-ep / +3.50 floor |
| N0059 PoM-Morph | producer swap (matched, full-token) | +1.31 floor / +2.13 same-ep |
| N0060 XScale-MAX | 2nd coarse MAX on SAME ROI | **+3.24** |

**Refined law**: every 2nd spatial summary + every operator swap + cardinality/fine-entropy all NEGATIVE (H0085 booked). Exemplar coarse-summary/aggregation axis fully mapped with ONE positive (mean-XScale +0.95). Moving forward: readout/tail-count direction (N0061) NOT on exemplar slot. Producer+consumer attention load-bearing (H0081 0.59).

## Server gotchas
- run_node.sh `git pull` hangs server-side: launch tmux directly (export PATH=/data/miniconda/bin). tmux libtinfo warning non-fatal. python /data/miniconda/envs/cac/bin/python; HF /data/asset/hf + hf-mirror. Sync via scp; `local/` gitignored (best.pth in local/feedback_src_N00{58,59,60}/).

## Queue
1. ✅ N0058-60 closed + N0061: research brief→idea(finalize, variant-ii chosen)→novelty(0.383)→code→local smoke (identity-at-init 0.000)→server smoke GREEN→launched.
2. Poll curve vs N0054; apply CONFIRM/WEAK/FAIL; 2nd seed if <19.40.
3. Feedback×4 + Diagnostic + synthesis + calibration + H0088 ledger + tree flip + commit/push.
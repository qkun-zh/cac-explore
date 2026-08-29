# STATE — Session 2026-08-29 (Lead=qkun-local)

**Mode**: Free-Research (autonomous). Directive: combine PoM (2604.06129) + ParTY (2603.09611) modules autonomously, no questions.
**Preflight**: creds ROTATED (port 44387) → install_key.py → SERVER_OK. Server engine has +5-line periodic-ckpt hotfix (save_every, default off).

## Champion (frozen regime, UNCHANGED)
**N0054_xscale_exemplar** (GCA + XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED. AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384.

## Today: 4 nodes closed (all NEGATIVE) — exemplar + count/readout axes now CLOSED
**N0058** producer PMOM (part-pool+−42%cap) best 23.151. **N0059** producer PoM-Morph (matched capacity) best 20.958. **N0060** XScale-MAX 2nd coarse best 22.886. **N0061** count-normalized autoscale best 23.502(+3.855 floor). H0080/82/83/84/88 all refuted; H0081→0.59; H0085 (single-slider), H0089 (readout-multiplier premise error) booked.

## Frozen-regime negative evidence (30ep @384) — exemplar + aggregation + count/readout FULLY MAPPED
| Node | Axis | Delta |
|---|---|---|
| N0054 | — | **19.647 CHAMPION** |
| N0055 | info-add 2K keys | +1.19 |
| N0056 | info-add extra fine scale | +3.06 |
| N0057 | consumer swap | +1.43 |
| N0058 | producer swap (part-pool+−42%cap) | +3.50 floor |
| N0059 | producer swap (matched, full-token) | +1.31 floor |
| N0060 | 2nd coarse MAX on SAME ROI | +3.24 |
| N0037 | N·p factorization readout | +0.5 |
| N0061 | multiplicative count autoscale | +3.855 |

**Refined law**: exemplar coarse-summary/aggregation axis (H0085) AND count/readout-normalization axis (H0089) both closed. GCA (additive, 0.02-attenuated, zero-init) = unique positive count-side module; mean-XScale = unique positive exemplar-side module. Producer+consumer attention load-bearing (H0081 0.59). N0061 root cause = PREMISE ERROR (gt_d is non-unit-mass; output multiplier never count-weights).

## Next direction (remaining champion-faithful frontier)
The exemplar axis and count/readout axis are done. Remaining levers, in order: (a) count-as-SUPERVISION via the engine's dead `tail_reweight` path (train.py:334-339) — engine change, currently forbidden; (b) a shared-interface aux on frozen features OUTSIDE count/exemplar axes; (c) regime/extent change (unfreeze / different eval). The frozen-head LOS is hardening; breakthrough likely requires (a) or (c).

## Server gotchas
- run_node.sh `git pull` hangs server-side: launch tmux directly (export PATH=/data/miniconda/bin). tmux libtinfo warning non-fatal. python /data/miniconda/envs/cac/bin/python; HF /data/asset/hf + hf-mirror. Sync via scp; `local/` gitignored (best.pth in local/feedback_src_N00{58,59,60,61}/).

## Queue
1. ✅ N0058-61 full cycles closed (idea→code→smoke→run→feedback×4→synthesis→calibration→ledger→tree flip). Ledger ~49 lines; index 24 hyps (H0085, H0088, H0089 booked; H0081 0.59).
2. Recommend next: enable engine `tail_reweight` (count-as-supervision) as a documented deviation (opens a NEW untested axis), OR regime/extent change. Do NOT retry multiplicative readout or ROI spatial summaries/operator swaps.
3. Remaining: commit & push session close.
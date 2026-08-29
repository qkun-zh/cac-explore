# STATE — Session 2026-08-29 (Lead=qkun-local)

**Mode**: Free-Research (autonomous). Directive: combine PoM (2604.06129) + ParTY (2603.09611) modules autonomously, no questions.
**Preflight**: creds ROTATED (port 44387) → install_key.py → SERVER_OK. Server engine has +5-line periodic-ckpt hotfix (save_every, default off).

## Champion (frozen regime, UNCHANGED)
**N0054_xscale_exemplar** (GCA + XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED. AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384.

## Today: 3 nodes closed (all NEGATIVE) — exemplar aggregation coarse-summary axis FULLY MAPPED
**N0058_pompart_exemplar** — producer PMOM (2×2 part-pool + polynomial moments), early-stopped E15. best 23.151 = +2.17 same-ep / +3.50 floor. Confound: −42% cap + part-pool. H0080 refuted.

**N0059_pom_morph** — producer PoM-PolyMorpher (capacity-matched, full 49 tokens, ParTY-excluded). best 20.958 = +1.31 floor / +2.13 same-ep. KILL. H0082 refuted; H0081 strengthened (0.5→0.59, load-bearing producer attention capacity-independent).

**N0060_xscale_max** — H0084: 2nd coarse MAX-order-statistic summary on the SAME fused prototype beside mean-XScale. best **22.886 = +3.24 floor**, final 23.010, RMSE +11. FAIL. H0083/H0084 refuted; refined negative **H0085**: coarse-summary slot is a SINGLE slider — exactly one additive coarse summary (mean-XScale) is positive; any 2nd ROI summary over the same spatial source is harmful/neutral.

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

**Refined law**: EVERY second summary of the same spatial source (mean/max/grid/part-pool: N0055/56/60) AND every attention/operator swap (N0057/58/59) is NEGATIVE. The exemplar coarse-summary/aggregation axis is fully mapped with exactly ONE positive: champion mean-XScale (+0.95). Producer+consumer attention is load-bearing (H0081 0.59). The frozen-regime head is essentially near-converged; champion-faithful progress must change the INTERFACE (aux on frozen backbone features / regime/extent change), not add parallel projectors onto the same prototype.

## Server gotchas
- run_node.sh `git pull` hangs server-side: launch tmux directly (export PATH=/data/miniconda/bin). tmux libtinfo warning non-fatal. python /data/miniconda/envs/cac/bin/python; HF /data/asset/hf + hf-mirror. Sync via scp; `local/` gitignored (best.pth in local/feedback_src_N00{58,59,60}/).

## Queue
1. ✅ N0058, N0059, N0060 full cycles closed (idea→code→smoke→run→feedback×4→synthesis→calibration→ledger→tree flip). Ledger 47 lines: H0080/82/84/83 refuted, H0083/84, H0085 booked, H0081 0.59. Index 22 hyps.
2. Next: exemplar coarse-summary & aggregation axes closed. Recommend a new interface direction (e.g. shared-interface aux on frozen backbone features, or regime/extent change) — do NOT retry ROI spatial summaries or operator swaps.
3. Remaining: commit & push session close.
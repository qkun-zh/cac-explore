# STATE — Session 2026-08-29 (Lead=qkun-local, temply direct)

**Mode**: Free-Research (autonomous, temply Lead does all roles directly). Frozen backbone hard constraint.
**Preflight**: SERVER_OK (port 44387). Engine has +5-line periodic-ckpt hotfix (save_every default off).

## Champion (frozen regime, UNCHANGED)
**N0054_xscale_exemplar** (GCA + XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED. AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384.

## Closed negative evidence (30ep @384) — exemplar + aggregation + count + resolution FULLY MAPPED
| Node | Axis | Delta |
|---|---|---|
| N0054 | — | **19.647 CHAMPION** |
| N0055 | info-add 2K keys | +1.19 |
| N0056 | info-add extra fine to exemplar agg | +3.06 |
| N0057 | consumer swap | +1.43 |
| N0058 | producer swap (part-pool+−42%cap) | +3.50 floor |
| N0059 | producer swap (matched, full-token) | +1.31 floor |
| N0060 | 2nd coarse MAX on SAME ROI | +3.24 |
| N0037 | N·p factorization readout | +0.5 |
| N0061 | multiplicative count autoscale | +3.855 |
| N0062 | decoder INPUT native-1/4 h1 injection | **+1.676** (21.323 @E17) |

**Laws**: H0085 single-slider (exemplar coarse-summary=1 only); H0089 readout-multiplier premise error; H0081 attention load-bearing 0.59; H0090 decoder-receiver-resolution via raw h1 closed (least harmful density-side add but still FAIL).

## Today: 5 nodes closed (all NEGATIVE)
N0058 PMOM 23.151; N0059 PoM-Morph 20.958; N0060 XScale-MAX 22.886; N0061 countnorm 23.502 (+3.86); N0062 fine_decoder 21.323 (+1.68, milder but FAIL >20.4, RMSE +1.91). H0080/82/83/84/88/90 all refuted; H0081→0.59.

## Frozen-regime LOS hardening
Exemplar aggregation, count/readout, and decoder-receiver resolution all mapped negative under frozen+plain-MSE. GCA+mean-XScale remain unique positives. Remaining untested frozen lever is count-as-SUPERVISION (`tail_reweight` train.py:334-339, dead, needs engine change — currently forbidden). Otherwise breakthrough requires regime/extent change (unfreeze/test-time routing already shown +0.5 via N0026 19.178). Frozen-head near-optimal within architecture.

## Server gotchas
- run_node.sh `git pull` hangs: launch tmux directly (export PATH=/data/miniconda/bin). python /data/miniconda/envs/cac/bin/python; HF /data/asset/hf + hf-mirror. Sync via scp; `local/` gitignored.

## Queue
1. ✅ N0058-62 full cycles closed (idea→code→smoke→run→feedback×4→synthesis→calibration→ledger→tree flip). Ledger ~54 lines; index 25 hyps (H0090 booked refuted).
2. Next: free-research continues until MAE <19.647 — but frozen-head design space is saturated; recommend documenting LOS hardening and proposing engine `tail_reweight` as next axis (requires deviation log) or deliver final report.

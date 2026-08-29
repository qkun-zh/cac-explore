# STATE — Session 2026-08-29 CLOSE (Lead=qkun-local, User-Guided)

**Mode**: User-Guided. Preflight SERVER_OK (44387). Deviations: N0063 37.49M>32M (backbone cap test); N0065 31.33M scale-embed (BMNet+ idea). Engine patched: timeout 30→2160 (36h) + resume from best.pth (failure_modes.md).

## Champion (frozen)
**N0054_xscale_exemplar** GCA+XScale 19.647/74.05/31.32M LOCKED. AdamW 1e-3 wd0.05 cosine bs16 AMP 30ep.

## Closed 2026-08-29 (7 nodes, all NEGATIVE)
| Node | Axis | Best | Δ |
|---|---|---|---|
| N0058 | part-pool producer | 23.151 | +3.50 |
| N0059 | matched producer | 20.958 | +1.31 |
| N0060 | XScale MAX | 22.886 | +3.24 |
| N0061 | count norm | 23.502 | +3.86 |
| N0062 | decoder h1 | 21.323 | +1.68 |
| **N0063** | large backbone 37.49M (H0091) | **25.893@E08** | +6.25 FAIL — param cap NOT bottleneck |
| **N0065** | scale-embed 20bin 31.33M (H0092) 128ep→81ep | **20.429@E28** | +0.78 FAIL + 10 nan→62 MAE collapse |

**Laws**: H0085 single-slider; H0089 readout premise error; H0081 attn 0.59; H0090 decoder-res closed; H0091 backbone-cap refuted; H0092 scale-embed refuted (no benefit + late instability).

## External SOTA survey
BMNet+ (CVPR22) val 15.74/27M total/13M train — best <32M with code. Cloned /data/repo/bmnet_ref, smoke 12.86M trainable but 32ep trial OOM bs16→4, E00 38.11, killed for N0065. SPDCN needs mmcv, skipped. SAFECount ~12M. Survey done, next step: port BMNet+ ideas (scale-embed already tried → fail; dynamic matcher/contrast loss remain).

## Server gotchas
- git pull hangs → tmux direct; HF /data/asset/hf hf-mirror; scp sync; PATH /data/miniconda/bin; timeout 36h + resume patch code/engine/train.py:195,281-290.

## Queue (next session)
1. N0065 synthesis + N0063 synthesis (Lead-booked, subagent network_error). Ledger 27 hyps (15 tested, 33% correct).
2. Frozen-head LOS hardened — remaining lever is engine tail_reweight (dead code train.py:334) or head redesign (dynamic matcher).

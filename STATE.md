# STATE — Session 2026-08-29 (Lead=qkun-local, User-Guided)

**Mode**: User-Guided (backbone swap per user: "换上更大的backbone，来看看是不是参数量限制"). Direct execution, temply Lead does all roles.
**Preflight**: SERVER_OK (port 44387). Deviation logged: N0063 allows total 37.49M >32M (max_params 60) to isolate backbone capacity — User-Guided override per AGENTS.md §1.

## Champion (frozen tiny regime)
**N0054_xscale_exemplar** (GCA + XScale, dinov3-convnext-tiny) 19.647 / 74.05 / 31.32M LOCKED.

## Closed negative evidence (30ep @384) — 5 nodes, frozen tiny
| Node | Axis | Delta |
|---|---|---|
| N0054 | — | 19.647 CHAMPION |
| N0055 | info-add 2K keys | +1.19 |
| N0056 | info-add extra fine to exemplar agg | +3.06 |
| N0057 | consumer swap | +1.43 |
| N0058 | producer swap part-pool | +3.50 |
| N0059 | producer swap matched | +1.31 |
| N0060 | 2nd coarse MAX | +3.24 |
| N0037 | N·p factorization | +0.5 |
| N0061 | multiplicative count | +3.86 |
| N0062 | decoder native-1/4 h1 | +1.68 |

**Laws**: H0085 single-slider; H0089 readout-multiplier premise error; H0081 attention load-bearing 0.59; H0090 decoder-res closed.

## Active: N0063_large_backbone (H0091) — PARAM-CAP DIAGNOSTIC
**Design**: timm `convnext_small.in12k` frozen (34.0M backbone, dims 192@1/8+384@1/16 same as tiny 27.8M, +6.2M total →37.49M) + identical GCA+XScale head (3.5M). `use_large_backbone=False` bit-identical (forward diff 0.0, smoke PASS 37.49M vs 31.32M). Same /255 input, frozen, 30ep.
**Hypothesis H0091**: larger frozen representation <19.647 if param cap is bottleneck; DISPROVED if ≥19.647.
**Smoke**: large 37.49M OK, small 31.32M identical, shapes (2,1,96,96) finite `model.py:14-27`.
**Status**: code+smoke done, launching 30ep User-Guided deviation run (max_params 60).

## Server gotchas
- run_node.sh git pull hangs: launch tmux directly (PATH=/data/miniconda/bin). HF /data/asset/hf + hf-mirror. Sync via scp.

## Queue
1. N0063 running 30ep — diagnostic for param cap; if FAIL, closes backbone-capacity axis and confirms frozen-head saturation is architecture/head, not size.
2. Ledger 26 hyps, index H0091 run booked.

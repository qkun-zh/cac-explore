# STATE — Session 2026-08-29 (Lead=qkun-local)

**Mode**: Free-Research (autonomous). Directive: combine PoM (2604.06129) + ParTY (2603.09611) modules autonomously, no questions.
**Preflight**: creds ROTATED (port 44387) → install_key.py → SERVER_OK. Server engine has +5-line periodic-ckpt hotfix (save_every, default off).

## Champion (frozen regime, UNCHANGED)
**N0054_xscale_exemplar** (GCA + XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED. AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384.

## Active node: N0060_xscale_max (RUNNING, 30ep)
**H0084 test — the ONLY positive axis.** Add coarse MAX-order-statistic summary (adaptive_max_pool2d over already-aligned 7×7 ROI → xproj→256 → additive onto the SAME fused prototype) beside the retained mean XScale. Producer self-attn + condenser cross-attn + GCA untouched. Red-team corrected H0083: a 2nd global MEAN is redundant (corr 0.85–0.97 with XScale per-channel mean → tie); MAX is an order statistic near-orthogonal under heavy-tailed ConvNeXt, catches peak/foreground signal. Single switch use_xscale_max (False = exact champion bit-identical). Params +98,560 → head 3.60M / total 31.42M.
**H0084** IF coarse MAX-order-statistic summary added additively to fused prototype alongside mean XScale IN frozen champion THEN val MAE < 19.647 BECAUSE max is order-statistic-orthogonal to the mean, supplying peak/foreground signal, without N0055 cardinality or N0056 fine entropy. DISPROVED IF best > 20.0.
**Gates**: CONFIRM <19.45 (2nd seed if first <19.40) · WEAK-KEEP 19.45–20.0 (axis saturated/informative tie, NOT a kill) · FAIL >20.0.
Launched ~19:0x tmux node_N0060_xscale_max (direct, run_node.sh git-pull hangs).

## Frozen-regime NEGATIVE table (30ep @384); N0059 cleared confounds
| Node | Axis | Delta |
|---|---|---|
| N0054 GCA+XScale | — | **19.647 CHAMPION** |
| N0055 XScale-Key | info-add 2K keys | +1.19 |
| N0056 XFine | info-add extra scale | +3.06 |
| N0057 cond-matcher | consumer swap | +1.43 |
| N0058 PMOM | producer swap (part-pool+−42%cap) | +2.17 same-ep / +3.50 floor |
| N0059 PoM-Morph | producer swap (matched, full-token) | +1.31 floor / +2.13 same-ep |

**Refined law**: ALL aggregation axes negative (info-add, consumer swap, producer swap both confounded & matched). Self-attention on exemplar pathway is load-bearing, capacity-independent (H0081 0.5→0.59 via N0059). THE ONE positive axis = XScale-style coarse single-slot ADDITIVE granularity summary (XScale +0.95). N0060 now probes whether a 2nd order-statistic (MAX) on that axis has headroom (CONFIRM) or is saturated (WEAK-KEEP tie).

## Server gotchas
- run_node.sh `git pull` hangs server-side: launch tmux directly (export PATH=/data/miniconda/bin). tmux libtinfo warning non-fatal. python /data/miniconda/envs/cac/bin/python; HF /data/asset/hf + hf-mirror. Sync via scp; `local/` gitignored.

## Queue
1. ✅ N0058 closed, N0059 closed, N0060: idea(finalize+red-team)+novelty(0.498)+code+local smoke+server smoke GREEN+launched.
2. Poll curve vs N0054 same-epoch; apply CONFIRM/WEAK-KEEP/FAIL bands (2nd seed if <19.40).
3. Feedback×4 + Diagnostic + synthesis + calibration + H0084 ledger + tree flip + commit/push.
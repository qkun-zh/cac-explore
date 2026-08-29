# STATE — Session 2026-08-29 (Lead=qkun-local)

**Mode**: Free-Research (autonomous; user directed: "你是自主模式，不必请求我的意见")
**Preflight**: creds ROTATED (port 44387, new host gxkkqyad0izmmwnlsnow.deepln.com) → reran `python3 scripts/install_key.py` → SERVER_OK. git up to date. tmux/python/HF gotchas below unchanged.

## Champion (frozen regime, UNCHANGED)
**N0054_xscale_exemplar** (GCA + XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED DELIVERABLE. Recipe: AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384, MSE+SmoothL1, augment. use_gca=True, use_ddca=False, use_xscale=True, xscale_size=3.

## Today: PoM × ParTY generation (one node executed, NEGATIVE)
Papers: **PoM** (2604.06129) polynomial mixer (moments + gate), **ParTY** (2603.09611) part-guidance/fusion.

**N0058_pompart_exemplar — NEGATIVE, EARLY-STOPPED E15.** PMOM replaced ONLY the exemplar aggregation operator (producer swap): 2×2 part-pool → 2nd-order moments `[h;h²]` → part-gate softmax → moment_proj, single fused prototype preserved, condenser MHA/XScale/GCA untouched. Single-switch smoke-proven (use_pmom=False = exact 31.32M/3.50M). Real run: best **23.151@E13**, E15 23.805 = **+2.17 same-epoch** vs N0054 21.635; floor +3.50. KILL bar ≥20.4 fired E7. H0080 refuted; H0081 "load-bearing attention law" booked (producer OR consumer learned-attention swap ⇒ ≥+1.4; boundary: aggregation swaps only, NOT XScale-type enrichment; falsifier = ≥1.5M-trainable param-matched swap beats 19.647).

## Frozen-regime NEGATIVE table (30ep @384)
| Node | Axis | Delta |
|---|---|---|
| N0054 GCA+XScale | — | **19.647 CHAMPION** |
| N0055 XScale-Key | info-add 2K keys | +1.19 |
| N0056 XFine | info-add extra scale | +3.06 |
| N0057 cond-matcher | consumer swap | +1.43 |
| N0058 PMOM | producer swap | +2.17 same-ep / +3.50 floor |

**Refined lesson**: producer AND consumer learned attention are BOTH load-bearing; info-adds collapse the prototype; the ONE positive axis is XScale (additive, single-slot, pre-attention granularity). Structural head innovation in this regime is now fully mapped — all four quadrants (info-add / producer-swap / consumer-swap / density-bias) tested negative.

## Server gotchas
- tmux at `/data/miniconda/bin/tmux` (libtinfo warning non-fatal) — export PATH in command.
- python `/data/miniconda/envs/cac/bin/python`; HF cache `/data/asset/hf` + `HF_ENDPOINT=https://hf-mirror.com`.
- Sync via `scp` (git pull flaky); `local/` gitignored (119MB best.pth held there, do not commit).

## Queue
1. ✅ Full PoM×ParTY cycle for N0058 closed: novelty→smoke→run→early-stop→feedback×3+Diagnostic→synthesis (calibration table pasted)→ledger (H0080 refute, H0081 book)→tree flip.
2. Remaining: commit & push session close.
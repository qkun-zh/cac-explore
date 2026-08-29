# STATE — Session 2026-08-29 (Lead=qkun-local)

**Mode**: Free-Research (autonomous). Directive: combine PoM (2604.06129) + ParTY (2603.09611) pluggable modules autonomously, no questions.
**Preflight**: creds ROTATED (port 44387) → install_key.py → SERVER_OK. Server engine has +5-line periodic-ckpt hotfix (save_every, default off) — benign source-of-truth.

## Champion (frozen regime, UNCHANGED)
**N0054_xscale_exemplar** (GCA + XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED DELIVERABLE. AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384, MSE+SmoothL1. use_gca=True, use_ddca=False, use_xscale=True, xscale_size=3.

## Today: PoM × ParTY generation — 2 nodes closed (both NEGATIVE)
**N0058_pompart_exemplar** — NEGATIVE, early-stopped E15. Producer swap → PMOM (2×2 part-pool + 2nd-order polynomial moments + part-gate). Confound: −42% capacity AND part-pool info destruction. best 23.151 / E15 23.805 = +2.17 same-ep vs N0054 21.635, floor +3.50. H0080 refuted.

**N0059_pom_morph — NEGATIVE, KILL, 30/30ep success, best 20.958@E18 / final 21.415.** PRE-REGISTERED FALSIFIER of H0081. Removed BOTH N0058 confounds: 2× PoM-PolyMorpher (2604.06129 eq.3: token-averaged 2nd-order moments H + per-token sigmoid gate) over ALL 49 tokens, capacity matched (head 3.522M vs 3.505M, total 31.34M), residual path matched (norm_first), ParTY part-pool EXCLUDED, condenser/GCA/XScale untouched. Single switch use_pom (False = exact N0054). Gates: KILL both prongs (best 20.958≥19.90; early-stop E16 +2.13). **H0082 REFUTED** (20.958≥20.40). **H0081 confirmed-by-failure, strengthened 0.5→0.59** — producer attention load-bearing is capacity-independent. Mechanism: shared moment state + gate lack per-query softmax contrast; D=352≫n=49 rank ceiling; tail overfit-to-scale.

## Frozen-regime NEGATIVE table (30ep @384); N0059 clears confounds
| Node | Axis | Delta |
|---|---|---|
| N0054 GCA+XScale | — | **19.647 CHAMPION** |
| N0055 XScale-Key | info-add 2K keys | +1.19 |
| N0056 XFine | info-add extra scale | +3.06 |
| N0057 cond-matcher | consumer swap | +1.43 |
| N0058 PMOM | producer swap (part-pool+−42%cap) | +2.17 same-ep / +3.50 floor |
| N0059 PoM-Morph | producer swap (matched, full-token) | +1.31 floor / +2.13 same-ep |

**Refined law**: ALL aggregation axes tested negative — info-add (N0055/56), consumer swap (N0057), producer swap both confounded (N0058) and matched-capacity (N0059). Self-attention on the exemplar pathway (producer AND consumer) is genuinely load-bearing, capacity-independent. The ONE positive axis is XScale-style single-slot coarse GRANULARITY enrichment — next step H0083 (a 2nd single-slot coarse GAP summary on the champion prototype).

## Server gotchas
- run_node.sh `git pull` hangs server-side: launch tmux directly (export PATH=/data/miniconda/bin).
- tmux libtinfo warning non-fatal. python /data/miniconda/envs/cac/bin/python; HF /data/asset/hf + hf-mirror.
- Sync via scp; `local/` gitignored (best.pth 119MB in local/feedback_src_N0059/).

## Queue
1. ✅ N0058 closed (NEGATIVE early-stop) + N0059 full cycle closed: novelty→code→smoke→run(30ep)→KILL→feedback×4→synthesis→calibration→ledger (H0082 refute, H0081 support, H0083 book)→tree flip.
2. Next generation per refined law: H0083 (2nd XScale-style coarse single-slot additive summary on champion) — the only positive axis. Do NOT re-attempt aggregation/operator swaps or info-adds.
3. Remaining: commit & push session close.
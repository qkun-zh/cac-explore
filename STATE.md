# STATE — Session 2026-08-30 OPEN (Lead=qkun-local, User-Guided)

**Mode**: User-Guided. Preflight SERVER_OK (fz58r...:42258, RTX3060 12G, tmux apt-installed). User directive 2026-08-30: backbone mid-layer FT allowed, probe intermediate vs final output.
**Regime**: PARTIAL-FT — stages 1,2 (hs 2,3) may FT @0.1× head LR; final vs intermediate comparison via hs_map. AGENTS.md + research_direction.md updated, 31 dead files + 594M weights removed.

## Champion (frozen baseline)
**N0054_xscale_exemplar** GCA+XScale 19.647/74.05/31.32M. AdamW head 1e-3 / backbone 1e-4, wd0.05, cosine, bs16, AMP, 30ep.

## 2×2 heuristic map — CANONICAL (corrects earlier N0066/N0067 design)
Rows=readout layer, Cols=backbone state. FROZEN intermediate = N0054 champion.
| | hs(2,3) intermediate | hs(3,4) final |
|---|---|---|
| **FROZEN** | **N0054 = 19.65** ✓ | **N0068_frozen_final** (queued) |
| **FT @1e-4** | N0066 = 28.36 FAIL H0093 | N0067 (running, E011 28.30) |

KEY: N0054 ALREADY reads intermediate hs(2,3) + frozen — "direct intermediate use" is the champion. N0066 conflated readout-layer w/ unfreeze (mis-attribution); N0068/N0067 finish the table.

## Running / Queued (30ep @384, GCA+XScale)
| Node | hs_map | dims | tune | Params | Status |
|---|---|---|---|---|---|
| **N0067_midft_final** | (3,4) | 384,768 | [2,3] @1e-4 | 31.60M | **RUNNING** (E011 28.30; freestanding proc, tmux wrapper died on libtinfo) |
| **N0068_frozen_final** | (3,4) | 384,768 | [] frozen | 31.60M | **QUEUED** (smoke GREEN; setsid launcher auto-launch on N0067 terminal) |

**Hypotheses**: H0093 mid-FT intermediate beats frozen → REFUTED (28.36 vs 19.65); H0094 final worse by ≥0.5 (holds iff N0067/N0068 ≥ intermediate+0.5); H0095 frozen-final layer control (N0068 vs N0054: CONFIRM<19.45 / TIE 19.45–20.15 / FAIL>20.15).

## Cleanup 2026-08-30
- Deleted: cac_si/*, code/counting/*, docs/proto4dme/DISTILLED/arXiv/research_notes, scripts/axiom/coin/oir/verify/smoke/train_god/hf_prepare/check_data (31 files, commit 238794d, push main).
- Weights: local/feedback_src_N*/best.pth 594M removed; server /data/runs/* archived to archive_2026-08-30 (1.9G), N0054 retained, smoke checkpoints cleared, cac_uot dir removed.
- Probe via tmux: hs 1:96/2:192/3:384/4:768, stages 0:3/1:3/2:9/3:3; FineFuser size-adaptive to 96 (N0067/N0068 base).

## Server gotchas (update)
- New host fz58rq9zeriulqjksnow.deepln.com:42258 (was gxkkqy...44387 timeout), tmux installed via apt-get update.
- HF_HUB_OFFLINE=1 for probes; /data/asset/hf cache holds dinov3-convnext-tiny.
- Run dir hygiene: /data/runs holds only active lineage (N0054); stale moved to archive_2026-08-30.

## Next
1. Collect N0067 (FT final) + N0068 (frozen final), feedback ×3+synthesis, book H0093 (booked refuted)/H0094/H0095/H0096, update tree/STATE, commit.
2. Read the full 2×2 diagonal: FROZEN row (N0054 vs N0068) isolates layer-quality cleanly; FT row (N0066 vs N0067) isolates unfreeze harm per-layer. If frozen-final ties/beats, final layer is the better readout and mid-FT N0066's failure = unfreeze harm; if frozen-final also worse, intermediate features are load-bearing.
3. tmux wrapper is unreliable (libtinfo kills server); use freestanding run_node.sh + setsid launcher. Never kill-tmux then new-session in separate calls.

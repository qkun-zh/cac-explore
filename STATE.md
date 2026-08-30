# STATE — Session 2026-08-30 OPEN (Lead=qkun-local, User-Guided)

**Mode**: User-Guided. Preflight SERVER_OK (fz58r...:42258, RTX3060 12G, tmux apt-installed). User directive 2026-08-30: backbone mid-layer FT allowed, probe intermediate vs final output.
**Regime**: PARTIAL-FT — stages 1,2 (hs 2,3) may FT @0.1× head LR; final vs intermediate comparison via hs_map. AGENTS.md + research_direction.md updated, 31 dead files + 594M weights removed.

## Champion (frozen baseline)
**N0054_xscale_exemplar** GCA+XScale 19.647/74.05/31.32M. AdamW head 1e-3 / backbone 1e-4, wd0.05, cosine, bs16, AMP, 30ep.

## Running / Queued (30ep @384, GCA+XScale)
| Node | hs_map | dims | tune | Params | Status |
|---|---|---|---|---|---|
| **N0066_midft_intermediate** | (2,3) | 192,384 | [1,2] @1e-4 | 31.32M | **RUNNING** tmux node_N0066 (launched 10:20, 135% CPU, 4.8G) |
| **N0067_midft_final** | (3,4) | 384,768 | [2,3] @1e-4 | 31.60M | **QUEUED** tmux queue watcher (auto-launch after N0066) |

**Hypotheses**: H0093 mid-FT intermediate <19.65 beats frozen; H0094 final worse than intermediate by ≥0.5 (DISPROVED IF final < intermediate+0.5). Both smoke PASS via tmux (N0066 47s, N0067 45s).

## Cleanup 2026-08-30
- Deleted: cac_si/*, code/counting/*, docs/proto4dme/DISTILLED/arXiv/research_notes, scripts/axiom/coin/oir/verify/smoke/train_god/hf_prepare/check_data (31 files, commit 238794d, push main).
- Weights: local/feedback_src_N*/best.pth 594M removed; server /data/runs/* archived to archive_2026-08-30 (1.9G), N0054 retained, smoke checkpoints cleared, cac_uot dir removed.
- Probe via tmux: hs 1:96/2:192/3:384/4:768, stages 0:3/1:3/2:9/3:3; FineFuser now size-adaptive to 96.

## Server gotchas (update)
- New host fz58rq9zeriulqjksnow.deepln.com:42258 (was gxkkqy...44387 timeout), tmux installed via apt-get update.
- HF_HUB_OFFLINE=1 for probes; /data/asset/hf cache holds dinov3-convnext-tiny.
- Run dir hygiene: /data/runs holds only active lineage (N0054); stale moved to archive_2026-08-30.

## Next
1. Collect N0066/N0067, feedback ×3+synthesis, book H0093/H0094, update tree/STATE, commit.
2. If intermediate FT wins, explore narrower/wider FT (stage2 only, stage1+2+3) and LR sweep 0.05×/0.2×; if final wins, revisit head adapter capacity.

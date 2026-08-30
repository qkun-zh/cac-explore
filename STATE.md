# STATE — Session 2026-08-30 OPEN (Lead=qkun-local, User-Guided)

**Mode**: User-Guided. Preflight SERVER_OK (fz58r...:42258, RTX3060 12G, tmux apt-installed). User directive 2026-08-30: backbone mid-layer FT allowed, probe intermediate vs final output.
**Regime**: PARTIAL-FT — stages 1,2 (hs 2,3) may FT @0.1× head LR; final vs intermediate comparison via hs_map. AGENTS.md + research_direction.md updated, 31 dead files + 594M weights removed.

## Champion (frozen baseline)
**N0054_xscale_exemplar** GCA+XScale 19.647/74.05/31.32M. AdamW head 1e-3 / backbone 1e-4, wd0.05, cosine, bs16, AMP, 30ep.

## 2×2 heuristic map — COMPLETE (all 4 cells run)
Rows=readout layer, Cols=backbone state.
| | hs(2,3) intermediate | hs(3,4) final |
|---|---|---|
| **FROZEN** | **N0054 = 19.65** ✓ | N0068 = 26.87 (+5.17@E22) FAIL |
| **FT @1e-4** | N0066 = 28.36 (+6.02) FAIL | N0067 = 25.94 (+4.83) FAIL |

**TWO LAWS (decisive, all 4 cells):**
1. **Intermediate hs(2,3) is the load-bearing count-semantic readout.** Frozen-row: mid 19.65 vs final 26.87, intermediate wins ~7 pts. Final layer (1/32 res, 768ch) loses fine spatial count info.
2. **Unfreezing backbone is net-harmful regardless of layer.** Frozen-intermediate (19.65) beats every FT config (25.9–28.4) by 6–9 pts.

"Direct intermediate use" was ALREADY the champion N0054; N0066 conflated readout-layer w/ unfreeze (mis-attribution); N0068/N0067 finished the table. H0093/H0094/H0095 all REFUTED. Backbone layer/FT axis FULLY MAPPED & NEGATIVE vs champion.

## Cleanup 2026-08-30
- Deleted: cac_si/*, code/counting/*, docs/proto4dme/DISTILLED/arXiv/research_notes, scripts/axiom/coin/oir/verify/smoke/train_god/hf_prepare/check_data (31 files, commit 238794d, push main).
- Weights: local/feedback_src_N*/best.pth 594M removed; server /data/runs/* archived to archive_2026-08-30 (1.9G), N0054 retained, smoke checkpoints cleared, cac_uot dir removed.
- Probe via tmux: hs 1:96/2:192/3:384/4:768, stages 0:3/1:3/2:9/3:3; FineFuser size-adaptive to 96 (N0067/N0068 base).

## Server gotchas (update)
- New host fz58rq9zeriulqjksnow.deepln.com:42258 (was gxkkqy...44387 timeout), tmux installed via apt-get update.
- HF_HUB_OFFLINE=1 for probes; /data/asset/hf cache holds dinov3-convnext-tiny.
- Run dir hygiene: /data/runs holds only active lineage (N0054); stale moved to archive_2026-08-30.

## Next
1. Feedback ×3 + synthesis over N0066/N0067/N0068; book H0093 (refuted)/H0094/H0095; calibration bin; update tree/STATE; commit.
2. Backbone axis is closed (all-negative). Remaining levers: head-side innovation (new pluggable component on N0054 interface) or lock 19.647 as deliverable. Suggest dispatching parallel idea agents (pure-math/lineage/counter-intuitive) on head-side only.
3. tmux wrapper unreliable (libtinfo kills server); run_node.sh's freestanding train proc survives — use that. pkill -f self-kills the ssh (pattern in cmdline); match narrowly or re-ssh.

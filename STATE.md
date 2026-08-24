# STATE — Session 2026-08-24 (evening, Lead=qkun-local)

**Champion ckpt**: N0021_dino_partialft val MAE **20.438** @23.11M (`/data/runs/N0021_dino_partialft/best.pth` server-side ckpt was smoke-overwritten; retrain scheduled)
**Effective best**: routed readout (N̂@392≥200→518) **MAE 19.18 / RMSE 66.37** split-half honest
**GPU**: RTX3060 idle · creds `local/address_and_password.md`

## Read `docs/DISTILLED_2026-08-24.md` first (levers/facts), then GOD v6 §1 facts + §4 tombstones.

## What happened this block (all CPU/GPU probes under /data/asset/r0i_probe/, outside repo)
- **Seed nodes both duds**: N0029_loghead FAIL (incomplete E33/40, best 20.753 > champion); N0030_sizenorm degenerate execution (9.3s, mae==rmse==151.95) invalid. H0043/H0044 unresolved, not refuted.
- **N0028 zombie cleaned**: died E3 silently long ago; tree.json honesty pass done.
- **R0-I localization suite** (my runs):
  - global-stats readout on frozen feats: 63.4 MAE → scalarization trap reconfirmed
  - linear spatial cell probe (sqrt target): **47.1** — no-exemplar linear ceiling ≫ champion
  - D1: token-norm anti-correlates with local density (−0.205 cell / −0.564 image) — norm≠info
  - **frozen-backbone + fresh champion head stack = 22.431** vs partial-FT 20.403 ⟹ **backbone FT worth ~2.0 MAE**; features not information-starved at low-20s
- **CPU error-law verification** (verify_channel_floor.py): two-regime law `|err| ≈ 2 + 0.69·(N−50)₊` confirmed; errors are systematic gain BIAS (94% undercount tail), not variance; **resolution U-shape discovered** — bulk worst at 518 (10.6 vs 3.3@392), only dense-side routing safe.
- GOD v6 rewritten: tombstones updated (monotone-resolution & cross-res-consistency routing dead), §8 channel-floor program v2 with DERIVED/FITTED/ASSUMED labels.

## Implications (the one paragraph that matters)
Readout re-parameterization around the current recipe is a ±0.5 noise game (N0029 concurs). Backbone FT buys ~2.0. Everything points the same way: to move toward ≤10 you must ADD information — dense-side bandwidth, visibility/amodal evidence, exemplar-conditioned mass calibration — not reshuffle the readout. Bulk floor ≈2 counts is already at channel level.

## Next queue
1. **R0-I3 synthetic floor** (CPU-heavy, ~30min): only remaining route to quantify the TAIL information limit (existing sweep can't extrapolate).
2. **O1 offline conditional-gain fit** (server CPU minutes): split-half `g(box_area, N_hat)` on existing dumps — prices how much of the 0.69 slope is reclaimable without training.
3. If pursuing bulk: backbone distillation/compression study (Tier-A gap says features have ~2× headroom over raw-pixel baseline of similar MAE 64 — both weak; real ceiling needs Tier-A with nonlinear probe).
4. Re-train N0021 ckpt on server (smoke-overwritten) before any deployment-grade eval.

## Rules refreshed
- Probe scripts live in /data/asset/r0i_probe (server) + /tmp/opencode (local); repo untouched by probes.
- Gates unchanged: novelty_check → check_hypothesis → calibration_report.

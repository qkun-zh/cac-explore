# Synthesis — N0021_dino_partialft (NEW CHAMPION: best MAE 20.438@E25, 23.11M, 1441s)

## Verdict on H0032 (partial FT blocks 10-11 @ lr×0.1 beats frozen inference features)
SUPPORTED, moderate. No prior H0032 entry existed → booking is create+evidence. Math (η=0.20):

| Step | Rule | Confidence |
|---|---|---|
| create H0032 | init c₀ | 0.500 |
| supports w=0.75 (this node) | c←c+η·w·(1−c) = 0.5+0.2·0.75·(1−0.5) | **0.575** |

Final: **0.5750** → `uncertain` (needs >0.75 to confirm). Strength capped at 0.75 because the
pre-registered stretch bar (≤18) was NOT met and RMSE calibration regressed (below).
Full-FT refutation NOT re-booked — already booked once under H0030 (2026-08-23T16:12 line);
no duplicate appended.

## Feedback consolidation — contradictions resolved
1. **Dropout confound between siblings** (fullft cfg 0.15 vs champion 0.1): rejected as
   material cause — fullft diverged from E1 (MAE 48.71, loss 46.29 before cosine decay), a
   regularizer bump cannot produce from-epoch-1 collapse 2.3× worse than parent; mechanism
   (unfrozen-depth drift compounding) dominates. Residual confound stands; any full-FT retry
   must equalize dropout=0.1.
2. **Single-seed caveat**: run σ≈±0.3 MAE vs −1.09 gain → direction robust, magnitude noisy
   (reflected in w=0.75).
3. **RMSE ratio ambiguity**: headline 83.06/20.44=4.06 mixes final-RMSE with best-MAE.
   Like-for-like still worse than parent: 3.88 final / 3.91 @best vs 3.63/3.53 — real tail
   drift (train loss 3.78→1.45 while val RMSE 79.8→83.1 over E26–40). MAE win ≠ free.
4. **H0030 ID collision** (fine-tuning evidence sitting on a point-detection hypothesis):
   resolved per quantitative.md recommendation — FT claims now under fresh H0032; H0030 keeps
   paradigm-only evidence.
5. **Sibling budget** (fullft killed @E10/30): not "needed more epochs" — E1–E10 plateau at
   loss ~35 with dead Δ=0.1 over E7–10 is a moving-target equilibrium, not slow convergence;
   replicated by N0022 EBC+fullft (best 21.708, final 27.40). Full FT refuted across two heads.
6. **Epoch waste**: best @E25/40, zero gain last 15 ep (~51% wall-clock) → 28ep cap + patience 5.

## What transfers (building blocks for future nodes)
- Differential-LR `param_groups()` split (backbone trainable @ base_lr×mult vs rest @ base_lr,
  model.py:54-66); frozen params correctly excluded from decay groups.
- Freeze-mask pattern: prefix match on `blocks.10./blocks.11./norm.` (model.py:39-43) works
  for ViT-S/14 timm features_only; fragile if depth >99 — use index-based mask for deeper nets.
- Timescale-separation principle: stable ⇔ backbone drift-rate < head fit-rate; scope×lr are
  JOINTLY load-bearing (sibling shared mult=0.1 yet collapsed → scope binds too).
- New base substrate: champion recipe + partial FT (23.11M total, 3.6M trainable backbone top).
- Checkpoint selection by val RMSE-constrained MAE (best MAE among epochs with RMSE ≤ running
  min +10% → picks E18 20.47/72.89): free −7 RMSE, zero training cost — implement in eval harness.
- Log per-image worst-20 val errors (high/low-count split) before choosing tail fixes.

## Next steps (ranked; per docs/research_notes_2026-08-24.md)
1. **TT-Norm at eval** — CounTR/CountGD ablation val 8.69→7.99, test 10.92→9.62; zero params,
   targets our binding constraint (tail/RMSE). Fastest path toward <19–20.
2. **lr_mult sweep {0.05, 0.2}** on blocks 10-11 (causal.md §3 pre-registered predictions:
   0.05→gain vanishes ⇒ dose-dependent; 0.2 beats cleanly ⇒ 0.1 not special; MGCAC uses ×0.1
   de facto). Decisive for whether to tune further.
3. **Blocks 9-11 @0.1** (graceful scaling predicts 20.0–20.6; CLIP-FT lit: top-half ≈ full FT).
4. **EMA or L2-anchor-to-init** stabilizers (ProLIP: LR-insensitivity; CLIP-FT: EMA +0.3–0.9%)
   — directly counters observed late tail-drift without touching scope/lr.
5. **Jitter-only light aug** merged into partial-FT recipe (mosaic hurts FSC147 in-domain per
   MGCAC/"Recipe for CAC"; weak aug preferred on pretrained backbones per CLIP-FT lit).
6. Avoid: mosaic, diffusion aug, Huber/tail-reweight (refuted levers), bare full-FT retries.

## Bookkeeping done this synthesis
- memory/hypotheses.jsonl APPENDED: create H0032 (c₀=0.5) + evidence supports w=0.75 → c=0.575.
- scripts/rebuild_index.py rerun (3 hypotheses); select_next verified quality 0.880 /
  avail 0.667 / score 0.795 for this node → tree.json fields filled (0.8799/0.6667/0.7946).
- Flag: tree.json `tested_hypotheses:["H0030"]` looks stale (this node tested H0032); left
  untouched per instructions — Lead may correct (would shift avail 0.667→0.333 here).

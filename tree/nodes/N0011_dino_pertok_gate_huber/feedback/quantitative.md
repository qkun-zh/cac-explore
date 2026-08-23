# feedback/quantitative.md — N0011_dino_pertok_gate_huber

## reasoning
Parent N0010 best MAE 21.531 (params 23.11M, 40ep, scalar gate, MSE) → N0011 best 26.678 / final 26.895, RMSE 94.34, params 23.16M (model.py:30-42 PerTokenGateMLP 768→64→2, ~50K extra), config.py:14-16 Huber delta=5.0, count-w 0.3. Δ=+5.15 (worse, 24% regression), not +1.0 gain.
Bars: H0019 required ≤20.5 (21.531−1.0); observed 26.68 → miss by 6.18. H0020 required RMSE/MAE <3.0 (disproved if >3.5); observed 94.34/26.68=3.53 (final 3.51) vs parent 3.63 (N0010: 81.97/22.61=3.63, best 3.53). Marginal −0.10, still >3.5 and far from <3.0.
Train loss (Huber scale) 9.27→3.43 over E01–E40, monotonic decrease; val MAE plateau 27–28.3 (E13–E34: 28.33→26.68) then flat 26.7–27.2 to E40 — no divergence, no OOM/instability. Huber is stable, so optimization confound ruled out.
Over-parameterized per-token gate is the mechanism: 784 locations × independent softmax gating adds variance without signal on 392px/DINOv2-S tokens; MSE→Huber not causal for MAE rise.

## actionable_feedback
- Revert per-token MLP to scalar layer gate (2 params); per-token variant is a regression, not just neutral.
- Do not reuse Huber delta=5.0 for outlier tail — tail unchanged; try count-aware reweighting or high-density oversampling instead.
- Isolate future ablations: test gating and loss changes separately; current node confounds both.

## hypothesis_updates
- H0019: contradicts, strength 0.90, reasoning: per-token gate → MAE 26.68 vs bar ≤20.5 (and vs parent +5.15), directionally opposite to ≥1.0 improvement; no epoch approached 21.5.
- H0020: contradicts, strength 0.80, reasoning: RMSE/MAE 3.53 >3.5 disproof threshold and >3.0 target; Huber capped gradients but did not move tail (stable training confirms not an optimization artifact).

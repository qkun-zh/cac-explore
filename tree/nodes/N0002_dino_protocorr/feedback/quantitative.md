# feedback/quantitative.md — N0002_dino_protocorr

## reasoning
Val MAE 42.05 / RMSE 122.06 in 317s (10ep, 22.17M) improves only ∆4.64 (9.9%) over S0001 (46.69 @0.01M, 2ep). H0001 predicted <30 — missed by 12 points. Steady descent (E1 48.5→E10 42.05, loss 13.96→7.76) with no plateau; best at final epoch so 10ep was too short vs τ_max 1800s (only 18% time used). RMSE >> MAE (2.9×) signals heavy-tail errors on high-count images. Synthetic smoke MAE 121 hints decoder scale mis-calibration at init, but real training recovered partly. Params well under 32M; throughput ≈31s/ep suggests ~50ep fits in budget.

## actionable_feedback
- Re-run same node with epochs=30-40 to fully use τ_max; expect ~38–40 MAE if trend continues.
- For children, raise `loss_count_weight` (0.3→1.0) or add normalized count loss to tame RMSE tail.
- Keep DINOv2-S reg4 (proven fast, 40–60 img/s) but pair with stronger head; do NOT shrink backbone.

## hypothesis_updates
- H0001: contradicts, strength 0.75. DISPROVED IF ≥30 — observed 42.05 @10ep, so frozen single-prototype cosine matching alone does not reach <30. Reasoning: head too shallow to calibrate similarity into absolute density; needs scale/context.
- H0002: neutral, strength 0.0. Learnable τ present (softplus) but no ablation vs fixed τ, so not tested.
- H0003: neutral, strength 0.0. Aux count head not implemented — no evidence.
- H0011: neutral, strength 0.0. Scale/magnitude embeddings not implemented — untested.

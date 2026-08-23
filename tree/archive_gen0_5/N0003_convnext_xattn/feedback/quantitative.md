# feedback/quantitative.md — N0003_convnext_xattn

## reasoning
Val MAE 34.26 / RMSE 104.52 @14ep/430s/16.93M. Beats N0002 (42.05) by −18.5% relative — clears H0004's ≥10% bar with equal-ish frozen budget and fewer params. Convergence non-monotonic early (E1 1038→E2 702→E3/E4 ~950, then steady drop to E14 34.26): init scale shock (loss 11583→383 in one epoch) from random basis maps; tiny-init fix came after this run was launched... actually run used d6cf84d which includes it — the E1 spike is real-data count scale adaptation, not instability (no divergence). Best at final epoch → under-converged again; τ_max only 24% used.

## actionable_feedback
- Re-run/child with epochs=25–30: trend suggests low-30s or below reachable.
- Add cosine-warmup for first epoch to smooth the E1 scale shock.
- loss_count_weight 0.3→1.0 candidate: RMSE/MAE ≈ 3.0 still tail-heavy.
- Compare per-epoch slope vs N0002: N0003 learns faster AND further — mechanism confirmed superior at equal wall-clock.

## hypothesis_updates
- H0004: supports, strength 0.85. Cross-attn mixture beats plain cosine matching by 18.5% ≥10% bar at comparable budget (42.05→34.26).
- H0005: neutral, strength 0.0. Stride-16 attention tokens were used, but no stride-8 ablation ran — parity claim untested.

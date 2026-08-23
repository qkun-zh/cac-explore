# feedback/quantitative.md — N0005_swin_promptseg

## reasoning
Best val MAE 32.66 / RMSE 102.13 @20ep/271s, 28.22M — BEST node so far (N0003 34.26, N0002 42.05). Fastest convergence: E4 already 35.0, E13 best; only 15% of τ_max used. H0008 (<40 @10ep) passed with margin at E9 (34.2). H0009 (implicit ≥10% worse than explicit) DISPROVED: implicit beat the closest-budget explicit node N0002 by 22% relative. Mild drift up after E13 (33.1) while train loss kept falling → slight overfit of adapter/head; cosine schedule may have helped had training continued.

## actionable_feedback
- Children: keep prompt-token conditioning as a core mechanism; add scale/magnitude input to prompt encoder (H0011 natural fit HERE — Fourier encoder already takes box coords, adding area is trivial).
- epochs=30 with eta_min lower to exploit unused time budget.
- Test H0010 properly by implementing the normalized-density KL term in a child (needs engine loss extension or model-internal renorm trick).

## hypothesis_updates
- H0008: supports, strength 0.90. 32.66 < 40 well within 10ep budget.
- H0009: contradicts, strength 0.85. Implicit WON by 22% vs comparable-budget explicit (N0002); conditioning mechanism ranking reversed.
- H0010: neutral, strength 0.0. Seg-KL loss not implemented (engine MSE used) — untested.

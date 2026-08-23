# feedback/causal.md — N0007_dino_promptv2

## reasoning
Causal confirmation: swapping substrate alone (all else equal-ish) delivered -15.3%, the largest single-lever gain of the whole search — validating the features>mechanism>schedule ordering from N0004/N0006 synthesis. Mechanism now: DINOv2's self-distilled patch embeddings cluster SAME-INSTANCE appearances across categories, so the adapter+head only needs to recalibrate mass, not discover correspondence. Remaining gap to <16 is dominated by (a) count-scale calibration on high-count images (RMSE 3.4x MAE), (b) train-category overfit onset, (c) absence of exemplar-guided VERIFICATION (LOCA-style iterative refinement untested).

## actionable_feedback
- Gen-2 priority A: DINOv2 multi-layer taps + cross-attn basis decoder (merges N0003+N0007 winners).
- Gen-2 priority B: explicit count-calibration path (aux-count normalizer; H0003-flavored) aimed at RMSE tail.
- Keep dropout 0.1; consider 0.15 if train-val gap widens further.

## hypothesis_updates
- H0014: supports, strength 0.85.

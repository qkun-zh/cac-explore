# feedback/causal.md — N0009_dino_xattn_stable

## reasoning
Clean adjudication: stable training removed the confound and the merge still lost (-11% vs parent). With features this strong, decoder context modeling is redundant; residual errors are count-scale calibration (RMSE ~3.1x MAE) and train-category overfit — neither addressed by attention. Head exploration is complete: implicit area-prompt + MLP head is the champion on frozen DINOv2-S.

## actionable_feedback
- Gen-3 scales the winner: 40ep, multi-layer taps (mid+final block), higher resolution 448, loss_count_weight >=1.0.

## hypothesis_updates
- H0016: contradicts, strength 0.75. H0015: contradicts, strength 0.55 (adjudicated).

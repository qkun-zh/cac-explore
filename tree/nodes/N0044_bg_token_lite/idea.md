# Idea — N0044_bg_token_lite (parent: N0036_gca_ddca, frozen)
Lightweight background token only (no mutual), inspired by MAFEA TBD loss. Add 1 bg token to exemplar set, zero-init, let Condenser learn to ignore background.

## Change (structural, FROZEN, optimizer unchanged)
**BG token (H0064, +0.01M).** Learnable bg token concatenated to e, participates in Condenser MHA. GCA+DDCA kept. Minimal risk.

## Hypothesis
**H0064** IF bg token THEN MAE ≤20.0 BECAUSE background decoupling. DISPROVED IF ≥20.5.

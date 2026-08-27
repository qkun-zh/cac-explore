# Idea — N0040_film_recal (parent: N0036_gca_ddca, frozen)
Cross-scale FILM recalibration: exemplar embedding modulates h2/h3 features via scale-wise gamma/beta before FineFuser.

## Change (structural, FROZEN, optimizer unchanged)
**FILM recalibration (H0060, +0.15M).** e_mean → gamma_i,beta_i for each h_i. Applied before FineFuser. Same GCA+DDCA backbone.

## Hypothesis
**H0060** IF FILM recalibration THEN MAE ≤19.6 BECAUSE exemplar adapts feature scales per-instance. DISPROVED IF ≥20.5.

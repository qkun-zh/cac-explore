# Idea — N0048_no_gca (parent: N0036_gca_ddca, frozen)
Ablation: remove GCA, keep only DDCA to test if GCA is actually helping. If DDCA alone matches, GCA is noise.

## Change (structural, FROZEN, optimizer unchanged)
**Remove GCA (H0068, -0.02M).** Keep FineFuserDDCA, no global count aux.

## Hypothesis
**H0068** IF no GCA THEN MAE ≈20.5 BECAUSE GCA 0.02 bias is negligible. DISPROVED IF ≥21.5.

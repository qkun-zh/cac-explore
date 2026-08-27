# Idea — N0045_iter_exemplar (parent: N0036_gca_ddca, frozen)
Iterative exemplar feature learning (PLOS One 2025). Refine exemplar prototypes by attending to similar objects in query, 2 iterations.

## Change (structural, FROZEN, optimizer unchanged)
**Iterative EFL (H0065, +0.30M).** After initial e, do 2× cross-attn with fine features to enrich class prototypes, then Condenser. GCA+DDCA kept.

## Hypothesis
**H0065** IF iterative EFL THEN MAE ≤19.8 BECAUSE enriched prototypes generalize. DISPROVED IF ≥20.5.

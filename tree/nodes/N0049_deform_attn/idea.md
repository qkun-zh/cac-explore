# Idea — N0049_deform_attn (parent: N0036_gca_ddca, frozen)
GECO2 deformable attention for dense queries. Replace Condenser MHA with deformable attention (3x3 offsets) for spatial efficiency.

## Change (structural, FROZEN, optimizer unchanged)
**Deformable attn (H0069, +0.30M).** fine tokens as queries, e as values, deformable offsets learned via linear, 4 sampling points.

## Hypothesis
**H0069** IF deformable THEN MAE ≤19.9 BECAUSE sparse sampling handles scale variance. DISPROVED IF ≥20.5.

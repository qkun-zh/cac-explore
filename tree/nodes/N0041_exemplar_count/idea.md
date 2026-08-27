# Idea — N0041_exemplar_count (parent: N0026_res_sweep, frozen)
Radical paradigm shift: instead of density map regression, predict per-exemplar counts and sum. Each exemplar gets a learned attention ROI → count head → total = sum. Density map still predicted for loss, but count is the primary output.

## Change (structural, FROZEN, optimizer unchanged)
**Exemplar-count head (H0061, +0.3M).** For each exemplar: attention over fine features → 128→64→1 count. Sum all K counts = total. Density map secondary. GCA still active.

## Hypothesis
**H0061** IF exemplar-count THEN MAE ≤19.5 BECAUSE per-exemplar counting is more identifiable. DISPROVED IF ≥20.5.

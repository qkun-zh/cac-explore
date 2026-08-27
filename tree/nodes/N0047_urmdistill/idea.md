# Idea — N0047_urmdistill (parent: N0036_gca_ddca, frozen)
URM-inspired CLIP distillation: add frozen CLIP image encoder branch, distill universal prototypes into exemplar prototypes via cosine loss (no extra inference cost).

## Change (structural, FROZEN, optimizer unchanged)
**CLIP distill (H0067, +0.15M).** e → proj → cosine vs CLIP text embedding (class name) with 0.1 weight aux loss. GCA+DDCA kept.

## Hypothesis
**H0067** IF CLIP distill THEN MAE ≤19.7 BECAUSE universal prototypes generalize to unseen classes. DISPROVED IF ≥20.5.

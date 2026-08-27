# Idea — N0046_scale_prior (parent: N0036_gca_ddca, frozen)
Scale-prior deformable convolution (BMVC 2022) + CREAM adaptive density. Use exemplar size to generate scale-aware Gaussian sigma per location.

## Change (structural, FROZEN, optimizer unchanged)
**Scale-prior (H0066, +0.20M).** e wh → scale embedding → modulates decoder's dilated conv dilation rate via FiLM-like gating. GCA+DDCA kept.

## Hypothesis
**H0066** IF scale-prior THEN MAE ≤19.8 BECAUSE adaptive receptive field for scale variance. DISPROVED IF ≥20.5.

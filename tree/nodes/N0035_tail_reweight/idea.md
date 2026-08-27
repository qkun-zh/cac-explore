# Idea — N0035_refine_frozen (parent: N0026_res_sweep, frozen backbone)
Champion: frozen ConvNeXt-Tiny 384 (TEST 18.33). N0033 SALF+MoE and N0034 SHA both failed (MAE 24-25, instability) due to 4-stage complexity + router collapse under 40ep budget. Revert to stable 2-stage FineFuser but add lightweight structural refinement.

## Change vs parent (1 structural, FROZEN, optimizer unchanged)
**Density refinement residual (H0052, +0.25M).** Keep champion FineFuser(h2/h3)+Condenser+DensityDecoder0 identical, same optimizer (AdamW 1e-3, wd 0.05, cosine, 40ep, bs16, augment). Add second stage:
- dens0 = Decoder0(fine+cond) [96x96]
- ref_in = concat[fine (128ch), dens0 (1ch)] -> RefineBlock: Conv3x3 129->64 GN+GELU -> Conv3x3 64->32 GN+GELU -> Conv1x1 32->1 -> residual dens = softplus(dens0 + refine)
Addresses: single decoder must solve both coarse count and fine localization; residual refine lets second stage correct systematic bias (especially dense clusters) without touching backbone. Param total ~28.95M <32M, step0 ~ dens0 (refine zero-init).

## Hypothesis
**H0052** IF residual density refine (129->1) IN frozen 2-stage THEN val MAE ≤19.3 AND tail RMSE improves vs parent BECAUSE second stage corrects quantization/cluster errors. DISPROVED IF MAE ≥20.1 or refine output variance <1e-5 (no effect).

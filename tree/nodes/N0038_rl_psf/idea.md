# Idea — N0038_rl_psf (parent: N0026_res_sweep, frozen)
Physics lens: Poisson deconvolution. Keep FineFuser 2-stage, add flux-conserving 5x5 PSF + one-step RL unfold.

## Change (structural, FROZEN, optimizer unchanged)
**RL-PSF (H0055, +0.01M).** After Decoder0, learnable 5x5 k=softmax(params) sum1, unfold: dens = dens0 * (1+0.2*tanh(Conv3x3(fine))) convolved with k, flux conserving. Same optimizer 1e-3.

## Hypothesis
**H0055** IF RL PSF THEN MAE ≤19.5 BECAUSE inverts optics blur. DISPROVED IF ≥20.1.

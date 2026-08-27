# Idea — N0043_dynamic_kernel (parent: N0036_gca_ddca, frozen)
Replace Condenser MHA with Support-Conditioned Dynamic Convolution (SCDC, ICASSP2026) + keep GCA+DDCA. Exemplar generates depthwise kernel.

## Change (structural, FROZEN, optimizer unchanged)
**Dynamic kernel (H0063, +0.22M).** e_mean → 3x3 depthwise kernel (128ch) via Linear 256→128*9, softmax per channel. Apply to fine. Condenser removed.

## Hypothesis
**H0063** IF dynamic kernel THEN MAE ≤19.6 BECAUSE exemplar-specific spatial filtering. DISPROVED IF ≥20.5.

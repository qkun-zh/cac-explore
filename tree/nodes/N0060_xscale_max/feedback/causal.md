# Causal — N0060_xscale_max

## Verdict
**H0084 DISPROVED; H0083 (the 2nd-coarse-summary extension of the positive axis) REFUTED.** Best
22.886 (E23) / final 23.010 / RMSE 85.11 vs champion N0054 19.647 / 74.05 = **+3.24, FAIL gate (>20.0).**
The positive axis is a SINGLE slider, not a stack.

## Why the 2nd coarse summary hurt when the 1st helped (+3.24 vs +0.95)
Hypothesis (a) is the mechanism; (c) is its gradient-level expression; (b) is refuted.

- **(a) SINGLE-slider saturation — PRIMARY.** N0054's XScale is the MARGINAL additive coarse-mean summary:
  it sits at the representational/optimization sweet spot of the fused prototype. N0060 shows the mean is
  not a member of a stackable family: a SECOND coarse summary over the SAME ROI (whatever the statistic)
  exceeds the additive slot's headroom. The plateau is the signature — train loss falls monotonically
  3.69→2.70 (E17-30) while val pins at 22.9-23.4, i.e. the extra 98,560 params + additive term add
  capacity that overfits-to-scale with no val gain (N0058/N0059 overfit-profile family). The "orthogonal-
  to-the-mean MAX" premise was conceptually sound but moot: the slot, not the statistic, is the binding
  constraint.
- **(b) MAX-noise — REFUTED.** Under heavy-tailed ConvNeXt the max IS a single-hot-cell statistic, but if
  it injected high-variance peak noise we'd see the instability/divergence of N0056 (loss spike, stuck
  ~24.3 from E11), not a clean monotone convergence to a higher plateau. N0060 optimizes cleanly to a
  worse optimum — additive-over-saturation, not noise-driven override of the mean.
- **(c) gradient collision — CONTRIBUTING (re-expression of a).** Mean and max are NOT near-orthogonal as
  claimed: both are functionals of the SAME 7/3×7/3 ROI of the SAME heavy-tailed distribution (max is its
  extreme quantile), and two parallel Linear(384→256) maps add onto the same (B,K,256) fused vector. The
  two projectors compete in the same additive subspace → jointly over-parameterize the single proto slot.

## Not a re-run of N0056; a NEW failure type
N0056 (fine entropy, +3.06) WAS chaotic/stuck and ES@17; N0060 converges to a clean-but-worse floor over
all 30ep. N0055 (key split, +1.19) was a slow cardinality dilution; N0060 is a sharper additive-saturation.
So this is a NEW **"too-many-coarse-summaries-saturate"** phenomenon, the 3rd flavor of "more exemplar
info degrades," distinct from both prior mechanisms.

## Lineage meaning of H0083/H0084
- **H0084 refuted** outright (22.886 > 20.0): a coarse MAX summary alongside the mean does not beat 19.647.
- **H0083 refuted in its extension form**: a 2nd single-slot coarse additive summary on the same ROI does
  NOT exploit the positive axis — the axis is a **single-slider**. The champion's +0.95 mean-XScale stays
  VALID and untouched; only the *stacking* is killed. Do NOT book any further coarse-summary-on-same-ROI
  retry; the next champion-faithful move must change the interface (e.g. shared-interface aux on the
  frozen-backbone features), not add a parallel projector onto the same prototype.

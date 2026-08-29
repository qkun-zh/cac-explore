# Diagnostic — N0061_countnorm (FAIL, 30/30, best 23.502@E27 / final 23.718)

## Root cause — PREMISE falsified, not implementation
Clean run (status=success, 30/30, oom=false, params 31.35M). FAIL gate (>20.0) fired (+3.855 vs 19.647; RMSE 87.61 vs 74.05). Train loss fell 3.06→2.66 while val stuck ~23.5 from E16 — the multiplicative factor overfits a per-image count-rectification game, not the tail.

## Implementation faithfulness — FAITHFUL (model.py:204-210)
```
if use_countnorm: dens=out["density"]      # post-GCA: dens+bias (model.py:202)
                  z=clamp(MLP([GAP(fine)||e_mean]),-2,2); f=exp(z); out["density"]=dens*f
```
Multiplicative branch applied AFTER GCA additive bias, identically per idea.md:14 ("out=density of the intact GCA path"), z clamp [-2,2], W2 zero-init (model.py:168-169) ⇒ f=1 at init. No coupling (§5.14). Ordering additive→multiplicative is sound; the small 0.02-attenuated GCA bias is merely scaled along with the density — a minor secondary interaction, NOT a genuine double-count-supervision conflict.

## Premise error — the mechanism never achieved count-weighting
idea.md's BECAUSE requires shape error to be count-free: out = n_hat·Ð with Ð unit-mass. But `champion_density` integrates to the predicted count `s`, NOT to 1, and gt_d is un-normalized Gaussian blobs (sum=gt_c). So out = dens·f keeps the **count-scaled** density, not a unit-mass Ð. For a decent model dens≈gt_d (both ∝ count), so per-image MSE(dens·f, gt_d) ≈ f²·MSE(dens,gt_d): the factor merely rescales the raw count-scaled MSE by f² per image — reweighting WHICH images dominate the loss, not decoupling shape error from count. The engine MSE just gets per-image scaled; the tail-steering premise was falsified. The +3.86 floor is this scale distortion (RMSE ballooning to 87.61), not a survivor of a failed mechanism.

## Classify — NEW design-premise negative, no failure_modes.md append
Not a re-run (first probe of the count-normalization readout axis), not an implementation bug, not an ops pitfall. Design-premise negative ⇒ conservative NO append.

## Recommendation
Multiplicative count-normalization readout is CLOSED (N0037 unit-mass factor +0.5 pre-champion; N0061 multiplicative autocale +3.86 frozen — both negative). The frozen-regime count-side slot has ONE unprobed boundary: GCA is the unique POSITIVE count module and it is 0.02-attenuated and loss-level. The genuinely untested lever is *count-as-supervision* rather than count-as-readout-rescale — e.g. engine `tail_reweight` (train.py:334-339, present but dead) reweights the raw MSE by 1/gt_c at the loss, not the readout. That requires a (currently forbidden) engine change; short of that, the count-normalization direction does not close further without re-opening the decoder/recipe.

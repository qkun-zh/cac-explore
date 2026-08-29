# Idea — N0062_fine_decoder (parent: N0054_xscale_exemplar, frozen champion)

## Header
- **Parent**: N0054 (GCA+XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED.
- **Premise (research gold)**: the head's `fine` (B,128,96,96) feeding the DensityDecoder is produced by
  FineFuser fusing h2(1/8,192)+h3(1/16,384) then bilinear-upsampled to 1/4 — so `fine` is NEVER native
  1/4; it is upsampled-from-1/8. The frozen backbone's hs index 1 = **h1 (native 1/4, 96ch)** is available
  but **EXPOSED TO NOTHING**. Dense tail (17/83 imgs = 75.86% SSE; RMSE 74 vs MAE 19.6) fails by
  cell-quantization mass loss; N0026: tail error falls monotonically with resolution.

## Change (structural, FROZEN, optimizer unchanged, single switch `use_fine_decoder`)
**Native-1/4 receiver-resolution enrichment of the DensityDecoder INPUT only.**
- Extend `Backbone` hs_map from `(2,3)` to include index 1 → expose h1 (96ch @ 1/4) via
  `forward_feature_map` as an extra shared frozen-interface output (backbone feature interface, §5.14).
- Injector: `h1 ->(1x1 conv 96->8)+GroupNorm(2,8)` → 8-ch native-1/4 map; concat as extra input channels
  to `DensityDecoder.block[0]`, so `in_ch` 192 → 200.
- Module math: first-conv delta `(200-192)*256*9 = 18,432` + injector `96*8+8 = 776` ≈ **+20k params**;
  hidden 256 unchanged.
- Interface invariant: h1 touches the decoder INPUT only — FineFuser (fused prototype), Condenser,
  ExemplarEncoder, GCA, and all exemplar/count pathways untouched. `use_fine_decoder=False` = exact
  champion bit-identical.
- No exemplar / condenser / count-multiplier / attention / engine change. `out["density"]` sole gradient
  carrier (via the widened first conv, which now routes h1 gradients).

## Why / grounding
- **Native-1/4 per-cell fidelity** is a NEW receiver-resolution axis, distinct from all closed negatives:
  N0056 added a FINE scale to the EXEMPLAR AGGREGATION (producer side, hurt); RGA (N0053) was a
  spatial OUTPUT BIAS (hurt). This widens the DensityDecoder's INPUT at native 1/4 — the tail's
  cell-quantization mass-loss axis (N0026: tail error falls monotonically with resolution).
- Frozen backbone h1 is free, currently dead. Injecting it gives the decoder a true native-1/4
  representation instead of the upsampled-from-1/8 `fine` it reads today.

## Hypothesis (verbatim, pre-registered)
**H0090** IF [a frozen native-1/4 backbone feature h1 is injected as extra input channels to the
DensityDecoder (receiver-resolution enrichment, exemplar/condenser/GCA/attention untouched)]
IN [frozen N0054, 30ep @384, no engine change] THEN [val MAE < 19.647] BECAUSE [the dense-image tail
(76% SSE in 17 imgs N>=500) fails by cell-quantization mass loss from reading only upsampled-from-1/8
fine; native 1/4 per-cell fidelity places peak mass correctly (N0026: tail error falls with resolution),
and this decoder-RECEIVER axis is orthogonal to N0056's exemplar-aggregation entropy and RGA's
output-bias]. DISPROVED IF best val MAE > 20.4.

## Gates
- R1 smoke: stub backbone, `use_fine_decoder=True`; params ≤32M (+20k); density (B,1,96,96); loss finite
  & drops; verify `use_fine_decoder=False` removes injector + first-conv widening (= N0054 bit-identical).
- R2 30ep @384 frozen recipe. Gates vs 19.647: **CONFIRM <19.45** (2nd seed if <19.40);
  **WEAK-KEEP** (native-res receiver helps marginally / tie informative) 19.45-20.0;
  **FAIL >20.0** (decoder-input fine enrichment re-triggers a known negative).
  Early-stop ep16+ ≥+1.5 worse than parent best.

## Risk (honest)
Decoder-input widening is the FIRST probe of the receiver-resolution axis in this frozen regime; N0056
(injecting exFINE example-scale entropy, exemplar side) and RGA (density output bias) both hurt, so the
positive exemplar/count axes are saturated. If native-1/4 injection re-triggers a known negative (e.g.
redundant high-res channels that overfit to the tail, mirroring N0060/N0061 overfit-to-scale), the FAIL
gate fires decisively. +20k params is small but not zero; the widened first conv changes the champion's
training basin even at identity-ish init.

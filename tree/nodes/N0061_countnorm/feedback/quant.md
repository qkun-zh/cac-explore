# Feedback — N0061_countnorm (Quant)

## Verdict: FAIL (clean, pre-registered R4 · >20.0). H0088 DISPROVED.

## Headline numbers (vs champion N0054 = 19.647 / RMSE 74.05)
- Best val MAE **23.502** @E27, final 23.718, RMSE 87.61.
- Best delta vs champion: **23.502 − 19.647 = +3.855** — well outside noise (±0.25) and the 20.0 FAIL bar.
- RMSE **87.61 vs 74.05 = +13.56** — the "RMSE-tail shrink" mechanism predictions went the wrong way too; harm corroborated on the same metric H0088 claimed to improve.

## Same-epoch trail vs champion (the tail, not just the floor)
- E16: 24.308 vs 22.338 = **+1.97**
- E18: 24.275 vs 21.567 = **+2.71** ← crosses the early-stop bar
- E21: 23.562 vs 21.695 = **+1.87**
- E27: 23.502 vs 19.664 = **+3.84**
- Pattern: N0061 NEVER closes to within ~+1.9 of champion after E16. It plateaus ~23.5 while champion descends to 19.65. Even the best epoch trails the champion's worst of the four by >1.8. No epoch, anywhere in the run, is competitive.

## Gate resolution
- CONFIRM (<19.45): no. WEAK-KEEP (19.45–20.0): no. **FAIL (>20.0): fired.**
- H0088 DISPROVED IF best val MAE > 20.4 → 23.502 > 20.4 → **DISPROVED.**
- Early-stop trigger: E18 same-epoch **+2.71 ≥ +1.5** at ep16+ → bar fired as designed.

## N=1 sufficiency
FAIL at N=1 is defensible: the margin over the bar (+3.86 vs 20.0) and over the disprove line (+3.10 vs 20.4) is an order of magnitude beyond the ±0.25 noise floor. A second seed cannot plausibly recover the ~3.9-MAE gap to re-open CONFIRM/WEAK; no 2nd-seed clause is required for FAIL. Bars to 2nd seed only apply to the CONFIRM (<19.40) branch.

## Attribution
With identity-at-init verified (use_countnorm=False bit-identical to champion) and CountNormHead being the sole single-switch change, the +3.855 regression is attributable to the multiplicative count-consistency autoscale (density·exp(z)) over the champion decoder. The multiplicative-log-scalar readout direction is **refuted for this head in the frozen regime**. Closing: FAIL · H0088 DISPROVED · do not retry multiplicative count-autoscale for a count-normalized readout.

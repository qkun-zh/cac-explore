# Feedback — N0062_fine_decoder (Quant)

## Verdict: FAIL (clean 30/30ep, pre-registered R4 · >20.0). H0090 DISPROVED.

## Headline numbers (vs champion N0054 = 19.647 / RMSE 74.05)
- Best val MAE **21.323** @E17, final 22.298, best RMSE 75.955 (+1.91 vs 74.05).
- Best delta vs champion: **21.323 − 19.647 = +1.676** — well outside noise (±0.25) and the 20.0 FAIL bar (+1.323 over).
- RMSE **75.96 vs 74.05 = +1.91** — the dense-tail RMSE benefit H0090 promised (tail error falls with native res) did not materialize; harm on both metrics.

## Same-epoch trail vs champion (full 30ep logs on server)
- E01: 30.639 vs 32.660 = **−2.02** (early win)
- E02: 28.726 vs 30.315 = **−1.59**
- E06: 28.646 vs 25.608 = **+3.04**
- E07: 25.485 vs 28.727 = **−2.24**
- E11: 23.711 vs 22.523 = **+1.19**
- E15: 24.122 vs 21.635 = **+2.49**
- E16: 23.640 vs 22.338 = **+1.30**
- E17: 21.323 vs 21.250 = **+0.07** ← best epoch, near-tie but still behind
- E19: 24.764 vs 21.108 = **+3.66** ← early-stop bar fired
- E20: 25.109 vs 20.922 = **+4.19**
- E24: 22.275 vs 19.970 = **+2.31**
- E27: 22.669 vs 19.664 = **+3.01**
- E30: 22.298 vs 19.720 = **+2.58**
- Pattern: competitive/tie through E17 (best 21.32 ≈ 21.25), then champion descends 21.25→19.65 (E17-29) while N0062 plateaus 21.3→22.3 and never recovers — late-training collapse, not early misfit. Plateau ~22.0-23.6 from E18-30 vs champion 19.6-20.9.

## Gate resolution
- CONFIRM (<19.45): no. WEAK-KEEP (19.45–20.0): no. **FAIL (>20.0): fired.**
- H0090 DISPROVED IF best val MAE > 20.4 → 21.323 > 20.4 → **DISPROVED.**
- Early-stop trigger: E19 same-epoch **+3.66 ≥ +1.5** at ep16+ → bar fired as designed (E16 +1.30 already >1.5? actually E16 +1.30 <1.5, E19 is first clear cross).

## N=1 sufficiency
FAIL at N=1 is defensible: margin over FAIL bar +1.323 and over disprove line +0.923 is >3× noise floor (±0.25). Champion's late descent (1.6 MAE drop E17-29) is systematic; N0062's train loss 2.29 final vs 2.37 champion shows it fit train equally well — val gap is generalization, not optimization. A second seed cannot plausibly recover 1.67 gap plus the late divergence trend (13 epochs stuck). No 2nd-seed clause required for FAIL.

## Attribution
With identity verified (use_fine_decoder=False bit-identical: state_dict diff 0.0, forward diff 0.0, param delta 0) and the native-1/4 injector (1x1 96→8+GN) being the sole single-switch change (+19,224 params, 31.34M), the +1.676 regression is attributable to widening the DensityDecoder input with frozen h1. The decoder-receiver-resolution direction is **refuted for this head in the frozen regime** — at least for low-level h1 injection.

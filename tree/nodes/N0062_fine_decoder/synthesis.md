# Synthesis — N0062_fine_decoder (parent N0054, 30/30ep, verdict NEGATIVE/FAIL)

## Verdict
**KILL — H0090 DISPROVED.** Native-1/4 decoder injection fails.
Best **21.323 @E17**, final 22.298, floor delta **+1.676** vs champion N0054 19.647 (RMSE 75.96 vs 74.05, +1.91). FAIL gate (>20.0) fired (+1.323 over, ~5σ). Params +19,224 → head 3.52M / total 31.34M ≤32M ✓. Single switch `use_fine_decoder`; off = bit-identical N0054 (verified 0.000 diff, model.py:32-48,125-148).

## Change summary
On frozen N0054 (GCA+XScale): expose frozen hs[1]=h1 (96ch @1/4, currently dead; hs[1] verified 96,96,96) via Backbone hs_map+(1) and inject `h1→1x1 96→8+GN(2,8)` as extra input channels to DensityDecoder.block[0], in_ch 192→200. ExemplarEncoder, Condenser, GCA, attention, FineFuser, recipe untouched. Meant to give the decoder native 1/4 per-cell fidelity to fix cell-quantization mass loss on the dense tail (75.86% SSE in 17 imgs N≥500; N0026 tail error falls with res).

## Result + gates
- **R4 FAIL fired**: best 21.323 >20.0; +1.676 floor beyond ±0.25 noise.
- **H0090 DISPROVED** (21.323 >20.4 bar). N=1 defensible (FAIL band, margin +1.32 over FAIL, +0.92 over disprove; late divergence 13 ep plateau makes 2nd seed implausible).
- Same-epoch trail: competitive early (E01 -2.02, E02 -1.59), tie at best epoch E17 (21.323 vs 21.250 +0.07), then champion descends 21.25→19.65 (E17-29) while N0062 plateaus 21.3→22.3. E19 +3.66 crosses early-stop bar.
- RMSE 75.96 vs 74.05 — no tail RMSE win; train loss 153.58→2.29 = overfit-to-texture, not tail repair.

## Evidence chain — frozen head density-side inputs (vs 19.647)
| Node | Axis | Delta |
|---|---|---|
| N0053 RGA | spatial output bias | +1.74 |
| N0056 XFine | info-add extra fine to exemplar agg | +3.06 |
| N0061 countnorm | multiplicative count autoscale | +3.86 |
| N0062 fine_decoder | decoder INPUT native-1/4 injection (H0090) | **+1.68** |

Milder than other density-side adds but still decisively negative. Decoder input is the least harmful density-side axis yet probed, but harmful.

## Exemplar / aggregation axis (context, unchanged)
| Node | Delta |
|---|---|
| N0055 XScale-Key | +1.19 |
| N0057 cond-matcher | +1.43 |
| N0058 PMOM | +3.50 floor |
| N0059 PoM-Morph | +1.31 floor |
| N0060 XScale-MAX | +3.24 |
Unchanged by N0062 (exemplar pathway byte-identical). Count/readout axis also unchanged (N0037/61).

## Mechanism attribution (all 4 feedbacks agree — MECHANISM INSUFFICIENCY, not ops)
- **Wrong abstraction (primary):** hs[1] is stem-stage texture (Gabor-like), not count-semantic; `fine` is already count-tuned via FineFuser fusion+refine. Adding raw h1 injects count-irrelevant high-freq noise the decoder must suppress → val plateau while train fits.
- **Basin shift (contributing):** new 8 columns Kaiming-initialized (no zero-init/attenuation) perturb champion optimum at step 0 → E03 +5.76 spike, tie at E17, then failure to descend final 1.6 MAE. Unlike GCA zero-init, no identity preservation.
- **Target-level limit (contributing):** dense-tail error is overlapping Gaussian kernel quantization, not input-res starvation; plain MSE on raw blobs (train.py:344-345) cannot be fixed by input widening alone (same PREMISE LIMIT as N0061 but milder). Tail-steering needs target-side weighting (dead `tail_reweight` 334-339).
- Implementation FAITHFUL (bit-identical off, 31.34M). Not an ops pitfall → NO failure_modes.md append.

## Bookings (append-only ledger, quality-gated via check_hypothesis.py)
- **H0090** create (pre-registered RUN) → evidence `run` (N0062 executable) → **refutes** strength 1.0 (unscored): 21.323 >20.4; FAIL fired. Best 21.323 @E17 / final 22.298, floor +1.676, RMSE 75.96 vs 74.05.
- H0086/H0087 absent from ledger — not created.

## Calibration bin table (calibration_report.py, verbatim, eta=0.20)
```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N  correct    rate
[0.25,0.50)      1        0      0%
[0.50,0.75)     14        5     36%
[0.75,1.00)      0        0       -
<0.25            0        0       -
overall         15        5     33%
```
H0090 logs only unscored run/refutes (no scored test); H0089 similarly. 4 legacy creates (H0034/35/36/47) fail IN-marker lint — pre-existing.

## Next move (recommendation)
Decoder-receiver-resolution via raw h1 is CLOSED. Do NOT retry naive stem concatenation. Remaining frozen levers: count-as-SUPERVISION via dead `tail_reweight` (engine change, forbidden) is the only untested tail axis; otherwise frozen-head LOS is hardening (exemplar + count + resolution all mapped negative). Next champion-faithful headroom must be regime/extent change or a zero-init, semantic refinement of `fine` itself — not raw early-stage injection.

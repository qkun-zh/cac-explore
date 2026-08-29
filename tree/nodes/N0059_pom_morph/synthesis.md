# Synthesis — N0059_pom_morph (30/30ep, NEGATIVE)

## Verdict
**KILL — H0082 DISPROVED; H0081 confirmed-by-failure (falsifier did NOT disprove it).** The
param-matched, confound-cleared PoM-PolyMorpher producer swap failed: best val **20.958 @E18**, final
**21.415**, RMSE 76.06, floor delta **+1.31** vs champion N0054 19.647 (head 3.522M / total 31.34M,
capacity-matched). Producer self-attention on the 49 exemplar tokens is load-bearing independent of
capacity.

## Change summary
Single-switch `use_pom` on parent N0054 (frozen): replaced {proj 384→256 + 2× TransformerEncoderLayer
self-attention + attn-pool} with {proj + 2× PoM-PolyMorpher blocks (2604.06129 eq.3: token-averaged
2nd-order moments H + per-token sigmoid gate + W_o, norm_first residual)} over ALL 49 ROI tokens,
capacity matched to 3.52M via D=352/k=2, ParTY part-pool EXCLUDED (N0058's confound). Condenser, GCA,
XScale, shape_mlp, decoder, recipe untouched. use_pom=False restores exact N0054. Verified vs
result.json: status success, 30/30ep, params 31.34M.

## Result + gates fired
- **KILL (R4), both prongs**: best 20.958 ≥ 19.90; ep16+ same-epoch bar fired E16 (+2.13 vs 22.338).
- **H0082 DISPROVED**: 20.958 ≥ 20.40 (+0.56, non-marginal). N=1 defensible (not the CONFIRM band).
- Best-of-30 is an upward-lucky 1-epoch dip; tail E19–30 oscillates 21.09–23.45, typical ≈21.4
  (+1.75 above champion) — KILL more decisive than the best-episode number alone.

## Evidence chain — aggregation-swap axis (vs N0054 19.647)
| Node | Axis | Delta |
|---|---|---|
| N0055 XScale-Key | info-add 2K keys | +1.19 |
| N0056 XFine | info-add extra fine scale | +3.06 |
| N0057 cond-matcher | consumer swap (condenser MHA) | +1.43 |
| N0058 PMOM | producer swap (part-pool, −42% cap) | +2.17 same-ep / +3.50 floor |
| N0059 PoM-Morph | producer swap (matched, full-token) | +1.31 floor / +2.13 same-ep |

Producer, consumer, and info-add axes all degrade. N0059 subtracts N0058's capacity+pooling confounds
and matches residuals: floor gap shrinks +3.50→+1.31 (matched capacity lets the operator traverse the
late-training window), but the operator still loses +1.31 — the operator-class deficit is real and
capacity-independent.

## Mechanism attribution (all 4 feedbacks agree)
- PoM's shared token-averaged moment state + shared gate lack self-attention's per-query softmax
  contrast → no per-token context specialization; lowers to gated GAP+MLP.
- D=352 ≫ n=49 rank ceiling (moment state rank ≤49) partly neutralizes the matched capacity (wide W_o
  under-utilized).
- Train loss → 2.13 while val stuck ≈21.4 = overfit-to-scale, not exemplar-drive (no E18-27 late gain).
- No implementation bug (model.py:83-93 faithful to Eq.3; GELU-vs-clamp/α-init documented as incidental).
- Not a NEW failure mode — same family as N0057/N0058, confounds cleared. No failure_modes.md append.

## Bookings (append-only ledger, quality-gated via check_hypothesis)
- **H0082** created (pre-registered falsifier, book-as-RUN) then **refuted** (strength 1.0, best
  20.958 ≥ 20.40). Booked as `refutes` (unscored) per H0076/77/79/80 sibling convention.
- **H0081 strengthened** (NEW event, no overwrite): `supports` strength 0.9 → 0.50 → 0.59. Confirmed-
  by-failure: the exact param-matched falsifier the law demanded did NOT beat 19.647. Strength 0.9
  because the +1.31 best-episode floor sits marginally under the law's +1.4 text (same-epoch +2.13 /
  typical ≈+1.75 exceed it). Still uncertain (<0.75); advisory eta=0.20, STATE refuted set governs.
- **H0083 created** (K_synth=2): second single-slot COARSE additive XScale-style summary (keeps the
  single fused prototype, both attention paths untouched) — the ONE positive axis per all feedbacks.
  The alternative param-matched softmax-contrast-preserving attention swap was considered and
  deprioritized: three consecutive swap failures (consumer/producer-partpool/producer-matched) close
  the aggregation-operator axis.

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
H0080/H0082 refutations logged with `refutes` (unscored by the report), consistent with sibling
convention; ledger conf is advisory. 4 legacy creates (H0034/35/36/47) fail the IN-marker lint —
pre-existing, untouched this session.
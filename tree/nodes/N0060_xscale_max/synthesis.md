# Synthesis — N0060_xscale_max (parent N0054, 30/30ep, verdict NEGATIVE/FAIL)

## Verdict
**KILL — H0084 REFUTED; H0083 refuted (extension form).** A second coarse **MAX-order-statistic**
summary additively fused onto the SAME exemplar prototype beside the retained mean-XScale fails: best
**22.886 @E23**, final 23.010, floor delta **+3.24** vs champion N0054 19.647 (RMSE 85.11 vs 74.05,
+11.06 / +14.9%). FAIL gate (>20.0) fired decisively (+2.886 margin, ~3.3σ). Params +98,560 →
head 3.60M / total 31.42M ≤ 32M ✓. Single switch `use_xscale_max`; off = bit-identical N0054.

## Change summary
On frozen N0054 (GCA+XScale, use_ddca=False): ADDITIVE second coarse single-slot summary —
`adaptive_max_pool2d` over the ALREADY-aligned 7×7 ROI (model.py:88, zero extra FLOPs) → per-channel
MAX → `xproj_max: Linear(384→256)` → added to the SAME fused (B,K,256) prototype (model.py:100-103),
BESIDE the retained per-channel-mean XScale. Producer self-attn, condenser cross-attn, GCA, decoder,
recipe untouched; condenser still sees ONE fused prototype (N0055 cardinality NOT implicated). MAX
chosen over a 2nd MEAN (red-team: mean corr 0.85–0.97 = redundant) as a near-orthogonal order statistic.

## Result + gates
- **R4 FAIL fired**: best 22.886 > 20.0; +3.24 floor; ~3.3σ beyond the ±0.25 noise band.
- **H0084 REFUTED** (22.886 > 20.0 DISPROVED bar). N=1 defensible (KILL band, not CONFIRM; no 2nd seed).
- Tail E17–30 val plateaued 22.9–23.4 while train loss dropped 3.69→2.70 (−27%) = overfit-to-scale:
  the MAX head memorizes peak magnitude, generalizes nothing. Same-epoch gaps widen as champion
  descends (E16 +2.78, E23 +2.47, E29 +3.43).
- RMSE 85.11 vs 74.05 corroborates dual-metric harm, not an MAE artifact.

## Evidence chain — exemplar coarse-summary / aggregation axis (vs N0054 19.647)
| Node | Axis | Delta |
|---|---|---|
| N0055 XScale-Key | info-add 2K keys (cardinality) | +1.19 |
| N0056 XFine | info-add extra fine scale (entropy) | +3.06 |
| N0057 cond-matcher | consumer swap | +1.43 |
| N0058 PMOM | producer swap (part-pool, −42% cap) | +2.17 same-ep / +3.50 floor |
| N0059 PoM-Morph | producer swap (matched, full-token) | +1.31 floor / +2.13 same-ep |
| N0060 XScale-MAX | 2nd coarse MAX on SAME ROI | **+3.24** |

Every second summary of the same spatial source — mean (redundant), MAX (order-stat), grid/part-pool —
plus every attention/operator swap is negative. The exemplar coarse-summary axis is now FULLY MAPPED with
ONE positive: the champion mean-XScale (+0.95). H0083's "headroom on the positive axis" is answered: none.

## Mechanism attribution (all 4 feedbacks agree)
- **SINGLE-SLIDER saturation (primary)**: the additive coarse slot on the fused prototype is ONE slider —
  a single summary fills it; a second exceeds headroom regardless of statistic. The "order-stat-orthogonal
  to mean" premise is moot: the SLOT, not the statistic, binds. NEW phenomenon
  **"too-many-coarse-summaries-saturate"** — 3rd distinct flavor of "more exemplar info degrades"
  (≠ N0055 cardinality dilution, ≠ N0056 chaotic fine entropy; N0060 converges cleanly to a worse floor).
- **MAX-noise REFUTED** (causal): monotone convergence to a worse optimum, not instability.
- **Projector collision contributing**: two parallel Linear(384→256) projectors add into the same
  256-d subspace (over-parameterized single slot); mean+max are both functionals of the SAME ROI.
- NOT an ops pitfall (clean run, faithful code) — no failure_modes.md append. Operator-swap
  (N0057/58/59) axis and cardinality (N0055) NOT implicated.

## Bookings (append-only ledger, quality-gated via check_hypothesis.py)
- **H0083** `run` (N0060 = executable test) then **refuted**: best 22.886 ≥ 19.65 bar; unscored
  `refutes` (sibling convention). The extension of the positive axis has no headroom.
- **H0084** created (pre-registered, verbatim idea.md) then **refuted**: 22.886 > 20.0; strength 1.0,
  unscored `refutes`.
- **H0085 created** (K_synth=2): refined NEGATIVE law — the coarse-summary slot is a SINGLE slider;
  exactly one additive coarse summary (mean-XScale) is positive; any 2nd ROI summary over the same
  spatial source (mean/MAX/grid/part-pool) harms or is neutral; the exemplar coarse-summary aggregation
  axis is fully mapped. DISPROVED IF a 2nd coarse single-slot summary beats 19.647.
- **H0081 NOT touched**: N0060 is a granularity-enrichment probe, not an operator swap — no direct link.

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
H0083/H0084 logged with unscored `run`/`refutes`, so no new scored tests; conf is advisory, STATE
operational refuted set governs. 4 legacy creates (H0034/35/36/47) fail the IN-marker lint — pre-existing.

## Next move (recommendation)
Do NOT retry another order-statistic / spatial summary (mean / max / grid / L2) on the aligned ROI —
mapped, negative. Champion-faithful headroom must change the interface (shared-interface aux on frozen
backbone features, regime/extent change), not add a parallel projector onto the same prototype.
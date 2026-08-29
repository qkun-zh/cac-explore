# Synthesis — N0061_countnorm (parent N0054, 30/30ep, verdict NEGATIVE/FAIL)

## Verdict
**KILL — H0088 DISPROVED.** Multiplicative count-consistency autoscale on the frozen champion fails.
Best **23.502 @E27**, final 23.718, floor delta **+3.855** vs champion N0054 19.647 (RMSE 87.61 vs 74.05,
+13.56). FAIL gate (>20.0) fired decisively (+3.5σ). Params +24,705 → head 3.53M / total 31.35M ≤ 32M ✓.
Single switch `use_countnorm`; off = bit-identical N0054 (verified 0.000 diff).

## Change summary
On frozen N0054 (GCA+XScale): bolt CountNormHead PARALLEL to GCA reading ONLY shared interfaces
GAP(fine) (B,128) + e_mean (B,256). z = clamp(W2·GELU(W1·[·]),−2,2), W2 zero-init (identity-at-init f=1,
forward == champion at ep0); `out.density = champion_density * exp(z)` — per-image multiplicative,
channel-invariant, spatial-free log-link factor (model.py:160-171,204-210). Decoder, GCA additive bias,
condenser, exemplar producer, recipe, optimizer untouched. Meant to make MSE count-weighted (shape
count-free, total pinned to log-space count estimate) and steer gradient out of the RMSE tail.

## Result + gates
- **R4 FAIL fired**: best 23.502 > 20.0; +3.855 floor, well beyond ±0.25 noise.
- **H0088 DISPROVED** (23.502 > 20.4 bar). N=1 defensible (KILL band, no 2nd-seed clause for FAIL; margin
  ~3.9 MAE cannot plausibly recover).
- Same-epoch trail never competitive: E16 +1.97, E18 +2.71 (early-stop bar fired at ep16+), E21 +1.87,
  E27 +3.84. Plateau ~23.5 from E16 while champion descends to 19.65.
- RMSE 87.61 vs 74.05 — the very metric H0088 existed to shrink got WORSE (+13.56), corroborating
  dual-metric harm, not an MAE artifact. Train loss 3.06→2.66 = overfit-to-scale.

## Evidence chain — count/readout axis (vs N0054 19.647)
| Node | Axis | Delta |
|---|---|---|
| N0037 PPC-Head | N·p factorization (shape/count split, decoder re-arch) H0053 | +0.5 (20.99) |
| N0029 loghead | log-space density readout cousin (H0043, pre-champion) | -- (no recorded result in repo) |
| N0041 exemplar-count | per-exemplar count-sum (H0061) | 24.06 (+4.4) |
| N0061 countnorm | per-image multiplicative count autoscale (H0088) | **+3.855** |

Every effort to separate COUNT from SHAPE at the readout — factorized (N0037), log (N0029), per-exemplar
(N0041), residual multiplicative (N0061) — is negative. N0061 is the same family as N0037 but MORE severe
(+3.86 vs +0.5) because the factor acts on the champion's OWN count-corrected (GCA) output, stacking a
second global count channel on the same axis. **Readout count-normalization is CLOSED; the count axis is
fully occupied by GCA's additive, 0.02-attenuated, zero-init bias — the unique positive count-side module.**

## Exemplar coarse-summary / aggregation axis (context, N0054 19.647 — FULLY MAPPED)
| Node | Axis | Delta |
|---|---|---|
| N0055 XScale-Key | info-add 2K keys (cardinality) | +1.19 |
| N0056 XFine | info-add extra fine scale (entropy) | +3.06 |
| N0057 cond-matcher | consumer swap | +1.43 |
| N0058 PMOM | producer swap (part-pool, −42% cap) | +2.17 same-ep / +3.50 floor |
| N0059 PoM-Morph | producer swap (matched, full-token) | +1.31 floor / +2.13 same-ep |
| N0060 XScale-MAX | 2nd coarse MAX on SAME ROI | **+3.24** |
Unchanged by N0061 (exemplar pathway byte-identical). Exemplar axis done; count/readout axis now also done.

## Mechanism attribution (all 4 feedbacks agree — PREMISE ERROR)
- **PREMISE falsified (primary, not implementation, not ops)**: the engine loss is plain MSE(dens, gt_d)
  where gt_d = raw NON-unit-normalized Gaussian blobs (sum=count), NOT c-normalized. A multiplicative
  `dens→f·dens` only rescales the output; it cannot reweight the target's c² curvature. Weighting lives on
  the TARGET axis (the dead `tail_reweight`, train.py:334-339); an output multiplier can't replicate it.
  For dens≈gt_d, per-image MSE(dens·f, gt_d) ≈ f²·MSE(dens, gt_d) → the factor merely reweights WHICH
  images dominate, never decoupling shape error from count. Tail-steering NEVER occurred.
- **Axis collision with GCA (contributing)**: out = f·(d+bias); both count knobs read the same
  GAP(fine)+e_mean; L1(f·(s+0.02n_aux),c) is ONE equation TWO unknowns → non-unique fixed points,
  split attribution, destructive interference. GCA in the multiplication path amplifies it.
- **Range worst on the tail (contributing)**: f∈[e⁻²,e²]=[0.14,7.4]; wrong low f on a dense image
  collapses mass, worst MSE exactly on high-c (error²∝c²); ±2 clamp zeroes z-gradients at the boundary.
- Implementation FAITHFUL (single-switch, identity-at-init verified). Not an ops pitfall → NO
  failure_modes.md append (design-premise negative).

## Bookings (append-only ledger, quality-gated via check_hypothesis.py; H0086/H0087 absent from ledger — not created)
- **H0088** create (pre-registered RUN, verbatim idea.md:46) → evidence `run` (N0061 = executable test) →
  **refuted** strength 1.0 (unscored per sibling convention): 23.502 > 20.4; FAIL fired. Best 23.502 @E27
  / final 23.718, floor +3.855, RMSE 87.61 vs 74.05.
- **H0089 created** (K_synth=2, refined NEGATIVE): count-weighted readout via an output multiplier cannot
  work in this engine because gt_d is raw/non-unit-mass and only out[density] carries gradients
  (premise error); multiplicative count-autoscale closed (N0037 N*p AND N0061); GCA (additive, attenuated,
  zero-init) is the unique positive count-side module. DISPROVED IF a readout-side output multiplier beats
  19.647.

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
H0088 logs only unscored `run`/`refutes` (no scored test); H0089 is a create with no evidence yet — no
new scored tests. Conf is advisory; STATE operational refuted set governs. 4 legacy creates
(H0034/35/36/47) fail the IN-marker lint — pre-existing, unchanged.

## Next move (recommendation)
Do NOT retry multiplicative/scale count readouts — mapped, negative. The remaining unprobed count lever is
count-as-SUPERVISION (the dead `tail_reweight` loss path), which needs a currently-forbidden engine change;
short of that the count-normalization direction does not close further without re-opening the decoder/recipe.
Champion-faithful headroom must change the interface (a shared-interface aux on frozen features outside the
count/readout and exemplar axes) or the regime/extent.

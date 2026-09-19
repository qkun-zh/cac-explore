# Causal feedback — N0003_h0009 (per-channel gate on exemplar tokens)

Observed: best val MAE **22.6076 @ ep32** (= final epoch), 32 real-data epochs, 1718.9s/1800s (epoch cap hit, not budget), EMA-on val MAE.

## 1. Mechanism → outcome path
The claimed BECAUSE ("per-exemplar channel-gain lets the decoder express count-dependent
attention") is **not implementable by this code**, so the outcome cannot be credibly caused by it:
- `forward` computes ONE gate `sigmoid(W·GAP(fine))` of shape (B, 256) and applies it to ALL K
  exemplars at once (`e * gate.unsqueeze(1)`, model.py:168-170). It is an image-global, per-channel
  gain — it cannot up/down-regulate one exemplar relative to another, so "steering the condenser
  toward scale-consistent exemplar weights" is not expressible.
- A gate that converges near a constant is algebraically absorbable into the condenser's K/V
  projections (cross-attn sees `e` only after projection). Only two ingredients then survive:
  (i) ~33k added params, (ii) conditioning on fine-content statistics. Both are capacity/input
  effects, not the hypothesized attention mechanism.
- No gate statistics are logged (mean/saturation/variance), so we cannot even check the gate is
  non-constant or content-dependent. The single scalar outcome cannot discriminate mechanism vs
  inert reparam.

Against the *only available* baseline (unseeded parent 19.647) the child is **2.96 WORSE** — a
null result, not a mechanism success.

## 2. Confounders
- **(a) Seed.** Child is seeded (20260830, deterministic). The 19.647 parent value is the UNSEEDED
  migration artifact; the same-protocol seeded rerun is in flight. Any Δ vs 19.647 mixes the gate
  effect with seed regime.
- **(b) best-at-last-epoch endpoint.** best_epoch = 32 = last epoch; the run stops at the 32-ep
  cosine ceiling, not the budget (81s unused). The parent's own config note documents late-phase
  swings in ONE trajectory (probe sat in a 22–25 plateau for 14+ epochs before converging far
  lower) — final MAE is endpoint-sensitive, and this child lands exactly in that 22–25 band.
  (Per-epoch val curve is not readable from this node's tb; only final 22.608 ≈ best 22.6076.)
- **(c) Capacity.** `Linear(128→256)` = 33,024 params (32,768 weights + bias), ~0.9% of the 3.5M
  trainable — added with **no compensating capacity reduction** anywhere, so gain (had any existed)
  could not be attributed to the gate per se.
- **(d) EMA interaction.** All trainable weights (gate included) are EMA'd at 0.999 and the reported
  MAE is EMA-inference MAE. A multiplicative gate under strong EMA adapts with second-order lag;
  a slowly-adapting rescale pathway is consistent with "still improving at the last epoch" — a
  substitute explanation for the trajectory that needs no mechanism.
- **(e) Same champion loop.** Shared loss/EMA/GCA/cosine mechanics *cancel* in a paired Δ — good
  internal validity. But we only have one child run vs one (wrong-regime) baseline, so the pair
  gives one trajectory-shaped number, not a mechanism measurement.

## 3. Decision-relevant watch: falsifier headroom
The "≥0.30 lower" bar is roughly an order of magnitude smaller than the run-to-run phase dispersion
the parent itself records (a single configured trajectory waved through a 22–25 plateau). With
best-at-last-epoch, the reported child value is the honest stoppage point (ep31≈ep32, tail flat),
so the uncertainty that matters is the *location of the pending seeded baseline*, not the child's
endpoint. Under any plausible placement of that baseline — from a low-19s repeat of the unseeded
artifact to a low-20s seeded regime — the observed 22.61 satisfies the "≥0.30 lower" falsifier only
if the baseline lands mid-22 or worse, i.e. only in the no-gain case. The confirming branch would
require a >2.9-swing attributed to a mechanism this code cannot express. Recommend treating this
as refuted *contingent on the seeded parent rerun*; the verdict against 19.647 is already beyond
noise, and nothing in the run leaves room for the gate to be the cause of a large improvement.

## 4. Next probe IF refuted (and the gate-vs-capacity control)
First-order next probe, one shot: an **equal-capacity control** that keeps the exact +33k gate
affine but severs the fine→gate conditioning (gate input = frozen random vector, or input-constant
per-channel scale, or additive rather than multiplicative broadcast), with gate-activation
statistics logged (mean/std/saturation fraction, correlated against exemplar scale ratios) — that
is what cleanly separates capacity/scale effects from any content-conditioned attention mechanism;
register it as a control, not a follow-up action. If the mechanism hypothesis is refuted, also log
and inspect the gate weights themselves to test whether the (unimplemented) per-exemplar variant is
even worth one more run before further spend on this line.
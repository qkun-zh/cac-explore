# idea.md — N0019_dino_tailup

## Title
Tail-UP weighting: sign-corrected count^+0.5 loss reweighting on the champion recipe

## Motivation & Intuition
FSC147 counts are long-tailed: dense images are rare, so gradient mass pools on low-count
samples and the RMSE/MAE tail stays ~3.6x (N0010 synthesis). Sibling N0017 ran tail-reweighting
with an INVERTED sign (`tail_exp=+0.5` ⇒ w∝1/sqrt(count), DOWN-weighting dense images):
MAE 22.19 (+0.66 vs parent 21.53) and ratio worsened to 3.80 — exactly what tail-starvation
predicts. That inverted result is dose-response evidence in the OPPOSITE direction: starving
the dense tail hurts, feeding it should help. SeqCount (inspiration §4.7) independently names
the high-density long tail as the field's pain point and patches it data-side (mosaic); we patch
it loss-side. This is the LAST loss-space shot targeting the RMSE tail before closing the family.

## Architecture Spec
- core_ideas: champion recipe VERBATIM — frozen DINOv2-S reg4 dual taps(6,11) scalar gate +
  area-prompt + adapter768 + MLP head, 392px, 40ep, count-w1.0 → 21.53 @1275s/23.11M.
- core_blocks: unchanged (tap6/tap11 concat → scalar layer-gate → adapter768 → MLP head).
- network_structure: unchanged; 23.11M params ≤32M.
- tunable_aspects: only per-image sample weights in the training loss.
- invariants: frozen backbone; density contract; weights normalized to batch-mean 1 so total
  loss scale equals parent's — pure reallocation of gradient across images, no magnitude change.

## Proposed Hypotheses
- **H0028**: IF count^+0.5 up-weighting (`tail_reweight=True, tail_exp=-0.5`; w_i ∝ count_i^0.5,
  batch-mean-1 normalized) IN FSC147 champion recipe, THEN RMSE/MAE <3.4 AND val MAE ≤21.53,
  BECAUSE gradient allocation finally reaches the dense-sample tail that parent (uniform) and
  N0017 (inverted/starved) both under-served. DISPROVED IF ratio ≥3.63 OR MAE >22.2.

## Delta vs Parent
ONLY change vs N0010: add `tail_reweight=True, tail_exp=-0.5` to cfg — N0017's two keys with
the sign flipped. Zero code change: engine's per-image weight path is already generic;
model.py is a verbatim N0010 copy; every other hyperparameter byte-identical to the champion.

## Novelty Statement
No architectural novelty — a causal single-lever ablation completing a signed pair around the
champion: uniform (21.53/3.63) vs tail-starving (N0017 22.19/3.80) vs tail-feeding (this node).
A flip in the predicted direction validates loss-side tail allocation as a near-free lever for
counting heads on imbalanced count distributions.

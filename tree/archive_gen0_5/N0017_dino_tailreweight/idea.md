# idea.md — N0017_dino_tailreweight

## Title
Champion recipe + sample-level inverse-sqrt-count loss reweighting to attack the outlier tail.

## Motivation & Intuition
All single-mechanism variants since champion N0010 (N0011 26.68, N0012 26.03, N0013 22.40, N0016
collapsed) FAILED to beat 21.53. The persistent residual is the OUTLIER TAIL: a few high-density /
small-object images produce huge errors that dominate RMSE (RMSE/MAE=3.6) and drag average MAE.
inspiration_from_GOD §7 confirms this is a known long-tail property of counting datasets: high-density
samples are extremely rare, so uniform-batch training under-weights exactly the samples whose errors
dominate the metric. Huber capping (H0020) failed because it CAPS outlier gradients; the direct lever is
the opposite — reweight loss UP toward rare high-count images. Banked H0022 names tail-reweighting as
the untested alternative to capping; N0017 instantiates it concretely.

## Architecture Spec
- Champion recipe VERBATIM (N0010_dino_multilayer_long, val MAE 21.531 / RMSE 82): frozen
  vit_small_patch14_reg4_dinov2 features_only out_indices=(6,11) at 392px → per-layer Linear projections
  → learned softmax SCALAR layer-gate mixes projected token sets → Fourier+area prompt token prepended
  → adapter(384→768→384, dropout .1) → conv MLP head → density [B,1,28,28]. Train: 40ep, bs8, lr1e-3,
  AdamW wd1e-4, eta_min1e-5, AMP, loss = MSE density + count-w1.0 L1.
- ONE CHANGE — sample-level loss reweighting in the engine:
  1. compute per-image base loss_i (MSE density + count L1, unchanged formulas);
  2. weight w_i = 1/sqrt(max(count_i,1)) using ground-truth count;
  3. normalize weights to mean 1 over batch; batch loss = mean(w_i * base_loss_i).
  Config: `tail_reweight: true`, exponent configurable `tail_exp: 0.5` (1.0 → pure inverse-frequency).
- core_blocks: none added/removed — loss-space change only; model.py identical to parent.
- tunable_aspects: tail_exp {0.25, 0.5, 0.75}; reweight on density term only vs both terms (start: both,
  one knob); optional eval-time stratified logging by true-count bucket to verify mechanism.
- invariants: backbone frozen eval; total ≤32M (~23.11M, unchanged); input multiple of 14; weights
  normalized per batch so effective LR stays stable regardless of tail_exp.

## Proposed Hypotheses
- H0026: IF tail-reweighting (w_i = 1/sqrt(max(count_i,1)), mean-1 normalized) IN FSC147 champion
  recipe, THEN val MAE ≤19.8 AND RMSE/MAE <3.4, BECAUSE gradient allocation shifts to the
  error-dominating dense samples that uniform sampling starves. DISPROVED IF MAE >21.53.

## Delta vs Parent
Parent N0010_dino_multilayer_long (21.53). Zero architecture delta; single loss-level change
(tail_reweight=true, tail_exp=0.5). Cleanest isolation of any gen-4 variant: confound-free attribution.

## Novelty Statement
Standard class-balancing trick transplanted from classification to CAC sample weighting; novelty is not
claimed — value is a falsifiable, isolated test of H0022's tail-reweighting family on the proven champion.

## Risks & Falsification Notes
- If dense images are already well-fit, upweighting may overfit them and hurt median-case accuracy
  (MAE worsens while RMSE improves) — stratified eval will disambiguate.
- sqrt dampens vs linear inverse-frequency to avoid letting the few extreme counts (>1000) dominate;
  tail_exp sweep is the escape hatch if 0.5 is too weak.

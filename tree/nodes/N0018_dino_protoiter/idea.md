# idea.md — N0018_dino_protoiter

## Title
Champion recipe + LOCA-style test-time iterative pseudo-prototype refinement (T=2, K=16) inside forward().

## Motivation & Intuition
Every single-lever variant since champion N0010 (N0011 26.68, N0012 26.03, N0013 22.40, N0016
collapsed; N0017 tail-reweight pending) failed to beat 21.53. Residual = outlier tail + small
objects: a one-shot initial prediction cannot recover missed dense clusters. LOCA (Djukic et al.,
ICCV 2023, arXiv:2211.08217) shows ITERATIVE prototype adaptation - re-fusing image-derived object
prototypes with image features over L~3 rounds - cut FSC147 RMSE 20-30% vs SOTA; gains plateau by
L~3. Mechanism: a first-pass density map localizes candidate object mass; pooling features at the
highest-mass cells yields pseudo-exemplars that sharpen the similarity signal exactly on the dense /
small-object regions where single-pass misses live. This is a gen-5 MECHANISM lever (vs gen-4's
schedule/loss levers): backbone stays frozen and runs ONCE per forward - compliant.

## Architecture Spec
- Champion recipe VERBATIM (N0010_dino_multilayer_long, val MAE 21.531): frozen
  vit_small_patch14_reg4_dinov2 features_only out_indices=(6,11) at 392px -> per-layer Linear projections
  -> learned softmax SCALAR layer-gate mixes projected token sets -> Fourier+area prompt token prepended
  -> adapter(384->768->384, dropout .1) -> conv MLP head -> density [B,1,28,28]. Train: 40ep, bs8,
  lr1e-3, AdamW wd1e-4, eta_min1e-5, AMP, loss = MSE density + count-w1.0 L1.
- ONE CHANGE - iterative refinement loop inside forward(), identical train & inference path:
  1. Pass 0: champion forward as-is -> density d_0 from adapter features F in [B,384,h,w].
  2. Round r = 1..T (T=2): flatten d_{r-1} spatially; top-K=16 cells by mass (torch.topk; gather
     keeps grad wrt F); pseudo-prototype p_r = Linear(384->384)( mean(F at top-K cells) ).
  3. ADD p_r as one extra memory token in the prompt-token set (exemplar conditioning slot),
     re-run ONLY adapter+head (backbone/gate outputs cached) -> refined density d_r.
  4. Output density = mean(d_0..d_T) so the engine contract (`density`) is unchanged.
- Training uses the SAME loop; gradients flow through all iterations; loss on the AVERAGED map
  (average chosen over last-only for gradient stability across rounds).
- core_blocks: +1 trainable Linear(384->384)+bias ~= 0.148M -> total ~23.3M, within 32M.
- tunable_aspects: T {1,2}; K {8,16,32}; mean vs mass-weighted pooling; add-to-memory vs
  concat-projection; averaged vs last-map loss.
- invariants: backbone frozen and run once per forward; loop lives inside forward() (no engine
  change); input multiple of 14; topk index selection non-differentiable but feature gather is.

## Proposed Hypotheses
- H0027: IF iterative prototype refinement (T=2, K=16) IN FSC147 champion stack THEN val MAE <=19.5
  AND RMSE/MAE<3.4, BECAUSE pseudo-exemplar feedback corrects initial misses on dense scenes (LOCA
  evidence). DISPROVED IF MAE>21.53 or OOM.

## Delta vs Parent
Parent N0010_dino_multilayer_long (21.53). Architecture delta only: +Linear(384->384) prototype
projection + T=2 re-decode passes through adapter+head. No loss/schedule/data changes - mechanism
lever cleanly isolated against every gen-4 failure.

## Novelty Statement
LOCA's iterative prototype adaptation transplanted from exemplar-query fusion to self-derived
pseudo-exemplars on a frozen-DINOv2 density stack: the model bootstraps its own prototypes from its
predicted map at test time AND train time (no train/inference mismatch). Novel for this tree; first
mechanism-level feedback loop tried here.

## Risks & Falsification Notes
- Empty/near-empty scenes: top-K still selects background cells -> prototype injects noise; watch
  low-count bucket MAE. Escape hatch: mass-weighted pool or skip round if max(d) below threshold.
- Feedback loop could amplify early errors (confirmation bias) - averaged-map loss dampens this.
- 3x adapter+head cost is small (~backbone dominates); OOM unlikely at 784 tokens but is explicit
  disproof criterion.

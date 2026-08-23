# idea.md — N0010_dino_multilayer_long

## Title
Champion recipe scaled: multi-layer DINOv2 taps + 40ep + count-weighted loss.

## Motivation & Intuition
Head exploration is complete (N0008/N0009 refuted cross-attn transfer); the champion is N0007's implicit
area-prompt + adapter + MLP head. Three low-risk scale-ups remain untested and reviewers converged on all
of them: (1) DINOv2 layers specialize — mid-layer tokens carry richer correspondence structure than the
final semantic-global layer, so tapping blocks 7+12 with a learned layer-gate should add signal; (2) the
schedule was still improving at E25 and only used 43% of tau_max; (3) RMSE is 3.4x MAE — raising
loss_count_weight to 1.0 directly targets high-count calibration errors.

## Architecture Spec
- core_ideas:
  1. Frozen vit_small_patch14_reg4_dinov2 features_only out_indices=(6,11): two token grids [B,784,384].
  2. Per-layer Linear projections to adapter space; learned softmax layer-gate (scalar logits -> weights)
     mixes the two projected token sets; prompt token prepended after fusion.
  3. Rest identical to N0007: Fourier+area prompt, adapter(384->768->384, dropout .1), conv head -> 28x28.
- core_blocks: LayerGate fusion added; everything else shared with N0007 code.
- network_structure:
  imgs[392]->frozen taps{t6,t11}->proj each->gated sum->+prompt->adapter->head->density [B,1,28,28].
- tunable_aspects: which blocks (6/8/11); gate type; epochs 40; loss_count_weight 1.0; dropout 0.1/0.15.
- invariants: backbone frozen eval; total <=32M (~23.3M); bbox [B,4]; input multiple of 14; dynamic_img_size.

## Proposed Hypotheses
- H0017: IF the champion recipe is scaled with mid+final layer gating, 40ep, and loss_count_weight=1.0
  IN FSC147, THEN val MAE <= 26.0 AND RMSE/MAE ratio < 3.3, BECAUSE mid-layer correspondence signal plus
  longer training plus stronger count pressure attack the three identified residual causes simultaneously.
  DISPROVED IF MAE > 27.65 (no gain over parent).

## Delta vs Parent
Parent N0007_dino_promptv2 (27.65). Adds layer-gated multi-block taps, 40ep schedule, count weight 1.0.
No mechanism change — deliberate scale-up of the confirmed winner.

## Novelty Statement
Not novel per se; value is in compounding three reviewer-recommended increments on the proven champion
and quantifying their joint effect under a frozen-backbone budget.

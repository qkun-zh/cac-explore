# idea.md — N0013_dino_mosaic

## Title
Data-augmentation regularizer for champion: photometric + bbox-jitter + dropout 0.2 (mosaic-lite, no engine change).

## Motivation & Intuition
N0010 is champion (val MAE 21.53 best@E26, Δ=-6.12 vs N0007, 23.11M, 392px). Synthesis flags two residual failures: (1) overfitting — train loss 14.3→7.8 (-46%) while val plateaus ~22-23 after E26, best at 65% schedule; (2) long-tail catastrophic outliers — RMSE/MAE=3.63×, count ceiling 3. Sibling N0011 (per-token gate+Huber) regresses (best 27.0 vs 21.53) — DO NOT build on N0011; N0012 (518px clone) tests resolution orthogonally, queued. Inspiration §§7-8: FSC147 high-density long-tail is SOTA bottleneck; SeqCount+ gains via mosaic augmentation (inherits CACViT mosaic) — synthetic high-count variance improves tail robustness. True mosaic/copy-paste (paste 2-4 exemplar crops, sum densities) requires GT density surgery → engine/dataset change, violating no-engine-change constraint. Fallback lever must be implementable inside `model.py:forward` (self.training-gated) or via `config.py` flags engine already respects. Photometric jitter + exemplar bbox jitter + elevated regularization is mosaic-lite: same principle (inject count/appearance variance) with zero GT warp.

## Architecture Spec
- core_ideas:
  1. Frozen `vit_small_patch14_reg4_dinov2` features_only out_indices=(6,11) → [B,784,384]×2, dynamic_img_size=True — identical to N0010.
  2. Scalar layer_logits softmax gate (2 params) mixing t6/t11 projs → +Fourier+logArea prompt → adapter 384→768→384 (dropout 0.2) → conv head 384→128→1 → density [B,1,28,28]; engine sum-conserves to GT. In-model augmentation: when `self.training`, apply tensor ColorJitter (brightness 0.2, contrast 0.2, saturation 0.15) + Gaussian noise σ=0.02 with p=0.5, and bbox jitter ±15% (uniform scale/translate clamped to [0,S]) before prompt_enc — no GT change.
  3. Regularization bump: `dropout=0.2` (adapter+head), `weight_decay=5e-4` (5×), keep N0010 schedule (40ep, lr1e-3, count_w=1.0, amp True).
- core_blocks: PromptEncoderV2, LayerGate(scalar), Adapter, Head — unchanged structure; only training-gated jitter ops added in forward.
- network_structure: imgs[392]--jitter?(train)--frozen taps{t6,t11}--proj--scalar-gate sum--+prompt(bbox-jittered)--adapter(d0.2)--head--density 28×28.
- tunable_aspects: jitter prob/strength, bbox_jitter 0.10-0.20, dropout 0.1→0.2, wd 1e-4→5e-4; input_size fixed 392 to orthogonalize vs N0012.
- invariants: backbone frozen eval; total ≤32M (~23.11M, jitter adds 0 params); bbox [B,4]; S%14==0; dynamic_img_size.

## Proposed Hypotheses
- H0022: IF photometric (color jitter+noise p0.5) + exemplar bbox jitter 0.15 + dropout0.2/wd5e-4 is added IN frozen DINOv2-S dual-tap 392px stack, THEN val MAE ≤20.0 AND overfit gap reduced (best epoch >26 and train→val divergence shrinks) AND RMSE/MAE <3.4, BECAUSE injected appearance/exemplar variance regularizes the adapter/head that overfits after E26 and mimics SeqCount+ mosaic robustness to high-density long-tail variance without GT surgery. DISPROVED IF MAE >21.53 (no gain over N0010) OR best epoch ≤26 with same overfit divergence OR RMSE/MAE ≥3.63.
- H0017 (reuse): multi-tap gate benefit persists under augmentation — expect gate weights remain balanced (not collapsed) with jitter.

## Delta vs Parent
Parent N0010_dino_multilayer_long (21.53, 392px, drop0.1, wd1e-4, no jitter). N0013 clones N0010 verbatim (same 784 tokens, same scalar gate, same 40ep/cnt-w1.0) and only adds: (a) training-mode photometric jitter + bbox jitter 0.15 in `model.forward`, (b) `dropout 0.1→0.2`, `weight_decay 1e-4→5e-4`. No resolution change — strictly orthogonal to N0012 (518px). Isolates augmentation/regularization lever; no confound with N0011 Huber/per-token.

## Novelty Statement
Not mechanistic novelty — lightweight regularization augmentation as mosaic-lite under no-engine-change constraint. First CAC test of in-model photometric+bbox jitter on frozen DINOv2-S adapter stack to attack the synthesis-flagged overfit/tail failures; tests SeqCount+ insight (mosaic→high-density robustness) via implementable photometric proxy.

## Estimated Params
~23.11M total (identical to N0010; jitter/regularization adds 0 params: frozen 21M + projs/prompt/adapter/head ~2.1M). Well under 32M.

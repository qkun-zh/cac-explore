# idea.md — N0012_dino_highres518

## Title
High-resolution champion clone: identical N0010 recipe at 518px (37×37 tokens).

## Motivation & Intuition
N0010 is the frozen-DINOv2-S champion (val MAE 21.53, Δ=-6.12 vs N0007, best @E26/40) with RMSE/MAE=3.63× — catastrophic small-object misses dominate the long tail. Sibling N0011 (per-token gate+Huber, same parent) trends worse (best 27.0 @E24 vs parent 21.53) so the N0010→N0011 levers (H0019/H0020) empirically regress — do not build on N0011. Hypothesis bank: H0017 supported c=0.59, H0014 c=0.585 validate the multi-layer DINOv2 substrate; H0019/H0020 0.50 untested & now contra-indicated. The single most reliable lever left per synthesis is input resolution. FSC147 small objects are under-sampled at 392px (28×28=784 tokens); 518px gives 37×37=1369 tokens (+75% spatial tokens) with identical param count, reducing miss rate without mechanism risk. SeqCount (docs/inspiration_from_GOD.txt §5.2, §4) shows patch granularity drives scale robustness — fixed patch division is a known sensitivity and higher resolution refines granularity orthogonally to sequence-vs-density paradigm. DINOv2 dynamic_img_size=True makes 518 safe (518%14==0, native 518) per memory/failure_modes.md.

## Architecture Spec
- core_ideas:
  1. Frozen vit_small_patch14_reg4_dinov2 features_only out_indices=(6,11) via timm dynamic_img_size=True → [B,1369,384]×2.
  2. Per-layer Linear proj to 384 + scalar layer_logits softmax gate (2 params) mixing — identical to N0010.
  3. Same Fourier+log-area prompt (PromptEncoderV2), adapter 384→768→384 GELU drop0.1, conv head 384→128(1×1)→1(1×1) → density [B,1,37,37]; engine sum-conserves upsample to GT.
- core_blocks: No new blocks; only input_size changes vs N0010. Reuse N0010 model.py verbatim except cfg.
- network_structure:
  imgs[518]->frozen taps{t6,t11}[1369,384]->proj->scalar-gate sum->+prompt->adapter->head->density 37×37.
- tunable_aspects: input_size 518 (from 392); all else fixed (epochs 40, lr 1e-3, wd 1e-4, adapter_dim 768, dropout 0.1, count_w=1.0).
- invariants: backbone frozen eval; total ≤32M (~23.11M same as N0010 — resolution adds no params); bbox [B,4]; input multiple of 14.

## Proposed Hypotheses
- H0021: IF input resolution is increased from 392 to 518 with DINOv2-S dynamic_img_size IN FSC147 (otherwise identical N0010 stack), THEN val MAE ≤19.0 AND RMSE/MAE <3.4, BECAUSE finer token grid (+75% tokens) reduces small-object miss rate that dominates the 3.63× outlier tail, while preserving the validated H0017 multi-layer+40ep signal. DISPROVED IF MAE >21.53 (no gain over N0010 parent) OR RMSE/MAE ≥3.63.
- H0017 (reuse): continued evidence expected — multi-layer gated taps remain beneficial at higher resolution.

## Delta vs Parent
Parent N0010_dino_multilayer_long (21.53, 392px, 784 tokens). N0012 is an exact clone except `input_size`: 392→518 (28→37 patches/side, 784→1369 tokens). Backbone, scalar gate, prompt, adapter, head, loss (MSE+1.0·L1 count), epochs 40, lr 1e-3 all unchanged. Isolates the resolution lever; no confound with N0011's Huber/per-token changes.

## Novelty Statement
Not mechanistic novelty — deliberate ablation-style scale-up of the proven champion on its weakest axis (spatial resolution). Complements SeqCount's insight that patch granularity matters; tests the cheapest reliable path toward MAE<16 under the ≤32M budget.

## Estimated Params
~23.11M total (identical to N0010; resolution changes activation memory/tokens only). Frozen backbone ~21M + projs+prompt+adapter+head ~2.1M. Well under 32M.

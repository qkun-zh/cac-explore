# idea.md — N0014_dino_highres_augreg

## Title
Merge winners: highres 518 + augreg mosaic-lite on champion N0010 stack.

## Motivation & Intuition
N0010 champion 21.53 (frozen DINOv2-S reg4 dual taps scalar gate, 392px, 40ep, count-w1.0, wd1e-4, drop0.1). N0011 26.68 REFUTED (per-token+Huber, H0019/H0020 down) — do not build on it. N0012 highres518 running (same recipe 518px, ep5 best 29.43 early, still training); N0013 augreg queued (same 392px + photometric p0.5 + bbox 15% + wd5e-4 drop0.2, smoke green). Both levers mechanistically independent: resolution → spatial fidelity (+75% tokens, 784→1369) reduces small-object misses that drive RMSE/MAE=3.63×; augmentation → regularization prevents overfit after E26 and mimics SeqCount+ mosaic tail robustness (inspiration §7-8). Classic merge-winners step valid before siblings finish because levers are orthogonal — additive gain expected, no confound with N0011.

## Architecture Spec
- core_ideas:
  1. Frozen `vit_small_patch14_reg4_dinov2` features_only out_indices=(6,11) dynamic_img_size=True → [B,1369,384]×2 at 518 (37×37, 518%14==0).
  2. Per-layer Linear proj 384→384 + scalar layer_logits softmax gate (2 params) mixing t6/t11 → +Fourier+logArea prompt → adapter 384→768→384 GELU drop0.2 → conv head 384→128(1×1)→1(1×1) → density [B,1,37,37]; engine sum-conserves to GT.
  3. In-model augmentation (training-gated): tensor ColorJitter brightness0.2/contrast0.2/saturation0.15 + Gaussian σ0.02 with p0.5 on imgs + bbox jitter ±15% uniform scale/translate clamped [0,S] before prompt_enc — no GT warp.
- core_blocks: PromptEncoderV2, LayerGate(scalar), Adapter, Head — identical to N0010; only training-gated jitter ops added in forward.
- network_structure: imgs[518]--jitter?(train)--frozen taps{t6,t11}[1369,384]--proj--scalar-gate sum--+prompt(bbox-jittered)--adapter(d0.2)--head--density 37×37.
- tunable_aspects: input_size 518 (from 392); jitter prob/strength, bbox_jitter 0.15, dropout 0.1→0.2, wd 1e-4→5e-4; epochs 40, lr1e-3, count_w1.0 fixed.
- invariants: backbone frozen eval; total ≤32M (~23.11M, jitter adds 0 params); bbox [B,4]; S%14==0; dynamic_img_size.

## Proposed Hypotheses
- H0023: IF highres 518 + augreg (photometric p0.5 + bbox 0.15 + drop0.2/wd5e-4) is combined IN frozen DINOv2-S dual-tap scalar-gate 40ep stack, THEN val MAE ≤18.0 (-3.5 vs N0010 21.53) AND MAE ≤ min(N0012,N0013) AND RMSE/MAE <3.4, BECAUSE finer token grid reduces small-object miss tail while jitter+stronger regularization prevents overfit at higher token count — orthogonal levers stack additively. DISPROVED IF MAE >18.0 OR MAE > best sibling OR RMSE/MAE ≥3.63.
- H0021 (reuse): resolution lever persists — expect independent gain vs 392px at same regularization.
- H0022 (reuse): augreg lever persists — expect later best epoch (>26) and reduced overfit divergence vs N0010 at 518px.

## Delta vs Parent
Parent N0010_dino_multilayer_long (21.53, 392px, 784 tokens, drop0.1, wd1e-4, no jitter, 23.11M, 40ep). N0014 clones N0010 verbatim (same scalar gate, prompt, adapter, head, count-w1.0) and merges both orthogonal levers: (a) N0012 lever: input_size 392→518 (28→37 patches/side, 784→1369 tokens), batch 8→4 for 518 OOM safety; (b) N0013 lever: photometric jitter + bbox jitter 0.15 in `model.forward` (training-gated) + dropout 0.1→0.2 + wd 1e-4→5e-4. Zero confound with N0011 Huber/per-token (explicitly excluded).

## Novelty Statement
No new mechanism — deliberate additive ablation testing orthogonality of the two cheapest reliable levers under ≤32M/frozen-backbone budget. First CAC test combining resolution scale-up with in-model photometric+bbox regularization on the proven multi-tap DINOv2-S substrate; validates whether spatial fidelity and regularization stack linearly toward MAE<16.

## Estimated Params
~23.11M total (identical to N0010/N0012/N0013; resolution+jitter add 0 params: frozen ~21M + projs/prompt/adapter/head ~2.1M). Well under 32M. epochs 40, batch 4.

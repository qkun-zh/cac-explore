# idea.md — N0015_dino_highres672

## Title
Extreme high-resolution clone: identical N0010 recipe at 672px (48×48 tokens).

## Motivation & Intuition
N0010 is frozen-DINOv2-S champion (val MAE 21.53 best @E26/40, RMSE/MAE=3.63× — small-object miss tail dominates outliers). N0012 highres518 (37×37=1369 tokens, +75%) is running ep11 best 28.7 early — not yet beating champion but still early (N0010 needed >20ep to converge; plateau after E26). N0013 392+augreg and N0014 518+augreg are coded/queued. N0015 is contingency branch: if 518 succeeds → 672 tests ceiling (+194% vs 392, +68% vs 518); if 518 fails/OOM → data point on resolution scaling limit. Tiny objects <14px are sub-token at 392 (one patch=14px); 672 gives 48×48 grid where those become resolvable (≥1 token). Inspiration §§7-8: high-density long-tail is SOTA bottleneck (SeqCount+ via mosaic) and efficiency must stay controllable — 672 probes whether resolution alone resolves tail without augreg, at the 12GB memory boundary. DINOv2 dynamic_img_size=True makes 672 safe (672%14==0, native 518) per memory/failure_modes.md; BCHW, PATCH=14, Linear-on-tokens traps still apply.

## Architecture Spec
- core_ideas:
  1. Frozen vit_small_patch14_reg4_dinov2 features_only out_indices=(6,11) via timm dynamic_img_size=True → [B,2304,384]×2 at 672 (48×48, 672%14==0).
  2. Per-layer Linear proj 384→384 + scalar layer_logits softmax gate (2 params) mixing t6/t11 — identical to N0010.
  3. Same Fourier+log-area prompt (PromptEncoderV2), adapter 384→768→384 GELU drop0.1, conv head 384→128(1×1)→1(1×1) → density [B,1,48,48]; engine sum-conserves upsample to GT.
- core_blocks: No new blocks; only input_size changes vs N0010. Reuse N0010 model.py verbatim except cfg.
- network_structure:
  imgs[672]->frozen taps{t6,t11}[2304,384]->proj->scalar-gate sum->+prompt->adapter->head->density 48×48.
- tunable_aspects: input_size 672 (from 392); batch_size 2 (from 8) for 12GB OOM safety (672 needs ~2.9× memory of 392; 392×bs8≈672×bs2 by tokens). All else fixed (epochs 40, lr1e-3, wd1e-4, adapter_dim768, dropout0.1, count_w1.0).
- invariants: backbone frozen eval; total ≤32M (~23.11M same as N0010 — resolution adds no params); bbox [B,4]; input multiple of 14; dynamic_img_size=True.

## Proposed Hypotheses
- H0024: IF input resolution is increased from 392 to 672 with DINOv2-S dynamic_img_size IN FSC147 (otherwise identical N0010 stack, wd1e-4 drop0.1, no augreg), THEN val MAE ≤18.5 AND RMSE/MAE <3.4, BECAUSE finer token grid (784→2304, +194%) resolves tiny objects <14px that dominate N0010's 3.63× outlier tail, while preserving validated H0017 multi-layer+40ep signal. DISPROVED IF MAE >21.53 (no gain over N0010 parent) OR OOM/timeout at batch2 OR RMSE/MAE ≥3.63.
- H0017 (reuse): multi-layer gated taps remain beneficial at 672px — expect gate stays balanced, not collapsed.

## Delta vs Parent
Parent N0010_dino_multilayer_long (21.53, 392px, 28×28=784 tokens, bs8, drop0.1 wd1e-4). N0015 is exact clone except `input_size`: 392→672 (28→48 patches/side, 784→2304 tokens, +194%; vs N0012 37→48, +68%) and `batch_size` 8→2 for 12GB fit. Keeps wd1e-4 drop0.1 like champion (no augreg) to isolate resolution — strictly orthogonal to N0013/N0014 augreg levers and unconfounded with N0011 Huber/per-token. Contingency: isolates pure resolution ceiling at memory boundary.

## Novelty Statement
Not mechanistic novelty — deliberate extreme scale-up probing the spatial resolution ceiling and memory boundary of the proven champion. Complements N0012 518 test; informs MAE<16 path whether 518→672 gives linear gain or diminishing/OOM collapse under ≤32M frozen-backbone budget.

## Estimated Params
~23.11M total (identical to N0010/N0012; resolution changes activation memory/tokens only, not params). Frozen ~21M + projs+prompt+adapter+head ~2.1M. Well under 32M.

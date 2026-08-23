# idea.md — N0003_convnext_xattn

## Title
Frozen ConvNeXt-Nano pyramid + exemplar-conditioned cross-attention decoder (counting-as-detection lineage).

## Motivation & Intuition
Correlation maps are rigid: one similarity value per pixel, no context. CounTR (BMVC'22) showed cross-
attention between exemplar and image features beats plain matching; a cross-attention decoder can let a
small set of learnable "density queries" look at the whole image AND at the exemplar embedding, producing
spatially adaptive density bases whose mixture is exemplar-dependent. Conv features also give multi-scale
detail ViTs lack — important because FSC147 object sizes span two orders of magnitude.

## Architecture Spec
- core_ideas:
  1. Frozen timm `convnext_nano.in12k` (~15.6M) taps features at strides 8/16/32.
  2. Lightweight top-down FPN merge (~1M trainable) → single-scale map C=128 at stride 8.
  3. K=8 learnable queries + exemplar token cross-attend merged map (2 decoder layers); each query emits a
     basis map via 1×1 conv from its output embedding; softmax-mixed weights per query → final density.
- core_blocks:
  - FPN: lateral 1×1s to 128ch + top-down add + 3×3 smooth conv.
  - Exemplar token: RoI-align stride-8 map on bbox → masked mean → Linear(128→256).
  - Decoder: 2× TransformerDecoderLayer(d=256, heads=4) over flattened stride-16 tokens (memory-safe).
  - Basis head: per-query Linear(256→C) → dot with map tokens → K low-res maps → weighted sum.
- network_structure:
  imgs→frozen ConvNeXt{c3,c4,c5}→FPN→map[128,S/8]; tokens16[·,S/16²,256](proj from c4);
  queries[K,256]+exemplar_tok → cross-attn(tokens16) → w[K] = softmax(Linear(out));
  density = Σ_k w_k · Basis_k(map_tokens) reshaped [S/8,S/8].
- tunable_aspects: K (4/8/16); decoder layers/heads; FPN channels; stride of attention tokens; lr; aux count head.
- invariants: backbone frozen; total ≤32M (budget ~20M used); bbox [B,4] S-space; low-res density OK;
  attention token count ≤ (384/16)² = 576 for memory safety on 12GB.

## Proposed Hypotheses
- H0004: IF exemplar-conditioned cross-attention mixture-of-bases IN FSC147 beats plain cosine matching
  (N0002-style) under equal frozen-backbone budget, THEN MAE improves ≥10% relative, BECAUSE contextual
  attention resolves same-class distractors that pure feature similarity confuses.
  DISPROVED IF MAE ≥ N0002's val MAE × 0.9 when compared at equal epochs, or if N0002 unavailable,
  DISPROVED IF val MAE > 30 @ 10 epochs.
- H0005: IF stride-16 attention tokens (vs stride-8 full map) are used IN the decoder, THEN MAE parity holds
  with ≥2× step time reduction, BECAUSE counting needs global context more than fine alignment; the basis
  maps restore resolution. DISPROVED IF stride-16 variant loses >1.5 MAE vs stride-8 ablation.

## Delta vs Parent
None (gen-0 root). Independent branch: attention-based conditioning instead of direct matching; conv
multi-scale backbone instead of ViT single-scale.

## Novelty Statement
Frames CAC as exemplar-prompted dynamic convolution: a tiny DETR-style decoder predicts a per-image mixture
over learned density bases. To our knowledge mixing "query-set → basis-map weights" for class-agnostic
density regression is unexplored in the lightweight regime.

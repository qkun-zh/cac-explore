# idea.md — N0008_dino_xattn

## Title
Cross-attention basis-mixture decoder on DINOv2-S substrate (merging the two #1 mechanisms).

## Motivation & Intuition
The search has independently confirmed two mechanisms: DINOv2-S frozen substrate (N0007, -15.3%; H0014
supported) and exemplar-conditioned cross-attention mixture-of-bases (N0003, beat cosine matching by
18.5%; H0004 supported). Neither node combined them. N0007's remaining weakness is exactly what cross-attn
fixes: a per-token MLP head splits adjacent-instance mass crudely and has no neighbor context, which is
where the RMSE tail (3.4x MAE) lives. Porting the N0003 decoder onto DINOv2 784-token grids is the
highest-expected-value merge in the tree.

## Architecture Spec
- core_ideas:
  1. Frozen timm vit_small_patch14_reg4_dinov2.lvd142m (dynamic_img_size=True), input 392 ->
     patch tokens [B,784,384] (drop cls+reg prefix).
  2. Exemplar token: RoI-mean over projected patch features INSIDE bbox CONCAT area-aware Fourier prompt
     embedding (65d->MLP->256) -> exemplar token [B,256] (carries appearance AND scale).
  3. Decoder: 2x TransformerDecoderLayer(d=256, ffn=512, heads=4, norm_first) over all 784 tokens as
     memory; queries = 8 learnable + exemplar token.
  4. Basis mixture: each query emits Linear(256->384) basis vector dotted with token grid -> K=8 maps
     [B,8,28,28]; softmax gate per map; weighted sum -> density.
- core_blocks: ExemplarTokenV2, XAttnBasisDecoder (from N0003 lineage), backbone frozen eval-mode.
- network_structure:
  imgs[3,392,392]->frozen DINOv2->[B,784,384]->mem_proj->256d memory (+exemplar added);
  queries[K=8]+extok -> cross-attn -> weights & bases -> mixed density [B,1,28,28].
- tunable_aspects: K 8/16; decoder layers 1/2/3; dropout 0.1; lr; loss_count_weight 0.3/1.0;
  exemplar-token fusion type (concat vs add).
- invariants: backbone frozen eval; total <=32M (~23.5M est); bbox [B,4] S-space; low-res OK;
  input_size multiple of 14; dynamic_img_size=True.

## Proposed Hypotheses
- H0015: IF N0003's cross-attn basis-mixture decoder is ported onto the DINOv2-S substrate IN FSC147,
  THEN val MAE <= 25.5 (>=8% better than N0007's 27.65), BECAUSE contextual query mixing resolves
  adjacent-instance mass splitting that the per-token MLP head cannot.
  DISPROVED IF MAE > 25.5.

## Delta vs Parent
Parent N0007_dino_promptv2 (27.65): MLP mass head replaced by N0003-style cross-attn basis decoder;
exemplar token upgraded with area-aware Fourier prompt (H0011 lineage). Multi-layer DINOv2 taps and
loss_count_weight sweep deliberately deferred to siblings to isolate the mechanism variable.

## Novelty Statement
First combination of self-supervised ViT patch tokens with an exemplar-prompted dynamic basis-mixture
decoder for class-agnostic counting under a frozen-backbone budget — the direct product of the two
strongest confirmed hypotheses in the tree.

# idea.md — N0009_dino_xattn_stable

## Title
Stabilized retry of the DINOv2 x cross-attn merge (sibling of N0008, same parent N0007).

## Motivation & Intuition
N0008's decoder diverged for ~12 epochs at lr=1e-3 without warmup (DINOv2 token magnitudes + moving
memory under joint projection training), then recovered monotonically to 46.56 while still descending —
an optimization failure, not a mechanism ceiling. This node re-runs the identical architecture with a
stabilization recipe: lr 2.5e-4, single decoder layer, K=4 queries. A smaller decoder is easier to
optimize and its capacity was not the bottleneck at 30 epochs.

## Architecture Spec
Identical to N0008_dino_xattn/model.py (shared file copied): frozen vit_small_patch14_reg4_dinov2 @392,
784 tokens; exemplar token = RoI-mean appearance CONCAT area-aware Fourier prompt -> fuse MLP;
TransformerDecoder(d=256, ffn=512, heads=4, norm_first) over tokens+extok memory; K queries emit basis
vectors dotted with normalized token grid -> softmax-mixed density [B,1,28,28].

## Proposed Hypotheses
- H0016: IF the cross-attn basis merge is trained stably on DINOv2-S IN FSC147 (lr 2.5e-4, 1 layer, K=4),
  THEN val MAE <= 25.5, BECAUSE contextual basis mixing adds adjacent-instance modeling on top of N0007's
  substrate gain once optimization noise is removed. DISPROVED IF MAE > 27.65 (= parent; transfer refuted).

## Delta vs Parent
Parent N0007_dino_promptv2 (27.65): head swapped to cross-attn basis mixture; deltas vs failed sibling
N0008: lr 1e-3->2.5e-4, dec_layers 2->1, queries 8->4.

## Novelty Statement
Same as N0008 — this node exists to adjudicate H0015 cleanly under stable optimization.

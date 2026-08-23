# idea.md — N0006_swin_promptv2

## Title
Swin prompt-conditioning v2: exemplar-area prompt + two-scale gated fusion (gen-1 child of N0005).

## Motivation & Intuition
N0005 proved implicit prompt conditioning is the best root mechanism (32.66) and converges fastest
(271s). Two causal levers remain untested: (a) the prompt encodes box POSITION but not SIZE/magnitude —
CACViT showed normalized ViTs lose exactly that scale information, and all three N0005 reviewers flagged
scale as the top lever (Q_t exploitation pick H0011); (b) the 7x7 output grid is too coarse for FSC147's
small objects — a stride-16 + stride-32 dual-scale representation with a learned per-location gate brings
the multi-scale size-prior (H0006) into the winning architecture.

## Architecture Spec
- core_ideas:
  1. Frozen timm swin_tiny_patch4_window7_224.ms_in22k, features_only out_indices=(2,3):
     stage-2 [B,384,14,14] (stride16) + stage-3 [B,768,7,7] (stride32); handle channels-last BHWC.
  2. Prompt encoder v2: Fourier(cx,cy,w,h; 8 freqs)=64d CONCAT raw area term log(w*h/S^2) -> MLP 65->256->768
     (H0011: magnitude enters the conditioning signal itself).
  3. Trainable adapter on stage-3 tokens (768->384->384) as N0005; stage-2 projected by Conv1x1(384->384)+GELU.
  4. Two-scale gated fusion: upsample adapted s3 to 14x14, concat with proj s2 -> Conv1x1(768->2) softmax
     -> per-location weighted sum -> fused [B,384,14,14] (H0006-style scale gate inside this arch).
  5. Head: Conv1x1(384->128)->GELU->Conv1x1(128->1) -> mass [B,1,14,14] (2x finer than N0005).
- core_blocks: PromptEncoderV2, TokenAdapter, DualScaleGate, MassHead — all trainable, backbone frozen eval.
- network_structure:
  imgs[3,224,224]->frozen swin{s2[384,14,14], s3[768,7,7]}->s3 tokens adapter (prompt prepended);
  fuse gate(s2proj, up(s3adapted)) -> head -> density [B,1,14,14]; engine sum-conserving upsample.
- tunable_aspects: area-channel on/off (ablation child); gate temperature; adapter width 384/512;
  fusion order; epochs 30; eta_min 1e-5.
- invariants: backbone frozen eval-mode; total <=32M (~29.2M est); bbox [B,4] S-space; low-res density OK;
  input_size=224 (swin window constraint); channels-last handling for timm swin outputs.

## Proposed Hypotheses
- H0012: IF exemplar-area prompt + dual-scale gated fusion IN FSC147 (this node vs parent N0005),
  THEN val MAE <= 31.5 (>=3.5% better), BECAUSE scale info in the prompt plus finer multi-scale grid fix
  N0005's two identified bottlenecks (magnitude-blindness and 7x7 coarseness). DISPROVED IF MAE > 32.66.
- H0013: IF per-location scale gating is used at stride16/32 IN this architecture, THEN RMSE/MAE ratio
  drops below 3.0 (from ~3.1) at MAE parity-or-better, BECAUSE gating reallocates mass to correct scales,
  shrinking high-count tail errors. DISPROVED IF ratio >= 3.0 or MAE worsens.

## Delta vs Parent
Parent N0005_swin_promptseg (32.66). Adds: area channel to prompt (H0011), stage-2 tap + gated fusion +
14x14 output (H0006 flavor), longer schedule 30ep. Q_t items H0002/H0003 are cosine-map-specific (N0002
lineage) and NOT exercised here — left for a possible N0002-lineage child.

## Novelty Statement
First combination of SAM-style implicit prompt conditioning with explicit learned scale gating and a
size-aware prompt encoder for class-agnostic counting under a frozen-backbone budget.

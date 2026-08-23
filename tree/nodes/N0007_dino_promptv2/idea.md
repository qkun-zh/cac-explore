# idea.md — N0007_dino_promptv2

## Title
DINOv2-S substrate for the proven prompt-conditioning stack (gen-1 child of N0005).

## Motivation & Intuition
Cross-node causal verdict after 6 nodes: features > mechanism > schedule. DINOv2-S reg4 is the strongest
category-agnostic substrate available within budget (N0004/N0006 proved ImageNet-conv/hier features are
the bottleneck; CountingDINO showed DINOv2 localizes training-free). This node ports the confirmed stack —
Fourier+area prompt token, adapter, mass head — onto DINOv2-S at input 392: 28x28 patch tokens give 4x
finer resolution than N0005 AND better instance separability than swin stage-2. Dropout added to counter
the overfit pattern first seen in N0006.

## Architecture Spec
- core_ideas:
  1. Frozen timm vit_small_patch14_reg4_dinov2.lvd142m (dynamic_img_size=True), input 392 -> tokens
     [B,784,384] at stride 14.
  2. PromptEncoderV2 (from N0006): Fourier(cx,cy,w,h;8 freqs)=64d + log-area scalar -> MLP 65->256->384.
  3. Adapter: Linear(384->768)->GELU->Dropout(0.1)->Linear(768->384) on [prompt + 784 tokens].
  4. Head: Conv1x1(384->128)->GELU->Dropout(0.1)->Conv1x1(128->1) on patch tokens -> mass [B,1,28,28].
- core_blocks: PromptEncoderV2 (shared code), TokenAdapter+Dropout, MassHead — backbone frozen eval.
- network_structure:
  imgs[3,392,392]->frozen DINOv2-S->[B,789,384] take last 784->prepend prompt->adapter->head->
  density [B,1,28,28]; engine sum-conserving upsample to GT.
- tunable_aspects: adapter_dim 768/512/384; dropout 0.0/0.1/0.2; epochs 25; lr; area-channel ablation.
- invariants: backbone frozen eval-mode; total <=32M (~23.1M est); bbox [B,4] S-space; low-res OK;
  input_size multiple of 14; dynamic_img_size=True required (failure_modes).

## Proposed Hypotheses
- H0014: IF the proven prompt-conditioning stack runs on a frozen DINOv2-S reg4 substrate IN FSC147,
  THEN val MAE <= 29.0 (>=10% better than current best 32.10), BECAUSE self-supervised category-agnostic
  features supply instance separability that supervised-hierarchical features lack — the causal bottleneck
  identified in N0004 and N0006. DISPROVED IF MAE > 29.0.

## Delta vs Parent
Parent N0005_swin_promptseg (32.66): backbone swapped swin->DINOv2-S reg4, grid 7x7->28x28, adapter widened
to 768 with dropout 0.1, epochs 20->25, eta_min 1e-5, area-prompt carried over from N0006 lineage (H0012
direction confirmed, bar missed there). H0002/H0003 remain N0002-cosine-specific, not exercised here.

## Novelty Statement
First fusion of DINOv2 frozen patch tokens with exemplar-area-aware implicit prompt conditioning for CAC;
directly tests whether feature quality was the hidden ceiling of every gen-0 mechanism.

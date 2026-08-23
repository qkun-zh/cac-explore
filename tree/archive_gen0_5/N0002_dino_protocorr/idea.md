# idea.md — N0002_dino_protocorr

## Title
Frozen DINOv2-S patch embeddings + single-prototype cosine matching → density (correlation lineage).

## Motivation & Intuition
DINOv2 features are famously category-agnostic: same-class instances cluster in embedding space across
unseen categories. CAC needs exactly this. If we pool the exemplar-box patches into one prototype vector
and take its cosine similarity against all patch embeddings, the similarity map should already localize
target instances with zero training — CountingDINO (WACV'26) showed exactly this training-free with
DINOv2-L (+registers), and CACViT (AAAI'24) showed a plain pretrained ViT with extract-and-match reaches
~9 test MAE. A small trainable decoder then converts similarity into calibrated density under our ≤32M
budget (their models are 86M–300M+). This isolates the purest gen-0 question: is explicit frozen-feature
matching + tiny head enough?

## Architecture Spec
- core_ideas:
  1. Frozen timm `vit_small_patch14_reg4_dinov2.lvd142m` (~22M, register variant per "ViTs Need Registers";
     fallback `vit_small_patch14_dinov2.lvd142m`) extracts patch tokens at stride 14.
  2. Exemplar prototype = mean of projected tokens whose centers fall inside bbox (RoI pooling on grid).
  3. Density = f(cosine-sim map) via tiny conv decoder; engine upsamples low-res output sum-conserving.
- core_blocks:
  - `proj`: Linear(384→256) + LayerNorm (trainable adapter).
  - `decoder`: Conv3×3(1→32)→GELU→Conv3×3(32→32)→GELU→Conv3×3(32→1), upsample ×2 inside head.
- network_structure:
  imgs→frozen ViT→tokens[1+P,384]→drop CLS→proj→256d; proto=masked-mean(bbox); sim=cos(proto,tokens)∈[−1,1];
  sim_map[P] reshaped to grid (S/14)×(S/14) → decoder → density [B,1,S/7,S/7].
- tunable_aspects: proj dim (128/256/512); decoder depth/width; sim normalization (raw cos vs temperature τ);
  auxiliary count-MLP head on pooled global token; lr; loss_count_weight.
- invariants: backbone frozen (requires_grad=False, eval mode); total params ≤32M; single exemplar bbox [B,4]
  in S-space; density output may be any low-res grid (engine handles upsampling); input_size multiple of 14.

## Proposed Hypotheses
- H0001: IF density is produced by exemplar-prototype cosine matching over frozen DINOv2 tokens IN FSC147,
  THEN val MAE < 30 within 10 epochs, BECAUSE pretrained category-agnostic embeddings already separate
  instances so the head only learns calibration. DISPROVED IF val MAE ≥ 30 @ 10 epochs.
- H0002: IF a learnable temperature τ on the cosine map is added IN the same setting, THEN MAE drops ≥5%
  vs fixed raw cosine, BECAUSE sharpness control lets the decoder trade localization precision against
  noise. DISPROVED IF ΔMAE ∈ (−5%, +5%).
- H0003: IF an auxiliary MLP count head (global-token → scalar count, L1 loss) is co-trained, THEN main
  density MAE improves, BECAUSE count supervision regularizes the shared projection toward counting-relevant
  features. DISPROVED IF aux variant does not beat main by >0.3 MAE.
- H0011: IF CACViT-style scale/magnitude embeddings (exemplar box size + image magnitude codes injected into
  the decoder input) are added IN this node, THEN MAE improves ≥5%, BECAUSE normalized ViT tokens lose scale
  and order-of-magnitude info that density calibration needs (CACViT's stated motivation).
  DISPROVED IF ΔMAE ∈ (−5%, +5%).

## Delta vs Parent
None (gen-0 root). Derived from research_direction.md baseline note; grounded in CountingDINO / CACViT
findings, but shrinks the frozen encoder 86–300M → 22M and trains only a calibration head.

## Novelty Statement
CountingDINO is training-free with a 300M backbone; CACViT fine-tunes matching inside a pretrained ViT.
This root tests the minimal trained version: frozen 22M DINOv2-S + single cosine prototype map + tiny
calibration decoder — a deliberate ablation-floor and budget-compliant variant of the strongest known signal.

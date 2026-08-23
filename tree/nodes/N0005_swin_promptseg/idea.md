# idea.md — N0005_swin_promptseg

## Title
Frozen Swin-Tiny + exemplar prompt token, density as segmentation (implicit-conditioning lineage).

## Motivation & Intuition
All matching above makes conditioning EXPLICIT (similarity to exemplar features). CACViT (AAAI'24) showed a
plain pretrained ViT performs exemplar matching implicitly inside self-attention ("decoupled view") —
suggesting attention itself may suffice. SAM-style prompt tuning supports the same bet: feed the exemplar
box as a prompt token and let attention figure out correspondence. If implicit prompting matches explicit
matching, conditioning mechanisms matter less than backbone features — a fundamental gen-0 question.
Segmentation-view density also pairs naturally with frozen ViT-family token grids.

## Architecture Spec
- core_ideas:
  1. Frozen timm `swin_tiny_patch4_window7_224.ms_in22k` (~28.3M), input 224.
  2. Exemplar box → Fourier positional encoding of (cx,cy,w,h) → Linear → one prompt token prepended to
     the patch-token sequence before the trainable adapter projection.
  3. Density head = per-token MLP predicting mass logits; trained with normalized-density KL/BCE + count L1
     (segmentation view: GT density renormalized into per-pixel probabilities).
- core_blocks:
  - Adapter: Linear(768→384)+GELU+Linear(384→384) applied to patch tokens (trainable).
  - Prompt encoder: Fourier(8 freqs ×4 coords)→Linear(32→256→768)→token.
  - Head: Conv1×1(384→128)→GELU→Conv1×1(128→1); output grid S/32 reshaped to map.
- network_structure:
  imgs[3,224,224]→frozen Swin stages (features_only, 768ch @ S/32, 49 tokens)→prepend prompt_token→adapter
  →head→mass map [B,1,7,7]→engine upsamples sum-conserving; loss = KL(norm_gt ‖ softmax(mass))·λ_seg +
  MSE(count) + 0.3·L1(count).
- tunable_aspects: prompt encoding type (Fourier vs raw 4 coords); adapter depth; seg-loss weight λ_seg;
  whether prompt enters stage inputs at multiple depths instead of tokens only; lr.
- invariants: backbone frozen; total ≤32M (~30.5M used — tightest budget node, config asserts max_params_M=32);
  bbox [B,4] S-space normalized for the encoder; low-res density OK; input_size=224 (Swin window constraint).

## Proposed Hypotheses
- H0008: IF exemplar conditioning is delivered as an implicit prompt token IN FSC147 with a frozen Swin,
  THEN val MAE < 40 @ 10 epochs, BECAUSE pretrained global attention can route correspondence without an
  explicit similarity computation. DISPROVED IF val MAE ≥ 40 @ 10 epochs.
- H0009: IF implicit prompt conditioning is compared against N0002's explicit cosine matching at comparable
  budget, THEN implicit underperforms explicit by ≥10% relative MAE, BECAUSE single-box prompts give too
  weak a signal to localize small instances among distractors. DISPROVED IF gap ≤10% or implicit wins.
- H0010: IF segmentation-style normalized-density loss replaces pure MSE IN this node, THEN optimization is
  more stable (lower epoch-to-epoch MAE variance) at equal MAE, BECAUSE per-pixel normalization prevents
  loss domination by high-count images. DISPROVED IF variance not reduced or MAE worsens >5%.

## Delta vs Parent
None (gen-0 root). Independent branch: implicit conditioning + segmentation-view training objective.

## Novelty Statement
Imports SAM-style prompt-token conditioning into class-agnostic counting as a controlled counterfactual to
explicit matching, paired with a segmentation-view loss — testing whether CAC needs its classic correlation
inductive bias at all in the frozen-backbone era.

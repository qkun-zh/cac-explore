# idea.md — N0004_effnet_pyrmatch

## Title
Frozen EfficientNet-B0 multi-scale exemplar matching with learned scale gating (tiny-model lineage).

## Motivation & Intuition
FSC147's hardest failure mode is extreme size variation (same class, 10×–100× pixel area). Matching at a
single stride fails for either end; CACViT explicitly adds scale/magnitude embeddings for this reason, and
GeCo2 (2025) calls scale generalization the core open problem. A frozen small conv pyramid matched at EVERY
scale, combined by a per-location learned softmax over scales, lets the network pick "which zoom" each
image region is at — explicitly encoding the size-prior into the architecture. Also probes the budget axis:
can a 5M backbone compete with 22M ViTs?

## Architecture Spec
- core_ideas:
  1. Frozen timm `efficientnet_b0.ra4_in1k` (~5.3M) taps strides 8/16/32.
  2. Exemplar ROI pooled per scale → per-scale prototype → per-scale cosine maps.
  3. Per-location softmax gate over the 3 similarity channels (1×1 conv logits) → scale-selected matching
     map → conv decoder → density.
- core_blocks:
  - Per-scale proj: Conv1×1(c→64)+BN+GELU per level (c = 112/320/1280 → 64).
  - Gate: Conv1×1(192→3) on concatenated sims; sim_final = Σ softmax(gate)_s · sim_s.
  - Decoder: Conv3×3(1→48)→GELU→Conv3×3(48→48)→GELU→Conv3×3(48→1), ×4 upsample to stride-8 output.
- network_structure:
  imgs→frozen EffNet{P8[64ch? use c2=24→proj],P16,P32}→projs 64ch each; proto_s=RoI-mean(bbox@stride_s);
  sim_s=cos(proto_s,tokens_s); upsample sims to stride-8 grid; concat[192]→gate→weighted sum→decoder
  → density [B,1,S/8,S/8].
- tunable_aspects: which taps feed the gate (2 vs 3 levels); proj width 32/64; temperature on sims;
  decoder capacity; lr; loss_count_weight; whether prototypes are stop-gradient or trained-through.
- invariants: backbone frozen; total ≤12M (deliberately far under 32M); bbox [B,4] S-space; low-res density OK;
  input_size multiple of 32.

## Proposed Hypotheses
- H0006: IF multi-scale gated matching IN FSC147 is used, THEN it beats any single-scale variant by ≥8%
  relative MAE, BECAUSE per-location scale gating absorbs intra-class size variance that fixed-stride
  matching cannot. DISPROVED IF best single-scale ablation ties or beats gated (Δ ≤ 8%).
- H0007: IF a ≤12M-total frozen-EffNet model IN FSC147 reaches val MAE < 35 @ 10 epochs, THEN backbone scale
  is not the bottleneck at this stage, BECAUSE pretrained conv features already carry sufficient category-
  agnostic structure. DISPROVED IF MAE ≥ 35 (small-backbone hypothesis refuted for now).

## Delta vs Parent
None (gen-0 root). Independent branch: smallest viable backbone + explicit multi-scale size handling.

## Novelty Statement
Scale-gated exemplar matching treats object-size selection as a first-class learned spatial gate rather than
an implicit property of a single feature stride — cheap, interpretable (gate maps visualize chosen zoom),
and untested in the frozen-tiny-backbone CAC regime.

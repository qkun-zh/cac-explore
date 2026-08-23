# feedback/causal.md — N0004_effnet_pyrmatch

## reasoning
Why 40.37 despite multi-scale? Causal decomposition: (1) backbone features dominate — EffNet-B0 ImageNet features lack the category-agnostic clustering that made DINOv2 work training-free (CountingDINO evidence); (2) explicit per-scale cosine inherits N0002's calibration weakness AND adds gate-learning burden; (3) plateau at ~E14 = head exhausted extractable signal from 3.65M-feature space. Cross-node causal ranking now has evidence: conditioning mechanism (cross-attn >> prototype-cosine) interacts strongly with feature quality (DINOv2 >> EffNet-IN); scale handling alone did NOT rescue a weak backbone.

## actionable_feedback
- Future children should combine winners: DINOv2-S frozen + cross-attn + multi-scale taps (strides from different ViT stages) + magnitude embedding (H0011).
- Drop pure-EffNet lineage unless testing compute-constrained deployment specifically.

## hypothesis_updates
- H0006: neutral, strength 0.30 (confounded; mechanism plausible, features insufficient).
- H0007: contradicts, strength 0.80 — consistent across all three reviewers.

# idea.md — N0021_dino_fullft

## Title
Full backbone fine-tuning: unlock instance-level feature learning.

## Motivation & Intuition
Frozen-backbone ceiling identified at 21.53: discrimination solved by DINOv2 semantics, but separation
requires instance-boundary features that only emerge through task-specific fine-tuning. All 9 post-champion
variants failed because they tried to fix separation/calibration with HEAD-ONLY capacity while the bottleneck
was in FEATURES. Removing the frozen constraint lets DINOv2 learn FSC147's visual domain: small-object
boundaries, intra-class variation, background confusion patterns.

## Architecture Spec
Champion recipe verbatim (N0010): frozen→FINE-TUNED DINOv2-S reg4 taps(6,11), gate, area-prompt,
adapter(384→768→384), MLP head. Differential LR: backbone=lr×0.1, rest=lr.

## Proposed Hypotheses
H0030: IF backbone is fine-tuned with 10× lower LR IN FSC147 THEN MAE ≤16.5 BECAUSE instance-boundary
features emerge that frozen self-supervised features cannot express.
DISPROVED IF MAE >19.0 (= no meaningful gain from fine-tuning).

# feedback/qualitative.md — N0006_swin_promptv2

## reasoning
Area channel entered the prompt smoothly (no instability); gate trained without collapse (softmax stayed informative per loss curve). But swin stage-2 stride16 features add resolution without adding instance-level discriminability — ImageNet-supervised hierarchical features blur individual small objects exactly where FSC147 needs them; the 14x14 grid helps placement, not separation. Overfit pattern (train loss 60x drop, val flat) says the head memorized train-category quirks; category-agnostic generalization is the binding constraint now.

## actionable_feedback
- Swap features to DINOv2-S reg4 (proven category-agnostic; CountingDINO evidence) while KEEPING prompt+area mechanism — single highest-upside move.
- Add dropout (0.1) on adapter/head or weight-decay bump to counter overfit in next child.

## hypothesis_updates
- H0012: contradicts, strength 0.50. Mechanism reading: right ingredients, wrong feature substrate for the remaining gap.

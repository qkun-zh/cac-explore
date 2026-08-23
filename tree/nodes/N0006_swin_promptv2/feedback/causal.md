# feedback/causal.md — N0006_swin_promptv2

## reasoning
Why only −1.7%? Causal decomposition: (1) area-in-prompt supplies scale info to conditioning, but density CALIBRATION also depends on feature scale-sensitivity — swin tokens remain normalization-blind regardless of prompt; (2) dual-scale fusion improves placement (RMSE slightly down) yet cannot create instance separability that isn't in the features; (3) overfit onset at ~E14 means the model exhausted generalizable signal in swin-IN22k features for 3659 train images. Cross-node synthesis: features > mechanism > schedule at this stage. The confirmed stack is prompt-conditioning + frozen strong backbone; the untested high-upside lever is DINOv2-S as the substrate with everything else kept.

## actionable_feedback
- Gen-1 priority: N0007 = DINOv2-S reg4 @392 (28x28 tokens) + Fourier-area prompt + adapter + mass head; epochs 25; dropout 0.1.
- Book overfit risk into failure_modes: children must compare train-vs-val gap, not just val MAE.

## hypothesis_updates
- H0012: contradicts, strength 0.55. H0013: contradicts, strength 0.45. Both improvements real-but-sub-bar; levers stay in pool for DINOv2 substrate child.

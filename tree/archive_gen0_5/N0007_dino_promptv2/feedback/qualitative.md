# feedback/qualitative.md — N0007_dino_promptv2

## reasoning
28x28 DINOv2 tokens give the mass head 16x more spatial resolution than N0005's swin grid and instance-level clustering that separates same-class objects (visible in how quickly MAE dropped vs all conv/hier substrates). Area-aware prompt worked without instability. Residual weaknesses: (1) single-layer token readout ignores DINOv2's known layer-specialization (mid layers carry best correspondences; final layer more semantic-global); (2) pure per-token MLP head cannot model neighbor context — adjacent-instance mass splitting is crude; (3) count-scale calibration still learned implicitly from MSE on upsampled maps.

## actionable_feedback
- Read out mid+final layers (blocks 6 & 12 of 12) concatenated or gated — cheap, likely free gain.
- Replace MLP head with tiny conv context block (3x3) or the N0003 cross-attn mixture for neighbor modeling.
- Consider input 448/518 (dynamic_img_size handles it): 32x32 tokens, ~1.6x step cost, fits budget at ~20ep.

## hypothesis_updates
- H0014: supports, strength 0.80. Architecture reading concurs: substrate was the ceiling.

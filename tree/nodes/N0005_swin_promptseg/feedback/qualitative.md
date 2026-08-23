# feedback/qualitative.md — N0005_swin_promptseg

## reasoning
Mechanism reading: frozen swin_tiny stage-3 tokens [B,49,768] channels-last handled; Fourier box encoding (cx,cy,w,h × 8 sin/cos freqs) → MLP → single prompt token prepended; trainable adapter (768→384→384) processes [prompt+patches]; 1×1 conv head on patch tokens → 7×7 mass map. Qualitative strengths: conditioning is GLOBAL from layer-0 of the adapter — every patch token sees the exemplar via shared processing rather than per-pixel similarity; cheap and stable. Weaknesses: 49 tokens at stride 32 is coarse for tiny FSC147 objects (7×7 grid!); single global mass map cannot express size variation within an image; smoke showed eval MAE swings between epochs (19.8→173 on synthetic) indicating count-scale brittleness early in training.

## actionable_feedback
- Take multiple swin stages (stride 8/16 too) or switch to DINOv2-S tokens at stride 14 with same prompt mechanism — resolution is the binding constraint, not conditioning.
- Add exemplar AREA channel to Fourier vector (H0011).
- Stabilize count scale: initialize head bias to log-average count density or add count-L1 warmup.

## hypothesis_updates
- H0008: supports, strength 0.75. Attention routing works without explicit similarity computation.
- H0009: contradicts, strength 0.70. Architecture reading concurs: global implicit conditioning suffices when features are strong.

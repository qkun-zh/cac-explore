# feedback/qualitative.md — N0004_effnet_pyrmatch

## reasoning
Mechanism reading: per-scale RoI prototypes → cosine sims upsampled to stride-8 → per-location softmax gate mixes scales → conv decoder. Qualitatively sound size-prior design, but two structural weaknesses: (1) cosine similarity of ImageNet-supervised EffNet features is far less instance-discriminative than DINOv2's self-supervised features — matching quality ceiling set by features, not gating; (2) gate sees ONLY the three scalar sim channels (3→3 conv) — no appearance context to decide "small dense cluster vs large sparse", so gating likely degenerates toward global scale preference.

## actionable_feedback
- If revisited: condition gate on projected features (64ch each) not just sims; add temperature.
- Prefer DINOv2 multi-layer taps for any future multi-scale matching child — keep the gate idea, upgrade the features.

## hypothesis_updates
- H0006: neutral, strength 0.20. Design is plausible but feature ceiling masks the gating effect.
- H0007: contradicts, strength 0.70. Feature-quality/scale bottleneck visible in plateau shape.

# feedback/causal.md — N0003_convnext_xattn

## reasoning
Why does cross-attn beat prototype matching (34.26 vs 42.05)? Causal chain: (1) explicit cosine forces ONE similarity value per pixel — all calibration must flow through a scalar; cross-attn lets queries gather context (where similar things are, what the background looks like) before committing to a density pattern. (2) The exemplar-conditioned memory lets every location attend to the exemplar AND to other locations — implementing soft "is this region like the exemplar's neighborhood" rather than "is this patch vector-close". (3) Multi-scale FPN supplies size evidence absent in ViT-S tokens. Remaining bottleneck: count-scale calibration still slow to learn (early epochs dominated by scale adaptation) → magnitude embedding (H0011) should specifically accelerate this.

## actionable_feedback
- Child priority: inject exemplar area/magnitude into ex_tok (tests H0011 mechanistically where it matters).
- Longer schedule + warmup; expect sub-30.
- Keep convnext_nano: 16.93M total leaves 15M headroom for a heavier decoder if needed.

## hypothesis_updates
- H0004: supports, strength 0.80. Mechanistic account consistent with observed gap and learning-speed difference.
- H0011: neutral, strength 0.0, but causal analysis upgrades predicted importance for children.

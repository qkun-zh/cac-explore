# feedback/qualitative.md — N0003_convnext_xattn

## reasoning
Mechanism reading: K=8 queries + exemplar token cross-attend stride-16 memory (conditioned by adding ex_tok), each query emits a channel-basis vector dotted with the stride-8 FPN map → softmax-mixed density. Qualitative strengths: (1) exemplar token enters BOTH query set and memory → bidirectional conditioning, unlike N0002's one-way similarity; (2) mixture weights are image-adaptive — different scenes can weight different bases; (3) multi-scale FPN gives conv detail ViT lacked. Weaknesses: basis maps share ONE channel space so bases may collapse to similar patterns (no diversity regularizer); exemplar is a single mean token again (same tiny-bbox fallback issue as N0002); gate softmax over K has no temperature control.

## actionable_feedback
- Add basis-diversity regularizer (orthogonality on out[:, :-1]) or K learned scales instead of free bases.
- Exemplar token from RoI-align multi-level (c3+c4) instead of single stride-8 mean.
- Feed box area into ex_proj input (scale cue) — cheap H0011-style test in this architecture.

## hypothesis_updates
- H0004: supports, strength 0.70. Architecture reading agrees: adaptive contextual mixing resolves distractors that fixed cosine cannot.
- H0011 (cross-ref): neutral, strength 0.0 — not implemented here either, but same prediction applies.

# feedback/qualitative.md — N0009_dino_xattn_stable

## reasoning
Why does contextual mixing LOSE to a plain head here? DINOv2 patch tokens are already so instance-discriminative that density placement is nearly per-token classification; a softmax mixture over K basis vectors constrains output to a low-rank subspace of token-similarity patterns, discarding capacity the MLP head uses directly. Cross-attn helped ConvNeXt features (N0003) because those needed contextual disambiguation; self-supervised ViT tokens already did that work.

## actionable_feedback
- If revisiting attention, use it for exemplar conditioning only (implicit prompt), not output decoding.

## hypothesis_updates
- H0016: contradicts, strength 0.70.

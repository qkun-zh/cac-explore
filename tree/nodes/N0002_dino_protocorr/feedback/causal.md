# feedback/causal.md — N0002_dino_protocorr

## reasoning
Causal chain for MAE 42: frozen DINOv2 features ARE transferable (CountingDINO shows training-free localization), so bottleneck is NOT backbone. Bottleneck is conditioning→density mapping: (a) cosine similarity is bounded [-1,1] and temperature-scaled, but its distribution shifts per image/category; a shallow decoder without per-image normalization or scale context cannot map varying similarity histograms to absolute counts. (b) Single prototype discards exemplar shape/texture; FSC147 intra-class variance makes one vector insufficient → similarity map has false positives/negatives, decoder has no way to disambiguate. (c) No gradient through backbone means no adaptation to counting task; H0001 assumed calibration suffices, but calibration needs more capacity than 0.1M decoder on 784 tokens.

## actionable_feedback
- Next child: keep frozen backbone but widen conditioning bandwidth — N0003 cross-attention (already proposed) directly tests causal alternative; prioritize it.
- For N0002 lineage child: add scale-gated multi-scale matching (borrow N0004 idea) or at least exemplar-area channel; compare to parent to isolate scale cause.
- Extend training to 30ep and log per-image count scatter; if high-count tail persists, the cause is scale, not optimization.

## hypothesis_updates
- H0001: contradicts, strength 0.80. Mechanism predicted pure calibration suffices — evidence shows systematic under-capacity at same backbone scale. DISPROVED.
- H0004 (belongs to N0003 but relevant): supports, strength 0.35 (indirect). N0002's failure pattern (RMSE 2.9×MAE) is consistent with scale-variation hypothesis that cross-attn/multi-scale should help.
- H0006/H0007: neutral, strength 0.0 — not tested in this node but causal analysis predicts multi-scale will matter.

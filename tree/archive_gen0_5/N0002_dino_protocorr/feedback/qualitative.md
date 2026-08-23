# feedback/qualitative.md — N0002_dino_protocorr

## reasoning
No images saved; infer from code+log. Architecture isolates purest matching signal: frozen DINOv2-S patch tokens (28×28 grid) → RoI-mean prototype (masked mean inside bbox) → cosine map → softplus(τ) → 3×Conv decoder (1→32→32→1) → ×2 upsample. Qualitative risks: (1) single mean prototype collapses exemplar spatial structure & fails when bbox <1 patch (empty mask → fallback uniform mean → prototype becomes global mean, systematic error on tiny objects which dominate FSC147); (2) decoder sees ONLY scalar similarity, no raw appearance or scale cues, so count calibration must be learned from similarity statistics alone; (3) no scale/magnitude embedding despite CACViT evidence that normalized ViT tokens lose size info; (4) monotonic training curve suggests head is learning but under-capacity — final loss still 7.7 vs GT density MSE scale ~0.01.

## actionable_feedback
- Replace mean prototype with attention pooling or 2–4 prototypes (e.g., masked average + max) to retain intra-exemplar variance.
- Feed exemplar area (w·h/S²) and image magnitude as extra channels into decoder (tests H0011 cheaply).
- Increase decoder to 4 convs with BN and skip from projected tokens; or add a parallel raw-feature branch.
- Guard tiny-bbox case: if inside.sum==0, RoI-align a 1-patch crop instead of uniform fallback.

## hypothesis_updates
- H0001: contradicts, strength 0.60. Qualitative reading supports quant: single cosine map is informative but not sufficient — decoder cannot denoise ambiguous similarity without context.
- H0011: neutral, strength 0.0. Not exercised, but code inspection predicts H0011 should matter for size-varying FSC147.

# feedback/causal.md — N0005_swin_promptseg

## reasoning
Why does implicit prompting beat explicit matching here? Causal chain: (1) explicit cosine forces correspondence THROUGH a bottleneck scalar map computed BEFORE any context mixing; implicit conditioning lets the pretrained attention machinery — trained on vast data — route exemplar information itself; (2) Swin ms_in22k features are strong enough (unlike EffNet-B0, see N0004) for this routing to work; (3) coarse 7×7 output limits localization but counting integrates over mass, so coarseness costs less than wrong correspondence; (4) fast convergence suggests pretrained features needed only light recalibration. Combined cross-node causal picture: feature quality (DINOv2/Swin-IN22k >> EffNet-B0) is necessary, conditioning adaptivity (prompt/cross-attn >> fixed cosine) amplifies it, and explicit similarity is NOT required.

## actionable_feedback
- Gen-1 priority child: DINOv2-S reg4 + Fourier-prompt token + adapter + multi-stage taps (stride 14 base) + box-area channel — merges all confirmed winners and tests H0011 where it belongs.
- Second child: N0005 + 30ep + lower eta_min + head-bias init for count scale (cheap robustness).

## hypothesis_updates
- H0008: supports, strength 0.80.
- H0009: contradicts, strength 0.75. Causal account consistent across reviewers.
- H0011: neutral strength 0.0 but causally flagged as top-priority untested lever (all three reviewers converge here).

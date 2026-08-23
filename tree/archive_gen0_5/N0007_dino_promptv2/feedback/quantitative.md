# feedback/quantitative.md — N0007_dino_promptv2

## reasoning
Best val MAE 27.65 / RMSE 93.24 @25ep/768s, 22.81M. −15.3% vs previous best (N0006 32.10); H0014's <=29.0 bar PASSED at E8 already (29.20) and E14 hit 27.65. Fastest quality-per-epoch of all nodes: E3 already beats every gen-0 root. Plateau-ish after E14 (~27.7-28.4 oscillation) while train loss fell to 1.24 — mild overfit again but much later than N0006; dropout+stronger features delayed it. RMSE/MAE ≈ 3.4 — tail errors remain the dominant residual.

## actionable_feedback
- Two chained runs (resume from best.pth) or epochs=40 fit τ_max=1800s? No — 25ep used 768s; 40ep ≈ 1230s fits ONE run. Try epochs=40 + lower eta_min.
- Merge with N0003's cross-attn mixture decoder on DINOv2 tokens (both #1 mechanisms independently).
- loss_count_weight sweep (0.3/1.0/2.0) targets the RMSE tail directly.

## hypothesis_updates
- H0014: supports, strength 0.90. Clean pass of pre-registered bar; causal prediction (feature ceiling) confirmed quantitatively.

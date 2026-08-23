# feedback/quantitative.md — N0009_dino_xattn_stable

## reasoning
Best val MAE 30.67 / RMSE 95.03 @30ep/985s. Optimization fixed (smooth descent, no divergence — lr 2.5e-4 + 1 layer + K=4 worked), but 30.67 > parent 27.65 → pre-registered rule REFUTES the transfer: cross-attn basis mixing does not beat the per-token MLP head on DINOv2 tokens even trained stably. Plateaued after E26.

## actionable_feedback
- Close the cross-attn-on-ViT-tokens branch; N0007's adapter+MLP head is the reference head.
- Redirect budget to: longer N0007 schedule (40ep fits one run), multi-layer taps, higher res, count-weight tuning.

## hypothesis_updates
- H0016: contradicts, strength 0.75.
- H0015 (cross-ref): contradicts, strength 0.60 — confound resolved, refutation stands at mechanism level.

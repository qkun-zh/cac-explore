# Causal Feedback — N0011_dino_pertok_gate_huber

## reasoning
**Performance delta**: Parent N0010 best MAE 21.53 (final 22.61, RMSE 75.9/82.0, params 23.11M) → N0011 best 26.68 (final 26.89, RMSE 93.93/94.34, params 23.16M) Δ=+5.15 (+24% worse). Triple confound prevents clean attribution: (a) gate, (b) loss, (c) count-w.

**Gate mechanism - capacity vs signal**: N0010 `model.py:44` scalar `layer_logits` (2 params, global softmax) is a strong regularizer. N0011 `model.py:30-42` replaces it with `PerTokenGateMLP` 768→64→2 ≈49k params (49152+130, +0.05M), conditioned per-patch on noisy concatenated `[z6;z11]` tokens. With frozen DINOv2, per-patch features have high spatial variance; shared MLP can memorize training patch patterns rather than learning transferable depth preference. Val trajectory never approaches parent (E13 28.33 → best E34 26.68, late peak vs parent E26), consistent with overfitting from excess local capacity.

**Huber mechanism - suppressing the signal**: `config.py:15-16` Huber δ=5.0 caps gradients for |error|>5. In CAC outliers ARE the MAE signal: high-count tail drives MAE/RMSE. Suppressing their gradient stabilizes RMSE (ratio 3.52 unchanged: 93.93/26.68 vs parent 75.9/21.53) but hurts val MAE — model under-fits the tail it must get right. Train loss scale incomparable (N0011 4.99→3.44 vs N0010 14.3→7.8) masks this.

**Count-w confound**: `config.py:10` reverts 1.0→0.3 (N0010's only proven gain lever), reducing L1 count supervision 3.3×. We cannot tell if regression is gate, Huber, or weaker count gradient.

## actionable_feedback
1. **Isolate variables**: next children must be single-edit ablations from N0010: (A) per-token gate only + MSE + w1.0; (B) Huber only + scalar gate + w1.0; (C) w0.3 only. No compound edits.
2. **Prefer low-capacity fusion**: revert to scalar gate or test regularized per-token (hidden 8-16, dropout 0.3, or spatial-channel attention pooling) if spatial adaptivity is pursued.
3. **Keep MSE for CAC**: Huber δ=5 misaligned; if tail robustness needed, try log-count loss or sample reweighting that preserves outlier gradients, not caps them.
4. **Hold count-w=1.0** as N0010 proven; do not revert without isolated evidence.

## hypothesis_updates
- **H0019** (per-token gate → MAE -1.0): **contradicts**, strength 0.80. MAE +5.15 with compound confound; gate added 49k noisy-conditioned params, best never beat scalar 2-param baseline. `model.py:61,84`.
- **H0020** (Huber → RMSE/MAE <3.0): **contradicts**, strength 0.75. Ratio 3.52 (93.93/26.68) unchanged vs 3.52 parent; MAE worsened significantly. `config.py:15`.
- **H0018** (count-w=1.0 tail): **neutral→supports weakly** indirectly — reverting to 0.3 coincided with +24% MAE regression, but confounded; needs isolated w-ablation.

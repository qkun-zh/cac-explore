# Causal Feedback — N0010_dino_multilayer_long

## reasoning

**Performance delta**: Parent N0007 val MAE 27.65 → N0010 best 21.531 (Δ=-6.12, 22% improvement). Final-epoch MAE 22.61 still beats parent by 5.04.

**Mechanistic link — multi-layer taps**: N0010 taps DINOv2 blocks 6 and 11 (mid + final), projects each through separate Linear layers, and fuses via softmax-gated sum. The mid-layer block 6 carries richer local correspondence features (texture, part boundaries) while block 11 carries global semantic features. The learned gate (`layer_logits` initialized to zeros → near-equal weighting initially) allows the adapter to see both spatial scales simultaneously. This doubles the token information flowing into the adapter from 784 tokens at one feature level to 784 tokens fused from two levels. The adapter (384→768→384) and MLP head remain structurally identical to N0007.

**Mechanistic link — count-w=1.0 vs 0.3**: The loss formula is MSE + count_w·L1. Tripling count_w from 0.3 to 1.0 amplifies the L1 count penalty, directly penalizing absolute count error more strongly. This should reduce under/over-counting especially for high-count samples. However, the RMSE/MAE ratio of 3.63× suggests catastrophic outlier failures persist — the L1 term may not sufficiently penalize large individual errors compared to MSE.

**Mechanistic link — 40ep vs 25ep**: Training showed continuous improvement through epoch 26 (best MAE 21.531) then plateaued/slightly degraded. The parent's 25ep schedule likely stopped before convergence. The extra 15 epochs allowed the model to reach a deeper local minimum. However, train loss continued dropping (14.3→7.8) while val stalled at ~22-23, indicating overfitting after ~epoch 26.

**Epoch trajectory analysis** (key epochs):
- E13: 25.50 → E15: 24.32 → E21: 23.70 → E23: 23.06 → E26: 21.53 (BEST) → E40: 22.61
- Best occurs at 65% of training (26/40), then val oscillates 21.5-24.1 — classic overfitting but early-stopping would catch it
- The step from 23.06→21.53 (E23→E26) is a 1.5 MAE drop in 3 epochs, suggesting the model found a sharper minimum

**Confound assessment**: Three variables changed simultaneously (architecture, epochs, loss weight). The architecture change (multi-layer taps) is the primary structural novelty — it adds representational capacity. The epochs and loss weight are hyperparameter scale-ups. The 22% gain is large enough that architecture likely contributed meaningfully, but we cannot cleanly attribute the gain. An ablation isolating each change would be needed for certainty.

## actionable_feedback

1. **Multi-layer taps are validated**: The layer-gate fusion of mid+final DINOv2 features is the structural innovation and produced the largest single-architecture gain in the tree so far. Next nodes should build on this by exploring different block pairs (e.g., blocks 4+8, 6+8+11 for three-way taps) or making the gate per-token (attention-based) rather than a global scalar.

2. **Overfitting is the binding constraint**: Train loss dropped 46% while val plateaued. Two levers: (a) reduce epochs to ~26-30 with early stopping on val MAE, freeing wall-clock for other experiments; (b) add regularization — increase dropout from 0.1 to 0.15-0.2, or add weight decay specifically to the adapter/head (currently only global wd=1e-4).

3. **RMSE/MAE = 3.63× signals outlier vulnerability**: The model handles typical samples well but fails catastrophically on some. The count-w=1.0 change did not fix this. Consider: (a) Huber loss or smooth-L1 instead of MSE to cap outlier gradients; (b) explicit sample-weighting to downweight high-count training samples; (c) training on log-counts to compress the dynamic range.

4. **The gate is undertrained**: `layer_logits` starts at zeros (equal weights) and is the only new parameter in the gate. With only 2 scalar parameters, it converges quickly but may lock into a suboptimal ratio. Test per-token gating (2-layer MLP over concatenated z6/z11 features → 784 scalar gates) to let different spatial regions choose different layer blends.

5. **Ablation needed**: Run two minimal children from N0010: (a) same architecture, 26ep (to isolate epoch effect); (b) same 40ep, count-w=1.0, but single-layer tap at block 11 only (to isolate architecture effect). These cost ~25min each and would resolve the confound.

## hypothesis_updates

- **H0017** (multi-layer taps + 40ep + count-w=1.0 ≤ 26.0 MAE): **supports**, strength 0.9. MAE 21.531 << 26.0 threshold. Hypothesis confirmed. However the 3 confounded changes mean the mechanism attribution is uncertain — the hypothesis as stated (compound effect) is supported, but the individual contributions are not isolated.

- **New hypothesis**: IF per-token layer-gating replaces scalar gating IN multi-layer DINOv2 tap architectures, THEN val MAE further decreases by ≥1.0, BECAUSE different spatial regions (small objects vs. background vs. crowd patches) benefit from different feature depths, and a scalar gate forces a global compromise. DISPROVED IF MAE increase or no change. Confidence η=0.20, init 0.5.

- **New hypothesis**: IF Huber loss (δ=5.0) replaces MSE in the counting head loss IN multi-tap DINOv2 architectures, THEN RMSE/MAE ratio drops below 3.0, BECAUSE Huber loss caps gradient magnitude for large errors, reducing catastrophic outlier failures without sacrificing median-case accuracy. DISPROVED IF RMSE/MAE > 3.5. Confidence η=0.20, init 0.5.

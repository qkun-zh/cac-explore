# synthesis.md — N0010_dino_multilayer_long

## Decision: PARTIAL SUPPORT [quantitative, causal]

N0010 val MAE 21.531 (best @E26), parent N0007 27.65, Δ=-6.12 (22% improvement). The compound
recipe (multi-layer taps + 40ep + count-w=1.0) decisively beat the parent and the MAE≤26.0 bar.
However, three simultaneous changes confound attribution — we cannot isolate which lever drove
the gain. Final-epoch MAE 22.61 still beats parent by 5.04, confirming the improvement is
robust, not a lucky checkpoint.

## Hypothesis Resolutions

**H0017 (multi-tap + 40ep + count-w=1.0 → MAE ≤ 26.0): SUPPORTED** [quantitative].
Best MAE 21.531 << 26.0 bar; passed decisively at E26. The compound effect is real.
Mechanism attribution uncertain due to confounded design (causal).

**H0018 (count-w=1.0 improves accuracy without instability): INCONCLUSIVE** [quantitative, causal].
RMSE/MAE 3.63× vs parent ~3.4× — no improvement in outlier tail; ratio slightly worsened.
No ablation with count-w=0.1 on same architecture exists. Did not hurt, but did not fix
the catastrophic outlier failures.

## Key Observations [qualitative, causal]

- **Overfitting**: train loss 14.3→7.8 (46% drop) while val plateaued ~22-23 after E26.
  Best at 65% of training; early stopping at ~26-28ep would save wall-clock.
- **RMSE/MAE = 3.63×**: persistent catastrophic outliers. Small number of samples with
  predicted counts >> 3 or << true count. MSE loss dominates gradient for large errors;
  count-w=1.0 L1 term insufficient to counter this.
- **Gate undertrained**: layer_logits (2 scalar params) converges fast but locks to global
  compromise. Per-token gating would let spatial regions choose layer blends.
- **Class-name collision**: DinoPromptV2 redefined (vs N0007). Works in isolation but
  violates single-home rule for future shared imports.

## Quality Gate (7 dimensions)

| Dimension | Verdict | Notes |
|-----------|---------|-------|
| Mechanistic | ✅ | Layer-gate fusion of mid+final DINOv2 tokens doubles feature diversity |
| Scoped | ✅ | FSC147, DINOv2-S, frozen backbone, ≤32M |
| Predictive | ⚠️ | Bar met, but confounded — cannot predict individual lever effects |
| Falsifiable | ✅ | MAE > 27.65 was the bar; met decisively |
| Novel | ⚠️ | Compound scale-up, not novel mechanism; layer-gate is incrementally new |
| Transferable | ✅ | Multi-tap pattern applies to any hierarchical frozen backbone |
| Actionable | ✅ | Ablations, per-token gate, Huber loss are clear next steps |

## New Hypotheses

- **H0019**: IF per-token layer-gating replaces scalar gating IN multi-layer DINOv2 tap
  architectures, THEN val MAE further decreases by ≥1.0, BECAUSE different spatial regions
  (small objects vs background vs crowd patches) benefit from different feature depths, and
  a scalar gate forces a global compromise. DISPROVED IF MAE increase or no change.

- **H0020**: IF Huber loss (δ=5.0) replaces MSE in the counting head loss IN multi-tap
  DINOv2 architectures, THEN RMSE/MAE ratio drops below 3.0, BECAUSE Huber loss caps
  gradient magnitude for large errors, reducing catastrophic outlier failures without
  sacrificing median-case accuracy. DISPROVED IF RMSE/MAE > 3.5.

## Actionable Next Steps for gen-4

1. **Ablation A**: Same N0010 architecture, 26ep → isolate epoch contribution
2. **Ablation B**: 40ep, count-w=1.0, single-layer tap block 11 only → isolate architecture
3. **Per-token gate** (H0019): 2-layer MLP over [z6;z11] → 784 scalar gates
4. **Huber loss** (H0020): Replace MSE with Huber(δ=5.0), keep count-w L1
5. **Early stopping**: Cap at 28ep with patience=5 on val MAE to reduce overfitting

## Booking List

- evidence H0017 supports (strength 0.9) — N0010
- evidence H0018 neutral (strength 0.3) — N0010 (unisolated)
- create H0019 — per-token gating
- create H0020 — Huber loss

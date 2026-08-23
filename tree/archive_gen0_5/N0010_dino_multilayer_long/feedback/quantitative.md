# Quantitative Feedback — N0010_dino_multilayer_long

## Metric Comparison

| Metric | Parent N0007 | N0010 Final | N0010 Best (E26) | Bar (≤26.0) |
|--------|-------------|-------------|-------------------|-------------|
| val MAE | 27.65 | 22.607 | 21.531 | ✅ PASSED |
| val RMSE | — | 81.97 | 75.91 | — |
| RMSE/MAE | ~3.4× | 3.63× | 3.53× | — |
| ΔMAE vs parent | — | **−5.04** | **−6.12** | — |

Best MAE (21.53) beats both parent (−6.12) and bar (26.0 by 4.47). Final-epoch MAE (22.61) still beats parent by 5.04. The 1.07 gap between best and final indicates moderate epoch-to-epoch variance but no catastrophic overfitting.

## Variance Analysis (MAE trajectory, E013–E040)

Epoch MAE range: 21.53 (E26) – 26.51 (E14). After convergence (E21+):
- Mean MAE (E21–E40): ~22.8
- Std dev: ~0.7
- Oscillation pattern: ±1.0 around plateau, no monotonic degradation
- Train loss: 14.3 → 7.8 (steadily decreasing, 46% reduction), yet val MAE plateaus → train/val gap widening (overfitting to train distribution)

**RMSE concern**: RMSE 82.0 vs MAE 22.6 → ratio 3.6×. FSC147 count ceiling is 3; a perfect model would have RMSE ≈ MAE. Ratio 3.6× implies a small number of catastrophic outliers (predicted counts >> 3 on low-count images, or vice versa). The high RMSE is driven by worst-case errors, not broad miscalibration. loss_count_weight=1.0 did NOT resolve this — the outlier tail persists.

## Hypothesis Verdicts

**H0017**: "Multi-layer frozen backbone taps + ≥35ep yields val MAE ≤ 26.0"
→ **SUPPORTED**. Best MAE 21.53 << 26.0 bar. Multi-tap + longer schedule clearly helps. The layer-gated dual-tap architecture is a genuine improvement over single-layer extraction.

**H0018**: "count-w=1.0 improves accuracy without instability"
→ **INCONCLUSIVE**. No baseline with count-w=0.1 on the same architecture exists for ablation. The RMSE/MAE ratio (3.6×) is not worse than parent's ~3.4×, so count-w=1.0 did not hurt — but also did not fix the outlier problem. Instability: none detected (no OOM, no NaN, no diverging loss). The oscillation is within normal range for batch_size=8 on FSC147.

## Root Cause of High RMSE

The RMSE/MAE ratio suggests the model fails on specific hard samples — likely high-density images where density-map supervision is ambiguous, or very small objects where bbox encoding loses spatial precision. The Fourier prompt encoding (8 freqs × 4 coords × 2 + log_area = 65-dim → 384-dim) may lack capacity to distinguish fine-grained spatial differences. This is an **optimization confound** (loss weighting didn't help) rather than a **fundamental architecture** limit — the MAE is strong, so the backbone features are rich; the head just can't calibrate edge cases.

## Summary

N0010 is the new champion: 21.53 best MAE (18.2% improvement over parent). Architecture validated. The high RMSE is the next bottleneck — suggests downstream work should target outlier robustness (e.g., Huber loss, sample reweighting, or test-time augmentation) rather than further backbone exploration.

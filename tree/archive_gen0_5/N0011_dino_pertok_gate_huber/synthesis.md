# synthesis.md — N0011_dino_pertok_gate_huber

## Decision: REFUTED [quantitative, causal, qualitative]

N0011 best MAE 26.68 (final 26.89, RMSE 94.34, ratio 3.53) vs parent N0010 21.53 (RMSE 75.9, ratio 3.52) Δ=+5.15 (+24% worse) [quantitative]. Both primary bars missed decisively; per-token gate + Huber hurt rather than helped. Huber optimization stable (train Huber loss 9.27→3.43 monotonic, val plateau 27-28.3 E13-E34, no OOM/instability) ruling out optimization artifact [quantitative, causal]. Idea surgically scoped and falsifiable, disproved cleanly [qualitative].

## Hypothesis Resolutions

**H0019 per-token gate → MAE ≤20.53 (parent-1.0): CONTRADICTS** [quantitative, causal] strength 0.85, confidence 0.5→0.415. Observed 26.68 miss by 6.15; no epoch approached 21.5 (best E34). Mechanism: 49k-param per-patch MLP (model.py:30-42, 768→64→2) overfits frozen DINOv2 patch variance; scalar 2-param global gate (N0010 model.py:44) was stronger regularizer.

**H0020 Huber δ=5 → RMSE/MAE <3.0: CONTRADICTS** [quantitative, causal] strength 0.80, confidence 0.5→0.42. Observed 3.53 (94.34/26.68) vs target <3.0 and disproof >3.5 threshold; ratio unchanged vs parent 3.52. Huber caps gradients for |error|>5 where CAC tail signal lives, under-fitting outliers it must learn.

**H0018 count-w 1.0 vs 0.3 (indirect): WEAK SUPPORTS** [causal] strength 0.35, confidence 0.5→0.535. N0010 (w1.0) 21.53 beats N0011 (w0.3) by 5.15, directionally consistent with count supervision helping, but triple confound (gate+loss+weight) prevents isolated attribution.

## Key Observations [quantitative, causal, qualitative]

- Per-token gating adds capacity without signal: 784 independent softmax gates on noisy [z6;z11] tokens memorize train patches, not transferable depth preference [causal].
- Huber misaligned for CAC: suppressing tail gradients stabilizes RMSE (−0.1) but hurts val MAE; MSE's quadratic penalty is needed for rare high-count tail [causal].
- Count-w confound dominates: reverting 1.0→0.3 (config.py:10) reduces L1 supervision 3.3×; regression cannot be disentangled without single-edit ablations [causal, quantitative].
- Training healthy: 23.16M params, monotonic loss decline, late best E34 suggests overfitting from capacity, not under-training [quantitative].

## Quality Gate (7 dimensions)

| Dimension | Verdict | Notes |
|-----------|---------|-------|
| Mechanistic | ✅ | Patch-wise depth preference + gradient capping mechanisms explicit |
| Scoped | ✅ | FSC147, DINOv2-S reg4, frozen backbone, ≤32M |
| Predictive | ❌ | Both bars predicted improvement, observed opposite |
| Falsifiable | ✅ | MAE ≤20.53 and ratio <3.0 bars cleanly disproved |
| Novel | ⚠️ | Per-token spatial gating lightweight but unproven; Huber standard |
| Transferable | ⚠️ | Negative result transfers: warns against high-capacity per-patch fusion on frozen tokens |
| Actionable | ✅ | Isolations + tail-reweighting + resolution paths clear |

## New Hypotheses

- **H0021**: IF count-w=1.0 is isolated (scalar gate+MSE held fixed) IN multi-tap DINOv2-S 392px architecture, THEN val MAE drops ≥3.0 vs w0.3 variant, BECAUSE stronger L1 count supervision reinforces density calibration on high-count images. DISPROVED IF MAE delta <1.0 or w0.3 wins.
- **H0022**: IF tail-aware reweighting (log-count or count-stratified sampling) replaces Huber capping IN multi-tap DINOv2 architectures, THEN RMSE/MAE drops below 3.2 without MAE loss vs MSE, BECAUSE reweighting preserves outlier gradients while balancing head-heavy count distribution. DISPROVED IF ratio ≥3.5 or MAE worsens >1.0.

## Actionable Next Steps for gen-4/5

1. Single-edit ablations from N0010: (A) per-token-only +MSE+w1.0, (B) Huber-only +scalar+w1.0, (C) w0.3-only to isolate levers.
2. Hold count-w=1.0 as proven; do not revert without isolated evidence.
3. Keep scalar gate; if spatial adaptivity retried, use hidden 8-16 + dropout 0.3 or pooled attention, not 64-wide MLP.
4. Prefer N0012 (518px) and N0013 (mosaic) already queued for scale/long-tail; evaluate tail reweighting (H0022) next if they stall.
5. Early-stop ≈26ep (N0010 best E26; N0011 late peak E34 was overfit) to save wall-clock.

## Booking List

- evidence H0019 contradicts 0.85 — N0011
- evidence H0020 contradicts 0.80 — N0011
- evidence H0018 supports 0.35 — N0011 (indirect, confounded)
- create H0021 — count-w isolation
- create H0022 — tail reweighting

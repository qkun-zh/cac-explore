# Feedback — N0060_xscale_max (Quant)

Clean run confirmed (result.json): 30/30 ep, status=success, instability=false, oom=false, params 31.42M ≤ 32M ✓. Best **22.886@E23**, final 23.010, RMSE 85.11, ~1693s.

**Gate fired: R4 FAIL.** Best 22.886 > 20.0 (H0084 DISPROVED bar); FAIL margin +2.886, a ~3.3σ outlier against the ±0.25 noise band. Best delta vs champion = 22.886 − 19.647 = **+3.239** — nowhere near CONFIRM (<19.45) or WEAK-KEEP (19.45–20.0).

Same-epoch deltas, champion reference: E16 25.115 vs 22.338 = **+2.78**; E23 22.886 vs 20.419 = **+2.47**; E29 23.079 vs 19.647 = **+3.43**. The gap widens as the champion keeps descending while N0060 stalls: it only reaches N0054's E16 level (22.34) at E23, then saturates.

Tail plateau E17–30: 22.886–23.931 (E18 23.931 spike); best improves just 23.363→22.886 (−0.477) over E17→E23, then 7 epochs of zero gain, final +0.124 above best. Overfit-to-scale signature: train loss keeps dropping 3.69 (E17) → 2.70 (E30, −27%) while val is flat at ~22.9–23.4 — the MAX head memorizes peak magnitude, generalizes nothing.

RMSE corroborates harm: **85.11 vs champion 74.05 = +11.06** (+14.9%), alongside MAE +16.5% (−+3.24). Consistent ~15% dual-metric degradation, not an MAE artifact. Floor delta +3.24 ties N0058's +3.50, the worst corner of the frozen-regime negative table.

N=1 sufficiency: FAIL/KILL at N=1 is defensible per idea.md — gate is a deterministic band vs LOCKED champion, verdict sits 2.9 above bar, run clean; no 2nd seed (reserved for CONFIRM <19.40).

**Verdict: clean FAIL; H0084 refuted.** The coarse single-slot additive axis is saturated at ONE summary — a 2nd order-statistic (MAX) on the same prototype is harmful redundancy (+3.24). Close the H0083→H0084 extension line.
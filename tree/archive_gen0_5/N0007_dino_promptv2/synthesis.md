# synthesis.md — N0007_dino_promptv2

## Verdict
N0007 succeeded decisively: best val MAE 27.65 / RMSE 93.24 @25ep/768s, 22.81M. New best node (-15.3% vs N0006). H0014 (<=29.0) SUPPORTED — feature substrate confirmed as the dominant causal lever; features>mechanism>schedule ordering validated.

## Quality Gate (7 dims)
mechanistic pass · scoped pass · predictive pass (bar pre-registered, passed with margin)
falsifiable pass · novel pass (first DINOv2+area-prompt CAC under frozen budget)
transferable pass · actionable pass

## Deduplicated Updates
- H0014 supports 0.80-0.90 → booked w=0.85. No contradicted hypotheses this node.
- Reviewers converge on next levers: multi-layer taps, cross-attn basis decoder (N0003 merge), count-calibration path for RMSE tail, longer schedule.

## Booking List
- create H0014; evidence: H0014 supports 0.85 from N0007.

## Tested Hypotheses
[H0014]

## Scores
best_metric 27.652, train_seconds 768 → quality ≈ 0.71, avail ≈ 0.77, score ≈ 0.74 (best parent candidate); status → synthesized

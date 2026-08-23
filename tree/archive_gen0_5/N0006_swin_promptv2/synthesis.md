# synthesis.md — N0006_swin_promptv2

## Verdict
N0006 succeeded marginally: best val MAE 32.10 / RMSE 102.61 @30ep/405s, 28.37M. Beats parent N0005 by 1.7% but misses both pre-registered bars (H0012 <=31.5; H0013 ratio <3.0 → 3.18). Train loss 12.5→0.71 vs val stall after ~E14 = overfitting; schedule lengthening exhausted.

## Quality Gate (7 dims)
mechanistic pass · scoped pass · predictive pass · falsifiable pass · novel pass · transferable pass · actionable pass

## Deduplicated Updates
- H0012 contradicts 0.50–0.60 → booked w=0.55 (direction right, bar missed).
- H0013 contradicts 0.45–0.50 → booked w=0.50.
- Cross-node causal conclusion: features > mechanism > schedule. DINOv2-S substrate is the top untested lever.

## Booking List
- create H0012, H0013; evidence: H0012 contradicts 0.55, H0013 contradicts 0.50 from N0006.

## Tested Hypotheses
[H0012, H0013]

## Scores
best_metric 32.097, train_seconds 405 → quality ≈ 0.60, avail ≈ 0.83, score ≈ 0.69; status → synthesized

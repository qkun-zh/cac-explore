# synthesis.md — N0003_convnext_xattn

## Verdict
N0003 succeeded: val MAE 34.26 / RMSE 104.52 @14ep/430s, 16.93M total (frozen convnext_nano + FPN + 2-layer cross-attn decoder). Beats N0002 by 18.5% relative — H0004 confirmed at mechanism level; best node so far.

## Quality Gate (7 dims)
- mechanistic pass · scoped pass · predictive pass (H0004 ≥10% predicted, 18.5% observed)
- falsifiable pass (bar stated ex ante) · novel pass (exemplar-prompted mixture-of-density-bases)
- transferable pass (category-agnostic conditioning) · actionable pass (clear children)

## Deduplicated Updates
- H0004 supports 0.85/0.70/0.80 across quant/qual/causal → booked w=0.80.
- H0005 neutral (no stride ablation). H0011 neutral here but causally prioritized for children.

## Booking List
- create H0004, H0005 (texts from idea.md); evidence H0004 supports 0.80 from N0003

## Tested Hypotheses
[H0004]

## Scores
best_metric 34.264, train_seconds 430.5 → quality ≈ 0.53, avail ≈ 0.76, score ≈ 0.62; status → synthesized

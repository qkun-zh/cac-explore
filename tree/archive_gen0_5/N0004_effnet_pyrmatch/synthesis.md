# synthesis.md — N0004_effnet_pyrmatch

## Verdict
N0004 succeeded technically but underperformed: best val MAE 40.37 / RMSE 119.68 @20ep/535s, only 3.65M total. Beats N0002 by 4% (<8% H0006 bar), loses to N0003 by 18%. Plateaued ~E14. H0007 disproved: ≤12M frozen EffNet does NOT reach <35 — backbone scale is a real bottleneck.

## Quality Gate (7 dims)
- mechanistic pass · scoped pass · predictive pass (H0007 stated ex ante, cleanly disproved)
- falsifiable pass · novel pass (scale-gated matching) · transferable pass · actionable pass

## Deduplicated Updates
- H0006 neutral (~0.25–0.30; no same-backbone ablation, cross-node comparison confounded).
- H0007 contradicts 0.70–0.85 → booked w=0.80.

## Booking List
- create H0006, H0007 (texts from idea.md); evidence H0007 contradicts 0.80 from N0004; evidence H0006 supports 0.30 (weak, confound-noted) — net effect small positive drift on H0006.

## Tested Hypotheses
[H0006, H0007]

## Scores
best_metric 40.367, train_seconds 535 → quality ≈ 0.41, avail = 0.70, score ≈ 0.53; status → synthesized

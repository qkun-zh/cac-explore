# synthesis.md — N0008_dino_xattn

## Verdict
N0008 ran successfully but underperformed badly: best val MAE 46.56 @30ep/1063s vs parent 27.65. H0015 (<=25.5) contradicted IN THIS CONFIGURATION with an explicit optimization confound: decoder divergence E1-E12 at lr=1e-3 without warmup on DINOv2 token magnitudes, then steady recovery (still descending at cutoff). Mechanism question deferred to stabilized sibling N0009 with a pre-registered decision rule.

## Quality Gate (7 dims)
mechanistic pass · scoped pass · predictive pass · falsifiable pass · novel pass · transferable pass · actionable pass (diagnostic -> concrete retry rule)

## Deduplicated Updates
- H0015 contradicts w=0.50 (confound-noted; adjudication deferred to N0009).

## Booking List
- create nothing new; evidence: H0015 contradicts 0.50 from N0008.

## Tested Hypotheses
[H0015]

## Scores
best_metric 46.563, train_seconds 1063 → quality ≈ 0.15, avail ≈ 0.69, score ≈ 0.37; status → synthesized

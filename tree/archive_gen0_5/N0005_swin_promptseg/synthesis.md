# synthesis.md — N0005_swin_promptseg

## Verdict
N0005 succeeded as BEST node: val MAE 32.66 / RMSE 102.13 @20ep/271s, 28.22M (frozen swin_tiny ms_in22k + Fourier prompt token + adapter + conv head). H0008 supported (<40 passed @E9); H0009 DISPROVED — implicit conditioning beat comparable-budget explicit matching by 22%, reversing the expected ranking. Only 15% of τ_max used.

## Quality Gate (7 dims)
- mechanistic pass · scoped pass · predictive pass · falsifiable pass · novel pass
- transferable pass · actionable pass (clear gen-1 merge path)

## Deduplicated Updates
- H0008 supports ~0.75–0.90 → booked w=0.85.
- H0009 contradicts ~0.70–0.85 → booked w=0.80.
- H0010 neutral (seg-KL not implemented; engine MSE used).
- H0011 neutral but flagged by ALL reviewers as top untested lever for children.

## Booking List
- create H0008, H0009, H0010; evidence: H0008 supports 0.85, H0009 contradicts 0.80 from N0005.

## Tested Hypotheses
[H0008, H0009]

## Scores
best_metric 32.655, train_seconds 271 → quality ≈ 0.58, avail = 0.85, score ≈ 0.69 (new best parent)

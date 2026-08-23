# synthesis.md — N0018_dino_protoiter

## Verdict
REFUTED (Lead-booked abbreviated synthesis after 2× subagent network failures). Early-stopped at ~E34 on
NaN instability; best val MAE 23.403 vs parent champion 21.53 — H0027 (≤19.5) missed by +1.9 even before
collapse. Iterative pseudo-prototype refinement AMPLIFIED variance on frozen DINOv2 tokens instead of
correcting dense-scene misses: error feedback loop fed noisy top-K features back into conditioning,
diverging in late low-lr phase (NaN from ~E35).

## Quality Gate
mechanistic pass · scoped pass · predictive fail · falsifiable pass · novel pass · transferable partial · actionable pass

## Booking List
- create H0027; evidence H0027 contradicts w=0.70 from N0018 (NaN instability + bar miss).

## Tested Hypotheses
[H0027]

## Scores
best_metric 23.403, train_seconds 1273 → quality ≈0.30, avail ≈0.71, score ≈0.46; status → synthesized

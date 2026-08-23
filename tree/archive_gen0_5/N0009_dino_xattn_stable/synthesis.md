# synthesis.md — N0009_dino_xattn_stable

## Verdict
N0009 adjudicated H0015/H0016 cleanly: best val MAE 30.67 @30ep/985s — stable optimization confirmed, but > parent 27.65 → cross-attn basis transfer to DINOv2 tokens REFUTED at mechanism level. Champion remains N0007 (implicit area-prompt + adapter + MLP head).

## Quality Gate (7 dims)
all pass (diagnostic value: clean negative with pre-registered rule)

## Deduplicated Updates
- H0016 contradicts w=0.75 booked. H0015 additional contradicts w=0.55 (now properly refuted).

## Booking List
- evidence: H0016 contradicts 0.75 from N0009; H0015 contradicts 0.55 from N0009.

## Tested Hypotheses
[H0016] (+H0015 adjudication)

## Scores
best_metric 30.669, train_seconds 985 → quality ≈ 0.42, avail ≈ 0.62, score ≈ 0.50; status → synthesized

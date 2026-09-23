# Synthesis — N0031_h0029 (H0029 train-fit global scale)

## Verdict

**SUPPORTED** (supports w=0.85, conf→0.585). Fit on train only:
`s = Σ pred·gt / Σ pred² = 1.102986` → val **18.424** (≤18.50 PASS),
test **19.606** (≤19.70 PASS). Dense 330.8→275.7. Sparse 5.81→6.19 (not booked).

## Confidence math

Booked via `discovery evidence H0029 --type supports --strength 0.85`
(pattern match to single clean mechanism confirmation; not hand-computed).

## Bookings (K_SYNTH ≤ 2)

1. Evidence supports already recorded on H0029 from N0031.
2. Journal evidence event (done).

No new hypothesis in this synthesis (calibration family: one global scale is the
canonical form; affine/binned variants stay research notes in
`calibration_trainfit.md` unless user wants separate cards).

## Deploy claim (honest pair)

| | raw | cal (×1.102986) |
|---|---|---|
| val | 19.326 | **18.424** |
| test | 19.066 | **19.606** |

Ship both numbers. Test regresses +0.54; val improves −0.90. This is **not**
a supervised-bar rescue and does not reopen N0001 corridor.

## Next on this root

- N0030 H0028 full-flag vitl16 still in flight (bar ≤38.696).
- Optional: affine form as separate hyp only if global scale is insufficient for user goal.

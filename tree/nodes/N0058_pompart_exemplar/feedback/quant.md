# Quant — N0058_pompart_exemplar (parent N0054, EARLY-STOPPED @E15)

## Verdict: NEGATIVE. Pre-registered gate breached at E15 (+2.17 vs parent same-epoch); refutes H0080 (exemplar aggregation-operator swap).

## 1. Curve — train MAE/RMSE (local/feedback_src_N0058/train.log, `E*/30` lines)
| Ep | MAE | RMSE | best |
|----|-----|------|------|
| 01 | 32.484 | 101.279 | \* |
| 02 | 33.624 | 109.150 | |
| 03 | 30.868 | 106.102 | \* |
| 04 | 26.829 | 96.410 | \* |
| 05 | 32.661 | 90.782 | |
| 06 | 26.014 | 91.228 | \* |
| 07 | 25.511 | 90.781 | \* |
| 08 | 27.999 | 83.839 | |
| 09 | 28.263 | 94.086 | |
| 10 | 24.679 | 88.684 | \* |
| 11 | 25.468 | 95.700 | |
| 12 | 28.050 | 78.563 | |
| 13 | **23.151** | **77.823** | \* ← best_mae 23.1509 (matches result.json) |
| 14 | 23.505 | 85.327 | |
| 15 | 23.805 | 85.888 | (killed here) |

Best ep E13 (MAE 23.151 / RMSE 77.823). Non-best last ep E15 (23.805 / 85.888).
Converged headline value = best_mae **23.1509**. ES@E15: 3 contiguous losses decline, but MAE still rising 23.151→23.805 in last 2 eps → no late breakaway.

## 2. Head-to-head vs parent N0054
No N0054 train.log found locally (N0054 dir = model/config/idea/novelty/synthesis only; none in local/) → used given refs: 21.635@E15, 21.250@E17, 20.419@E23, ~19.65@E27-29, final 19.647/74.05.
- E15 same-epoch: **23.805 vs 21.635 = +2.17** (bar = parent-best+1.5 = 21.147 → breached by +0.66). Kill legit.
- Trajectory: N0058 is ≥+1.5 worse at every comparable epoch (E13 23.151 even ~22.4-gap = +0.75 best case). No epoch where PMOM tracks N0054.

## 3. Metrics honesty
- RMSE: run min 77.823 (E13); rises to 85.888 by E15. No RMSE momentum downward in final 3 eps. Parent final RMSE 74.05 (val caveat: our 77.82 is train — sets differ, but the within-run trend is flat/up).
- Gap: best_mae 23.151 vs parent final 19.647 = **+3.50**; vs parent E15 same-epoch 21.635 = +1.52 on best, +2.17 same-epoch.
- Params: use_pmom True **29.86M total / 2.04M trainable** vs N0054 31.32M / 3.50M. **→ CAPACITY CONFOUND, stated explicitly:** PMOM removed ~1.68M of the exemplar encoder (TransformerEncoder 1.58M + proj/attn) and added only ~209k. Same parameter-CLASS (≤32M) but NOT the same trainable budget (2.04M vs 3.50M, −42%). The +2.17 cannot be cleanly attributed to the operator alone; a parameter-matched control (e.g. rewidthed moment_proj) is required before claiming PMOM is "genuinely weaker at same budget". The pre-registered early-stop bar is capacity-independent, so the KILL stands regardless.

## 4. Noise-band assessment
PMOM best 23.151 (E13) vs N0054@E13–15 ≈ 21.6–22.4 → delta +0.75 … +1.55, all ≥0.7 and far outside the ±0.25 band (N0056/N0057 convention). Effect is real, not noise. **Verdict: NEGATIVE (worse than parent).**

## Note
`tree/nodes/N0058_pompart_exemplar/synthesis.md` does NOT exist yet. Early-stop + gate ⇒ **diagnostic feedback required** (ES root-cause: capacity loss vs operator quality).
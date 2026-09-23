# N0031_h0029 — evolution of N0029_tf_champion (H0029 train-fit scale)

## Hypothesis Set

1. **H0029** — IF a deployable inference-side calibration of the frozen N0015 champion that fits a single global multiplicative scale s on the TRAIN split per-image predictions (`s = sum(pred*gt)/sum(pred*pred)`, no val access, no checkpoint edit) and applies `pred_cal = s * pred` at test time, IN the N0029_tf_champion training-free tree evaluated on FSC147 val and test, THEN val MAE is at most **18.50** AND test MAE is at most **19.70**, BECAUSE train-fit least-squares scale already measured s=1.102986 giving val 18.424 / test 19.606 with dense 330.8 improving to 275.7, so the booked bars sit just above the observed pair while requiring test not to collapse past a pre-registered ceiling. DISPROVED IF val MAE exceeds 18.50, or test MAE exceeds 19.70, or the scale fit requires any val/test statistic beyond the fixed train-fit s.

→ conf 0.500 — one test event.

## Not a train_node card

Training-free / inference-side. Experiment = closed-form fit on
`N0015_h0014/train_perimage.json`, apply to val and test dumps.

| Field | Value |
|---|---|
| fit source | `tree/.../N0015_h0014/train_perimage.json` (n=3659) |
| eval | same node `val_perimage.json` (n=1286), `test_perimage.json` (n=1190) |
| formula | `s = Σ pred·gt / Σ pred²` on train only |
| measured s | **1.1029862538958246** |
| val raw → cal | 19.3259 → **18.4236** |
| test raw → cal | 19.0663 → **19.6058** |
| dense val raw → cal | 330.81 → **275.66** |
| sparse val raw → cal | 5.814 → 6.191 (context; not a booked bar) |
| script | `fit_scale.py` in this node (recomputes from dumps) |

## Bars (conjunctive)

- R1 val MAE ≤ **18.50** → measured **18.4236** PASS
- R2 test MAE ≤ **19.70** → measured **19.6058** PASS
- R3 no val/test statistic in the fit → formula uses train only PASS

⇒ **SUPPORTED** as booked (both bars + legality).

## Honest caveats (do not bury)

- Test **worsens** by +0.54 vs raw 19.066 — deploy claim must ship the pair.
- Sparse goes 5.81→6.19 (over the old supervised 5.45 guard; not this card's bar).
- This does **not** rescue supervised H0027; it is an inference-side lever on frozen N0015.

# Quantitative feedback — N0031_h0029 (H0029 train-fit scale)

**Verdict: SUPPORTED** — both booked bars PASS on recompute from N0015 dumps via fit_scale.py.

| Bar | Threshold | Measured | Verdict |
|---|---|---|---|
| val MAE (cal) | ≤18.50 | **18.4236** | PASS |
| test MAE (cal) | ≤19.70 | **19.6058** | PASS |
| fit legality | train-only s | s=1.102986 from train_perimage n=3659 | PASS |

Raw→cal: val 19.3259→18.424 (−0.902), test 19.0663→19.606 (+0.540), dense 330.81→275.66, sparse 5.814→6.191.

Source: N0015_h0014 {train,val,test}_perimage.json; recompute .

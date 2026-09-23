# Inference-side train-fit calibration (N0015 frozen preds) — research log

Legal lever: fit on **train** split perimage, apply to val/test. No checkpoint edit.
Source dumps: .

| Transform (fit=train) | val MAE | test MAE | dense val | sparse val |
|---|---|---|---|---|
| raw | 19.326 | 19.066 | 330.81 | 5.814 |
| scale s=1.102986 | **18.424** | 19.606 | **275.66** | 6.191 |
| affine a=1.11677 b=-2.10245 | **18.388** | 19.463 | — | — |
| pred-binned mean_ratio | 18.755 | 19.207 | — | — |
| pred-binned med_ratio | 19.082 | 19.045 | — | — |
| log-affine | 19.010 | 19.010 | — | — |

**Read:** val under-18 is reachable with train-fit scale/affine; **test worsens** (+0.4..+0.5).
Honest claim is the pair, not val alone. Sparse bar 5.45 is missed after scale (6.19).

CHEAT reference (fit-on-val, NOT deployable): scale/affine ~17.9-class; gt-binned affine oracle much lower — do not book.

This note is **not** a ledger hypothesis yet. Book H0029 only if we want a formal card
with pre-registered bars on the calibrated pair (e.g. val<=18.5 AND test<=19.1).

# STATE — Current Situation

**Stage**: SEARCH SATURATED — champion N0010 21.53; 9 consecutive refuted variants confirm local optimum
**Blockers**: none

## Verified Facts (do not re-learn)
- Champion: frozen DINOv2-S reg4 taps(6,11) gate + area-prompt + adapter768 + MLP head, 392px, 40ep, count-w1.0 → **21.53** @1275s/23.11M
- 9 refuted children (all single/dual-lever from champion): N0011 26.68 · N0012 26.03† · N0013 22.40 · N0014 28.42† · N0015 unrun · N0016 seq 81.62 · N0017 22.19 · N0018 23.40 NaN · N0019 23.36 · N0020 25.28†
- † = early-stopped per never-idle rule
- CAC decomposition: discrimination ✅ DINOv2 · separation ❌ frozen features can't split instances · calibration ⚠️ RMSE/MAE=3.6
- H0014 supported (DINOv2 substrate), H0004 supported (cross-attn on conv), H0008 supported (implicit prompt)
- H0007/H0009/H0012/H0013/H0015/H0016/H0019/H0020/H0023/H0025/H0026/H0027/H0029 all contradicted
- 27 hypotheses banked; subagent git hallucination logged in failure_modes.md

## Why <16 Is Not Reachable Under Current Constraints
1. Frozen backbone can't adapt features for instance separation
2. 30-min budget prevents full-schedule high-res training
3. Residual errors are small-object misses + dense-scene calibration, both requiring backbone adaptation
4. SOTA methods achieving <16 use full fine-tuning (CounTR 11.95) or much larger models (CountGD GroundingDINO)

## Options
A. Accept 21.53 as best-under-constraints result
B. Relax τ_max to 60min + run champion at 518px with grad-accum (H0023 retry)
C. Relax frozen-backbone constraint (violates mission)
D. Paradigm shift to detection-based (needs >100M backbone)

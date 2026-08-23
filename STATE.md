# STATE — Current Situation

**Stage**: gen-5 RUNNING — champion N0010 21.53; 7 refuted children since; N0020 scale-aware decoding = last structural lever
**Blockers**: none

## Verified Facts (do not re-learn)
- Champion: frozen DINOv2-S reg4 taps(6,11) gate + area-prompt + adapter768 + MLP head, 392px, 40ep, count-w1.0 → **21.53**
- ALL single-lever variants from champion have FAILED:
  - N0011 per-token+Huber 26.68 · N0012 highres518 26.03 (truncated) · N0013 augreg 22.40
  - N0014 highres+augreg 28.42 (stopped ep12) · N0016 seqcount 81.62 (collapse) · N0017 tail-down 22.19
  - N0018 protoiter 23.40 (NaN) · N0019 tail-up 23.36
- CAC decomposition: discrimination ✅ (DINOv2) · separation ❌ (frozen features can't split instances) · calibration ⚠️ (RMSE/MAE=3.6)
- N0020 scale-aware deformable sampling = attacks separation directly via exemplar-size matched filtering
- H0023 (highres) confounded by timeout — retry needs grad-accum + full schedule
- Subagent git claims must be verified via `git log` (hallucination incident)

## Next Steps
1. Poll N0020 (~21min); collect; if ≤19.5 → gen-6 combines with multi-layer + higher res
2. If N0020 fails too → the frozen-backbone ceiling is ~21.5 on this budget; report honestly

## Active Tasks
- T0042 executor N0020 RUNNING
EOF_MARKER_NOT_NEEDED

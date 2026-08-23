# feedback/quantitative.md — N0006_swin_promptv2

## reasoning
Best val MAE 32.10 / RMSE 102.61 @30ep/405s, 28.37M. Beats parent N0005 (32.66) by only 1.7% — H0012's <=31.5 bar MISSED (disproved). H0013 ratio 102.61/32.23 = 3.18 >= 3.0 → disproved. Train loss fell 12.5→0.71 while val stalled after ~E14: classic overfit; more epochs of the same data will not help. Best at E20 (32.097), late-epoch gains marginal (~0.1).

## actionable_feedback
- Stop lengthening schedules on this feature/mechanism pair — capacity/data-fit saturated.
- Attack the train/test domain gap instead: stronger augmentation needs loader support, or test-time prototype refinement (LOCA-style).
- The dual-scale gate + area prompt are kept (small but real gain); next upside is FEATURE swap: DINOv2-S reg4 stride14 gives 28x28 tokens with better instance clustering than swin stride16/32.

## hypothesis_updates
- H0012: contradicts, strength 0.60. Improvement direction correct (+1.7%) but below the pre-registered 3.5% bar.
- H0013: contradicts, strength 0.50. Ratio worsened slightly vs parent (3.13→3.18); gating did not shrink the tail here.

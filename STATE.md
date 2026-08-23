# STATE — Current Situation

**Stage**: gen-4 RUNNING — N0010 champion 21.53; N0011 refuted 26.68 (H0019/H0020 down); N0012 highres518 running, N0013 augreg queued
**Blockers**: none — pipeline full (never-idle)

## Verified Facts (do not re-learn)
- Engine: single box [B,4] S-space; low-res density OK; <32M total; MSE+0.3·L1 default; huber optional
- Trajectory: N0007 27.65 → N0010 21.53 (best) → N0011 26.68 worse (per-token+Huber refuted)
- Arch: frozen DINOv2-S reg4 dual taps scalar gate + 40ep + count-w1.0 is champion recipe
- H0019 0.415 & H0020 0.42 refuted (w≥0.80); H0018 0.535 weak support w1.0; 22 hypotheses
- timm traps: BCHW, PATCH const, dynamic_img_size, Linear on tokens only

## Next Steps (in order)
1. N0012 highres518 (23.11M, 518px) running — poll single-shot, collect when done
2. N0013 augreg (photometric+bbox jitter, 40ep) queued — launch when GPU frees
3. Continue iterating to MAE<16; next levers: isolated count-w, tail-reweight (H0022)

## Active Tasks
- T0025/T0026 N0012 running; T0027/T0028 N0013 coded/queued; N0011 feedback+sytnh done

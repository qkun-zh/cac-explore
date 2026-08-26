# STATE — Session 2026-08-26 (Lead=qkun-local)

**Mode**: 用户指导模式 (User-Guided) — confirmed at session start per user request
**Preflight**: git up-to-date · creds fresh (mtime 08:26 today) · SERVER_OK · RTX3060 idle, no tmux
**Champion**: CAC-D simplified — val MAE **19.15**; eval-routed effective best **19.18/66.37** (N0021_dino_partialft + resolution routing)
**Baseline 384**: DONE Ep32/32 best val **22.38** TEST 18.33-18.88 (see /tmp/cac_d_384_baseline.log, ckpt /data/runs/cac_d_baseline384/best.pth)
**Queue prompt encoder**: DONE screening @224 — q_mse 20.41 / q_ada 21.45 / q_bl 23.60 (all WORSE than 19.15 no-queue control; queue parked)
**CAC-SI line (SI-INR)**: RUNNING tmux si_224 — dual-stream frozen DINOv3 + B_H/S scale-invariant encoding + cross-attn + INR continuous decoding; 32ep @224 ~108s/ep; log /tmp/cac_si_224.log, ckpt /data/runs/cac_si_224/best.pth; 28.65M total 0.83M trainable
**Tree**: champion lineage N0027_norm_flip_swa done; children N0028–N0032 all failed/timeout

## Awaiting user direction
- cac_si 32ep in flight (~58min); compare vs 19.15 (224 lane control)
- Queue line parked (negative result); density variants ada/bl inconclusive without clean control
- Next: cac_si result -> if positive, promote to 384; resolution-shift eval (SI-INR's selling point)

## Gotchas
- pkill -f cac_d self-kills the ssh shell (cmdline match) → separate cleanup/launch calls
- precompute MUST pass explicit size override (model default 224); HF_ENDPOINT must be exported before python starts
- **Two caches**: `/data/cache/fsc147_features` = true-384 reference; `_224` = FAST-EXPERIMENT (~30s/ep, 3-4 concurrent, override cache_dir+image_size together, not comparable to 384)
- Server env: POT/triton removed, cv2 headless only; /data/runs = cac_d_redesign + N0027_norm_flip_swa + cac_d_baseline384

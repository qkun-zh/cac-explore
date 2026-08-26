# STATE — Session 2026-08-26 (Lead=qkun-local)

**Mode**: 用户指导模式 (User-Guided) — confirmed at session start per user request
**Preflight**: git up-to-date · creds fresh (mtime 08:26 today) · SERVER_OK · RTX3060 idle, no tmux
**Champion**: CAC-D simplified — val MAE **19.15**; eval-routed effective best **19.18/66.37** (N0021_dino_partialft + resolution routing)
**Baseline 384**: DONE Ep32/32 best val **22.38** TEST 18.33-18.88 (see /tmp/cac_d_384_baseline.log, ckpt /data/runs/cac_d_baseline384/best.pth)
**Queue prompt encoder**: MFU feature queue (E=32,m=2, per-class 147) semi-cached; 3×224-fast trainings RUNNING in tmux: q_mse/q_ada/q_bl — 6.9GiB total
**Tree**: champion lineage N0027_norm_flip_swa done; children N0028–N0032 all failed/timeout

## Awaiting user direction
- Queue改造已上线，3训练并行中（tmux q_mse/q_ada/q_bl，各32ep）
- Density variants: mse / ada_mse / bl (all +queue) screening vs 19.15 control
- Next: monitor Ep4 TEST, promote winner to 384

## Gotchas
- pkill -f cac_d self-kills the ssh shell (cmdline match) → separate cleanup/launch calls
- precompute MUST pass explicit size override (model default 224); HF_ENDPOINT must be exported before python starts
- **Two caches**: `/data/cache/fsc147_features` = true-384 reference; `_224` = FAST-EXPERIMENT (~30s/ep, 3-4 concurrent, override cache_dir+image_size together, not comparable to 384)
- Server env: POT/triton removed, cv2 headless only; /data/runs = cac_d_redesign + N0027_norm_flip_swa + cac_d_baseline384

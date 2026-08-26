# STATE — Session 2026-08-26 (Lead=qkun-local)

**Mode**: 用户指导模式 (User-Guided) — confirmed at session start per user request
**Preflight**: git up-to-date · creds fresh (mtime 08:26 today) · SERVER_OK · RTX3060 idle, no tmux
**Champion**: CAC-D simplified — val MAE **19.15**; eval-routed effective best **19.18/66.37** (N0021_dino_partialft + resolution routing)
**Tree**: champion lineage N0027_norm_flip_swa done; children N0028–N0032 all failed/timeout

## Awaiting user direction
- **Baseline RUNNING**: cac_d @ true 384px cache, 32ep (2w+10s+20c), val/ep + test/4ep, ~89s/ep, 7.6GiB → 12G card fits only 1 concurrent run
- Log `/tmp/cac_d_384_baseline.log` · ckpt `/data/runs/cac_d_baseline384/best.pth`
- Today's fixes: Condenser input proj (latent crash), FFN direction typo; dead code removed (-109 lines)
- Old 19.15 (224px cache) not comparable to new baseline — new reference point

## Gotchas
- pkill -f cac_d self-kills the ssh shell (cmdline match) → separate cleanup/launch calls
- precompute MUST pass explicit size override (model default 224); HF_ENDPOINT must be exported before python starts
- **Two caches**: `/data/cache/fsc147_features` = true-384 reference; `_224` = FAST-EXPERIMENT (~30s/ep, 3-4 concurrent, override cache_dir+image_size together, not comparable to 384)
- Server env: POT/triton removed, cv2 headless only; /data/runs = cac_d_redesign + N0027_norm_flip_swa + cac_d_baseline384

# STATE — Session 2026-08-26 (Lead=qkun-local)

**Mode**: 用户指导模式 (User-Guided) — confirmed at session start per user request
**Preflight**: git up-to-date · creds fresh (mtime 08:26 today) · SERVER_OK · RTX3060 idle, no tmux
**Champion**: CAC-D simplified — val MAE **19.15**; eval-routed effective best **19.18/66.37** (N0021_dino_partialft + resolution routing)
**Tree**: champion lineage N0027_norm_flip_swa done; children N0028–N0032 all failed/timeout

## Awaiting user direction
- User-Guided mode: next experiment follows user's directive
- Candidate directions from prior session Next-list: clean baseline of simplified model / count-loss variants / backbone fine-tuning
- Gates unchanged: novelty_check → check_hypothesis → calibration_report

## Gotchas
- Server env trimmed 08:46: POT/triton removed, cv2 = opencv-python-headless 5.0.0.93 only
- /data/runs holds only cac_d_redesign + N0027_norm_flip_swa after cleanup

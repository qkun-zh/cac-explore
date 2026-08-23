# STATE — Clean Slate

**Mission**: MAE ≤ 4 on FSC147 test. No architecture restrictions. Full fine-tuning allowed.
**Previous**: 20 nodes explored under frozen-backbone constraint → best 21.53 (N0010). Archived in `tree/archive_gen0_5/`.
**Blockers**: none

## Verified Facts (carry forward)
- Frozen DINOv2-S ceiling = ~21.5 val; full FT needed for <16
- SOTA reference: CountGD 5.74 · GeCo2 7.64 · DAVE 8.66 · CACViT 9.13 · LOCA 10.79
- Engine supports huber loss + param_groups differential LR + eval_frac subsample
- timm traps logged in failure_modes.md
- Server: RTX 3060 12GB; revproxy alive; HF cache at /data/asset/hf

## Next Steps
1. Deep research SOTA (CoDi diffusion, DAVE detect-verify, foundation models)
2. Design paradigm-shift architecture for MAE≤4
3. Execute fast loop

## Active Tasks
(none — clean slate)

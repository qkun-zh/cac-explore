# STATE — Current Situation

**Stage**: gen-3 COMPLETE — N0010 new champion val MAE 21.53 (Δ=-6.12 from parent N0007); gen-4 selection next
**Blockers**: none — revproxy alive; server OK

## Verified Facts (do not re-learn the hard way)
- Engine contract: single box [B,4] S-space; low-res density OK; <32M total; MSE+0.3·L1(count)
- Trajectory: N0001 baseline 46.69 → N0005 32.66 → N0007 27.65 → **N0010 21.53** (best)
- Architecture: frozen DINOv2-S reg4 + layer-gated dual taps (block6+11) + adapter(768) + MLP head
- Training: 40ep, lr=1e-3, count-w=1.0, best MAE at epoch21, overfitting signal after
- RMSE/MAE = 3.6× → catastrophic outlier failures persist; count ceiling = 3
- Hypothesis bank: 20 hypotheses (H0001–H0020); top supported: H0014 0.585, H0017 0.59
- timm traps logged in memory/failure_modes.md (img_size, features_only BCHW, patch_embed hidden, etc.)
- SeqCount inspiration: CAC as sequence generation (patch-level discrete tokens + cross-entropy)

## Next Steps (in order)
1. Idea hat: gen-4 child of N0010 — higher resolution 518/672 or SeqCount-inspired approach
2. Standard loop to synthesis; iterate toward MAE<16

## Active Tasks
- T0001–T0022 all done (gen-0 through gen-3 complete)

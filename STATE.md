# STATE — Honest Assessment After Paradigm Investigation

## Mission: MAE ≤ 4 on FSC147 test

## Current Best: N0010 val 21.53 (archived recipe)

## What We Learned From Research + R001 Failure

**SOTA methods achieving <6 test MAE all share:**
1. Foundation-model backbones (GroundingDINO 172M, SAM-HQ 636M, AM-RADIO ~100M+) — frozen
2. Point detection output with Hungarian matching loss — NOT density MSE
3. 150+ epochs of training on A100-class GPUs (15+ hours)
4. Inference-time calibration stack (TT-Norm, SAM correction, adaptive tiling)

**Why we cannot reach ≤4 under current constraints:**
- RTX 3060 12GB: cannot load GroundingDINO/SAM-HQ even frozen
- 30-min τ_max: detection paradigms need >100ep to overcome threshold barrier
- Single GPU: no distributed training or large-batch optimization
- FSC147 test requires cross-category generalization unseen in training

## Options Going Forward
A. **Full FT champion (N0021)**: expect val ~15-18, test ~16-19. Not ≤4 but big improvement.
B. **Test-time enhancement**: apply TT-Norm + multi-crop to N0010 checkpoint → maybe -2 MAE free
C. **Longer schedule**: τ_max=60min + full FT + mosaic → possibly val ~13-15
D. **Accept result**: 21.53 is strong given frozen-backbone + 30min constraint

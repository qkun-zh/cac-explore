# STATE — Gen-6 Fresh Start

**Mission**: MAE ≤ 4 on FSC147 test. Full FT allowed. Any architecture.
**Stage**: RESEARCH COMPLETE → root bootstrap next
**Blockers**: none

## Research Synthesis (3 parallel agents)
- Sub-6 SOTA methods ALL use: point detection + Hungarian matching loss (NOT density MSE)
- Best published: VQCounter 4.86 (GroundingDINO + prompt queue + VoronoiCost)
- Key insight: density regression saturates ~10-15; point detection reaches <6
- Inference tricks (TT-Norm, SAM calibration) worth 1-3 MAE for free
- Class imbalance (~95% empty patches) killed naive SeqCount; needs weighted sampling
- AM-RADIO > DINOv2 (multi-teacher distillation of CLIP+DINOv2+SAM)

## Root Bootstrap Plan (K=4 paradigms)
1. **R1: Point Detection + Hungarian Matching** (highest EV — proven by VQCounter/CountGD)
2. **R2: Density + Verification Two-Stage** (DAVE style, combines our density expertise)
3. **R3: Scale-Aware Deformable Attention** (SPDCN-style, exploits exemplar geometry)
4. **R4: Diffusion Location Generation** (CoDi-style, handles annotation noise)

## Verified Facts
- Engine supports huber, param_groups, eval_frac; needs Hungarian matching + CE additions
- FSC147 annotations include point coordinates (needed for detection targets)
- Server: RTX 3060 12GB; DINOv2-S full FT fits; GroundingDINO frozen-only fits
- 27 old hypotheses archived with gen0-5

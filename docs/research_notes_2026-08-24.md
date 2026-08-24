# Research Notes — 2026-08-24 (Researcher subagent, degraded-mode session)

## Verified SOTA anchors (FSC147, 3-shot)
- CACViT AAAI'24: val 10.63 / test 9.13 (ViT-B ~86M)
- DAVE CVPR'24: val 8.91 / test 8.66
- CountGD NeurIPS'24: val 7.10 / test 5.74–6.75 w/ TTA stack; 37M trainable (GroundingDINO-Swin-B+BERT frozen)
- VQCounter IJCAI'25: test 4.86 (current overall SOTA; params unverified)
- UpCount arXiv'26 (ref-free): val 13.62 — ViT-B + taps[2,5,8,11]+DPT+FeatUp
- CountingDINO WACV'26 train-free: test 20.93 (frozen DINOv2-L-reg prototypes)
- No published ≤35M-total method beats ~test MAE 9–12; sub-9 is heavy-model territory.
- Val≠test comparability caveat: exemplar-fusion designs differ from our exemplar-free head.

## Partial FT of pretrained ViTs
- CLIP-is-a-strong-fine-tuner (2212.06138): top-half FT ≈ full FT; minimize representation drift; LLRD 0.6 beats uniform LR (+0.9%); EMA +0.3–0.9%.
- ProLIP '24: last-proj FT + L2-anchor-to-init makes training LR-insensitive → cheap stabilizer candidate.
- MGCAC ACCV'24 uses backbone lr ×0.1 → our mult=0.1 matches de facto recipe. No counting-specific LLRD/#blocks ablation found (unverified).

## Augmentation for counters
- CounTR: jitter/noise/etc = "limited" gain; mosaic helps long-tail only.
- MGCAC/"Recipe for CAC": mosaic ALONE WORSENS FSC147 in-domain MAE → skip mosaic.
- CLIP-FT lit: weak aug preferred on pretrained backbones; removing MixUp/CutMix helped → supports jitter-only plan.

## Test-time adaptation — biggest inference-time lever
- CounTR TT-norm (calibrate density scale on exemplar regions): "significant boost".
- CountGD ablation: none→TT-Norm val 8.69→7.99, test 10.92→9.62; +SAM+crops → val 7.10/test 6.75.

## Registers & DINOv3
- ViTs-Need-Registers ICLR'24: reg tokens fix high-norm artifacts, better dense features — validates reg4 substrate.
- DINOv3 (2508.10104): Gram-anchoring, stable dense features, has ViT-S-class; NO published FSC147 result — open opportunity; patch16 changes token grid @392px (use 384/448); HF-mirror availability unverified.

## Actionable ranking (evidence/cost fit)
1. Exemplar-based TT-Norm at eval (zero params/train cost) → path to val<19
2. Partial-FT sweep: blocks 9-11 + lr_mult {0.05,0.2}; optional EMA / L2-anchor (ProLIP)
3. Jitter-only light aug (keep dropout/wd unchanged)
4. Deeper tap pyramid (add block ~2-3 or 8) — multi-layer reassembly repeatable win
5. DINOv3-S substrate swap — highest ceiling, medium risk
6. Avoid: mosaic, diffusion aug, MLLM routes

## Unverified / flags
- Param counts: DAVE, LOCA, VQCounter, CACViT heads
- "SeqCount+" paper name not found (mosaic sources: CounTR'22/MGCAC'24)
- FSC147 official split = 3659 train images; mission text said 6591 — confirm our manifest

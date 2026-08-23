# STATE — Current Situation

**Stage**: M3 — N0002 done (val MAE 42.05, 317s, 10ep) → Feedback×3
**Blockers**: none — revproxy PID 4978 + git proxy active

## Verified Facts (do not re-learn the hard way)
- torch==2.10.0+cu128 / torchvision 0.25.0 in env `cac`; CUDA works
- FSC147 VarV2 at /data/dataset/FSC147 (3659/1286/1190); S0001 smoke 46.69 @2ep
- Engine: single box [B,4] S-space; low-res density OK (sum-conserving); total <32M
- N0002: 22.17M (frozen DINOv2-S reg4 dynamic_img_size=True), val MAE 42.05 vs 46.7 baseline = +4.6 gain only; still trending down at ep10 (loss 13.9→7.7) but τ_max under-utilized (317s << 1800s)
- Revproxy SOCKS 1081 via 172.18.80.1:57777 needed for server git; HF via hf-mirror.com

## Next Steps (in order)
1. Feedback hats: quant/qual/causal for N0002 → feedback/*.md (≤60 lines each)
2. Synthesis: synthesis.md (≤100 lines), book H0001-H0003,H0011 evidence, rebuild_index, tree.json synthesized
3. Coding N0003-N0005 (fix their models similarly with dynamic_img_size / smoke)

## Active Tasks
- T0007-T0009 pending_feedback quant/qual/causal for N0002
- T0003-T0005 pending_coding N0003/N0004/N0005
- T0006 done_executor N0002 (collected)

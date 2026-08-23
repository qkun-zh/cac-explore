# STATE — Current Situation

**Stage**: M3 — N0002 synthesized (MAE 42.05) → Coding N0003/N0004/N0005 next
**Blockers**: none — revproxy PID 4978 + git proxy active; H0001 0.50→0.425 (contradicted)

## Verified Facts (do not re-learn the hard way)
- torch==2.10.0+cu128 / torchvision 0.25.0 in env `cac`; CUDA works
- FSC147 VarV2 at /data/dataset/FSC147 (3659/1286/1190); S0001 smoke 46.69 @2ep
- Engine: single box [B,4] S-space; low-res density OK (sum-conserving); total <32M
- N0002: 22.17M frozen DINOv2-S reg4, val 42.05/122.06 in 317s (10ep), trend 48.5→42.05 but H0001 contradicted; time under-used (18% of 30min)
- Hypotheses: H0001 0.425 (1 test, contradicts), H0002/H0003/H0011 0.50 (untested)
- Revproxy SOCKS 1081 via 172.18.80.1:57777 needed for server git

## Next Steps (in order)
1. Coding: N0003 convnext_xattn (T0003) and N0004 effnet_pyrmatch (T0004) — fix dynamic_img_size / smoke in tmux
2. Executor for whichever smoke-greens first → collect → Feedback → Synthesis
3. N0005 swin_promptseg follows; prioritize cross-attn/multi-scale (causal from N0002)

## Active Tasks
- T0003-T0005 pending_coding N0003/N0004/N0005 (priority: N0003 cross-attn first)
- T0007-T0009 done_feedback N0002 (feedback/*.md + synthesis.md + index rebuilt)

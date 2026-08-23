# STATE — Current Situation

**Stage**: M3 — N0002 smoke GREEN (22.17M, 7.7s) → Executor running
**Blockers**: none — revproxy PID 4978 + git proxy `socks5h://127.0.0.1:1081` active; timm 1.0.28 installed

## Verified Facts (do not re-learn the hard way)
- torch==2.10.0+cu128 / torchvision 0.25.0 in env `cac`; CUDA works (RTX 3060 12GB)
- FSC147 VarV2 at `/data/dataset/FSC147`; check_data passes (3659/1286/1190)
- Engine: model(imgs[B,3,S,S], bboxes[B,4]) — SINGLE box S-space; low-res density OK (sum-conserving upsample); asserts total <32M
- No local python/torch — all training on server; first DINOv2 download cached at /data/asset/hf via HF_ENDPOINT=hf-mirror.com
- N0002 model bug fixed: `dynamic_img_size=True` for 392 input on 518-pretrained vit_small_patch14_reg4_dinov2.lvd142m
- S0001_smoke success (7.7s, 22.17M); revproxy SOCKS 1081 via 172.18.80.1:57777 required for git on this instance

## Next Steps (in order)
1. Executor: `bash scripts/run_node.sh N0002_dino_protocorr` in tmux (10 epochs, ~20min) → watch → collect
2. Feedback×4 → Synthesis books H0001-H0011
3. Coding N0003-N0005 in parallel (after N0002 collect frees GPU)

## Active Tasks
- T0002 done_coding N0002 (smoke green @1463ace)
- T0006 pending_executor N0002 (claimed next)
- T0003-T0005 pending_coding N0003/N0004/N0005

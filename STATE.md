# STATE — Current Situation

**Stage**: M3 in progress — N0002 coded (smoke pending) · **BLOCKED: server instance reclaimed**
**Blockers**: `ssh cac-server` times out → need new DeepLn instance (AGENTS §6 rotation drill)

## Verified Facts (do not re-learn the hard way)
- torch==2.10.0+cu128 / torchvision 0.25.0 in env `cac`; CUDA works (RTX 3060 12GB)
- FSC147 VarV2 at `/data/dataset/FSC147`; check_data passes (3659/1286/1190)
- Engine: model(imgs[B,3,S,S], bboxes[B,4]) — SINGLE exemplar box, S-space; low-res density OK
  (sum-conserving upsample); eval via density sums; asserts total params < max_params_M (32)
- No local python/torch — all smoke/training must run on the server (draft-push → ssh smoke)
- S0001_smoke: success; throughput ⇒ ~7–15 real epochs fit in τ_max=30min
- Web grounding: CountingDINO (WACV'26, training-free DINOv2), CACViT (AAAI'24, plain-ViT extract-and-match
  + scale/magnitude embeddings), CounTR (cross-attn), GeCo2 (scale generalization is THE open problem)

## Next Steps (in order)
1. USER: rent new instance → paste 2 credential lines into local/address_and_password.md → `bash scripts/onboard.sh`
2. Smoke N0002 on server (`--smoke --epochs 2`, tmux + marker; first run downloads DINOv2-S reg4 weights)
3. Green smoke → card T0002 done → executor card → real run ≤30min → collect
4. Then N0003–N0005 coding/execution loop; Feedback ×4 → Synthesis books H0001–H0011

## Active Tasks
- T0002 claimed_coding N0002 (code pushed @7bc7bc2; awaiting server for smoke)
- T0003–T0005 pending_coding for N0003/N0004/N0005

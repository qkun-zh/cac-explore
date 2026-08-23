# STATE — Current Situation

**Stage**: M3 root bootstrap done → 4 root nodes proposed (N0002–N0005); Coding hat next
**Blockers**: none. The rented instance may be reclaimed at any time — prefer short tasks and collect results often.

## Verified Facts (do not re-learn the hard way)
- torch==2.10.0+cu128 / torchvision 0.25.0 in env `cac`; CUDA works (RTX 3060 12GB)
- FSC147 VarV2 at `/data/dataset/FSC147`; check_data passes (3659/1286/1190)
- Engine: model(imgs[B,3,S,S], bboxes[B,4]) — SINGLE exemplar box per image, S-space; low-res density OK
  (sum-conserving upsample); eval via density sums; asserts total params ≤ max_params_M (32)
- S0001_smoke: success, val MAE 46.69 @ 2ep/27s; throughput ⇒ ~7–15 real epochs fit in τ_max=30min

## Next Steps (in order)
1. Coding hat: N0002_dino_protocorr first (claim T0002) → smoke green → coded; then N0003–N0005
2. Executor hat: server run in tmux ≤30min → collect → tree.json done
3. Feedback ×4 → Synthesis books H0001–H0010 evidence into hypotheses.jsonl

## Active Tasks
- T0002–T0005 pending_coding for N0002/N0003/N0004/N0005 (root bootstrap by Idea hat)
- T0001 S0001_smoke done (collected)

# STATE — Current Situation

**Stage**: M2 done (first real node trained) → M3: batched hypothesis generation
**Blockers**: none. The rented instance may be reclaimed at any time — prefer short tasks and collect results often.

## Verified Facts (do not re-learn the hard way)
- torch==2.10.0+cu128 / torchvision 0.25.0 installed in env `cac`; CUDA works (RTX 3060)
- FSC147 VarV2 lives at `/data/dataset/FSC147`; check_data fully passes (3659/1286/1190)
- Engine contract: models may output low-resolution density; engine auto-upsamples with sum conservation; evaluation uses density sums
- S0001_smoke: status=success, val MAE 46.69 @ 2ep/27s (end-to-end verified on real data)

## Next Steps (in order)
1. Idea Agent produces root nodes N0002–N0005 under the frozen-backbone constraint (`tree/nodes/*/idea.md` + `tasks/T*_pending_*.md`)
2. Coding Agent implements model/config → local `--smoke` self-check → push
3. Executor runs real training in tmux (within τ_max=30min) → collect → four feedback reports
4. Synthesis books evidence into hypotheses.jsonl + confidence updates

## Active Tasks
- T0001 S0001_smoke → **done** (result.json collected and committed)

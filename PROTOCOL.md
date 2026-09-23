# PROTOCOL — evaluation and hardware

- Split: FSC147 official val. Rapid gate: fixed subset `/data/repro/val_subset286.json` (n=286, 89 bins, seed 20260922). Same file for every comparison.
- Full val 1286: champion rows only, only on explicit user request. Never auto-run.
- Target: subset286 MAE <= 20.0. Champion rows: full val only.
- Backbone: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` required in graph. Stack <= 64M.
- Seed: 20260830 fixed for any seeded op. No seed sweeps without user approval.
- Server: `ssh cac-server`, python `/data/miniconda/envs/cac/bin/python`, RTX 3060 12GB, HF cache `/data/asset/hf` with `HF_HUB_OFFLINE=1`.
- Sync: `tar czf - <paths> | ssh cac-server 'tar xzf - -C /data/cac'`. No rsync. Long jobs via `setsid nohup ... &`, poll with single ssh grep, never sleep-loop.

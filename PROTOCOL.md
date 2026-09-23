# PROTOCOL - evaluation and hardware

- Split: FSC147 official val. Rapid gate: fixed subset `/data/repro/val_subset286.json`
  (n=286, 89 bins, seed 20260922). Same file for every comparison in a wave.
- Full val 1286: champion rows only, only on explicit user request. Never auto-run.
- Target: subset286 MAE <= 20.0. Champion rows: full val only.
- Backbone: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` required in the
  inference graph (code location enforced by check.py). Stack <= 64M.
- Seed: 20260830 fixed for any seeded op. No seed sweeps without user approval.
- Server: `ssh cac-server` (alias in `~/.ssh/config`). Preflight first:
  `ssh -o ConnectTimeout=8 -o BatchMode=yes cac-server 'echo OK'`.
  On failure: creds are in gitignored `local/address_and_password.md`
  (line 1: `ssh -p <port> <user>@<host>`, line 2: password); re-onboard with
  `python3 scripts/install_key.py` (installs pubkey + rewrites the alias).
  python `/data/miniconda/envs/cac/bin/python`, RTX 3060 12GB,
  HF cache `/data/asset/hf` with `HF_HUB_OFFLINE=1`.
- Canonical server run dir: `/data/cac` (sync with `bash scripts/sync.sh`; the
  pre-v2 server tree is archived once at `/data/cac_v1`; the old v1 working
  copy `/data/cdino_run` stays untouched as historical reference only).
- Run: `ssh cac-server 'bash /data/cac/models/run_variant.sh <NODE_ID> [flags...]'`
  (one override flag only; logs land in `/data/repro/logs/<NODE_ID>_sub286.log`).
- After a run: `scp cac-server:/data/repro/logs/<NODE_ID>_sub286.log /tmp/` then
  `graph.py evidence --log` (MAE/type parsed from the log; never hand-typed).
- Long jobs via `setsid nohup ... &`, poll with a single ssh grep, never sleep-loop.

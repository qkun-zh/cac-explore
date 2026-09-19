# STATE.md

One session block; REWRITE it each session open (archive stale content to
journal/ first). Read AGENTS.md before anything below.

## Session (2026-09-19T14:30)
- **Mode**: Free-Research. Cycle 1 CLOSED (H0009 refuted, lab committed+pushed
  `6420e08`). Cycle 2 ACTIVE: **N0004_h0010** full budget run in flight (tmux
  `runN4`, `--budget-seconds 1800`); then N0005_h0010 queued (single GPU).
- **Tree**: N0001_champion (seeded baseline now **21.459** @ep32, result.json
  updated) → N0003_h0009 (H0009 DISPROVED, Δ=−1.149 vs +0.30 bar) → N0005_h0010
  (frozen-random gate control, H0010+H0011); and N0004_h0010 (H0010 per-exemplar
  gate + H0009, child of champion). Both coded: 2 switches, 31.36M, exemplar-gate
  identity init (bias +3), N0004 smoke + run launched, N0005 smoke green.
- **Memory**: H0010 (per-exemplar gate 0.500), H0011 (const-input control 0.500)
  booked. H0009 0.400 (contradicts). Standings: H0003 confirmed 0.770, H0005
  refuted 0.215, rest uncertain. Calibration: reliability error **0.195** (<0.2,
  WARN cleared), 23 tests, direction 14/14.
- **Engine fixes (this session)**: (1) budget phase was training on the smoke
  datamodule → real FSC147DataModule now used; (2) BestCheckpoint hook ordering
  → now a sink fed by the module with the fresh MAE; (3) per-run tb dir cleared
  so event files never mix across runs.
- **Server**: cac-server live (RTX 3060 12GB); tensorboard installed (tuna
  mirror). Runs: N0003 done, N0001 seeded rerun done (21.459), N0004 running.
- **Gotchas**:
  - run artifacts are gitignored (`**/run/`, tmp_ideas_round2/); do not `git add -A`
    blindly after syncing runs back.
  - run_node flags: `--budget-seconds N` AND `--budget-seconds=N` both valid.
  - HF is offline-first: set HF_* at the TOP of any ad-hoc server python before
    heavy imports (entrypoints do this via hub.setup_hf_env()).
  - Server copy is /data/cac; sync via tar-over-ssh (no rsync on this box);
    result.json lives at the NODE ROOT, run_node.log at run/latest/.
  - Champion's honest baseline is the SEEDED 21.459 (20260830), NOT the historic
    migration 19.647.
  - `local/` creds gitignored — check mtime before each lab session.
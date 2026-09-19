# STATE.md

One session block; REWRITE it each session open (archive stale content to
journal/ first). Read AGENTS.md before anything below.

## Session (2026-09-19T14:00)
- **Mode**: Free-Research. Evolution cycle ACTIVE: N0003_h0009 (H0009 condenser
  channel gate) **full budget run in flight** (tmux `runN3`, `--budget-seconds
  1800`, smoke green 1st attempt, launched 13:45). ~31 min wall; will mark
  `timeout` if the wall-clock ceiling trips.
- **Tree**: N0001_champion (seed root, MAE 19.647) → child **N0003_h0009**
  (coded, config diff = one line `use_channel_gate = true`; params TOTAL
  31.36M ≤ 32M, gate 32,768).
- **Memory**: H0001–H0009. H0003 confirmed (0.770), H0005 refuted (0.215),
  H0001/2/4 uncertain (0.664), H0006 (0.336), H0007/8 (0.269), H0009 (0.500).
  Calibration: reliability_error 0.227 (WARN), direction 14/14 — gate tripped.
- **Engine fixes this session (2 bugs, found via probe, both green)**:
  1. Budget phase reused the smoke datamodule → fake 4-batch training (26 s /
     32 "epochs", MAE 507). Fixed: real FSC147DataModule(smoke=False) for the
     budget fit; budget_hit works against the real run.
  2. BestCheckpoint read `callback_metrics` from a hook that fires before the
     module logs val/mae → recorded garbage (6593 vs actual 241). Fixed:
     BestCheckpoint is now a sink invoked by CountingLit with the fresh MAE
     (best_mae 246.14 @ ep1 in probe).
- **Server**: cac-server live (RTX 3060 12GB), tensorboard now installed (tuna
  mirror ~60 s — pypi index stalled). 3 next-round candidates queued
  (cond keynorm 0.435 / fine-stage exview 0.471 / background null tok 0.410).
- **Gotchas**:
  - run_node flags: `--budget-seconds N` AND `--budget-seconds=N` both valid.
  - HF is offline-first: set HF_* at the TOP of any ad-hoc server python before
    heavy imports (entrypoints do this via hub.setup_hf_env()).
  - Server copy is /data/cac; sync via tar-over-ssh (no rsync on this box).
  - Historic N0001 MAE 19.647 predates seeding; fresh reruns with seed 20260830
    are the reproducible artifacts.
  - `local/` creds gitignored — check mtime before each lab session.
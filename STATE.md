# STATE.md

One session block; REWRITE it each session open (archive stale content to
journal/ first). Read AGENTS.md before anything below.

## Session (2026-09-19T15:35)
- **Mode**: Free-Research. Cycles 1+2 CLOSED (N0003 H0009 refuted; N0004 joint
  gates refuted D=+1.92; N0005 H0011 refuted +0.202). Cycle 3 ACTIVE: two SOLO
  children of champion in flight (tmux `runN6`): **N0006_h0012** (H0012
  count-temp) then **N0007_h0013** (H0013 subpixel-up), each
  `--budget-seconds 1800`, single GPU sequential.
- **Mechanism change (rule-10, user-approved 'A', commits 4eb5b49 + 923ef76)**:
  composition-feasibility filter (`src/cac/expt/mechanisms.py` registry):
  co-composing hypotheses that share a head component or config switch are
  rejected; hypotheses whose `requires` switches are absent from the parent
  config are dropped (H0011 needs use_channel_gate); on explicit `--book/--new`,
  adjoins capped at ONE compatible (`--solo` = none). `run_node` now back-fills
  tested_hypotheses from idea.md (Eq.5-6 "not-yet-tested-on-ancestry" was
  silently disabled, re-testing hyps forever). Historical backfill done for
  N0003/N0004/N0005. Eq.5-6 SCORING untouched.
- **Tree**: N0001_champion (seeded **21.459**) → N0003_h0009 (H0009 refuted
  22.6076) → N0005_h0010 (H0011 refuted 22.8102); N0004_h0010 (joint refuted
  23.3824). Old infeasible N0006_h0009/N0007_h0013 pruned; re-booked SOLO:
  N0006_h0012=[H0012], N0007_h0013=[H0013]. Both coded (single-switch, 31.39M /
  31.37M), smoke green.
- **Memory**: H0009 0.320 refuted-track, H0010 0.400 (2 contradicts + 1 neutral;
  STILL never tested cleanly solo — recommend champion-only use_exemplar_gate
  next cyclically), H0011 0.400 (inert capacity), H0012 0.500 (untested), H0013
  0.500 (untested). H0003 confirmed 0.770, H0005 refuted 0.215. Calibration:
  reliability error **0.176** (<0.2), 25 tests, direction 15/15.
- **Engine fixes (earlier this session)**: real-data budget fit, BestCheckpoint
  as module-fed sink, per-run tb isolation. All committed+pushed.
- **Server**: cac-server RTX 3060 12GB; N0006/N0007 running in tmux `runN6`
  (~35 min each, ETA ~16:20).
- **Gotchas**:
  - run artifacts gitignored (`**/run/`, tmp_ideas_round2/); never `git add -A`
    blindly.
  - `--book H --solo` = one-hyp inference; `--book H` default = mandated + 1
    compatible adjoin. Read the node's idea.md BEFORE trusting which hyps a node
    actually tests.
  - run expected completeness = 32 epochs; budget 1800s may early-stop; mark
    `timeout` honestly if it does.
  - HF offline-first (hub.setup_hf_env() at top of entrypoints).
  - Server copy /data/cac not a git repo — sync src/scripts/tree via tar-over-ssh.
  - Champion honest baseline is seeded **21.459**, not migration 19.647.
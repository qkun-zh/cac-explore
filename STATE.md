# STATE.md

One session block; REWRITE it each session open (archive stale content to
journal/ first). Read AGENTS.md before anything below.

## Session (2026-09-19T13:00)
- **Mode**: Free-Research (next moves staged, no commit yet — awaiting go).
- **Tree**: N0001_champion (seed root, synthesized, MAE 19.647) → child
  **N0003_h0009** (proposed; first evolution child via `discovery hypo`, Q_t=[H0009]).
- **Memory**: H0001–H0008 seeded + H0009 (channel-gate condenser).
  Standings: H0003 confirmed (0.770), H0005 refuted (0.215),
  H0001/2/4 uncertain (0.664), H0006 (0.336), H0007/8 (0.269), H0009 (0.500).
- **Server**: onboarded — `ssh cac-server` live (RTX 3060 12GB;
  yzkczmrjwdtrqpkhsnow.deepln.com:48769). GPU smoke PASSED on server copy
  (`/data/cac`); champion model loads from `/data/asset/hf` offline, bf16 AMP.
- **Mechanism**: calibration upgraded (reliability primary = P(hold|conf),
  direction accuracy secondary, weighted reliability error >0.2 → WARN;
  current 0.227 WARN, direction accuracy 14/14). New CLI
  `discovery evidence <hyp_id> --type <t> --strength <w> [--node] [--note]`
  is the only sanctioned way to append evidence (append-only, phantom-id ban).
- **Open frontier**: multires (392/518), SWA, tail_reweight, dual_res_eval,
  EMA, pluggable head parts above frozen hs(2,3) + exemplar embedding
  (see docs/research_direction.md). Next evolution target: implement H0009
  delta on N0003_h0009 (Coding Agent) → server run → feedback → synthesis.
- **Gotchas**:
  - `local/` (creds) gitignored — check mtime before every lab session.
  - Server copy is `/data/cac`; sync code via tar-over-ssh (rsync absent on this box).
  - `hub.setup_hf_env()` is offline-first (HF_HUB_OFFLINE=1, mirror endpoint);
    set HF_* BEFORE heavy imports in entrypoints.
  - Historic N0001 MAE 19.647 predates seeding; a fresh rerun with seed
    20260830 is the reproducible artifact.
  - conformance is the commit gate — never commit with it red (currently green).
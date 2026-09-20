# STATE — session 2026-09-20 (first post-reset loop closed for N0002; N0003 training)

- **Champion/live parent**: `tree/N0001_champion` (23.293 @ep26, seed 20260830).
  New live reference: **N0002_h0001 = 22.5641 @ep30** (H0001 simprior, solo,
  32/32ep, 1755s, no budget hit). Future falsifier bars bind to 22.5641
  (next 0.30 bar = 22.26), not 23.293.
- **Ledger**: H0001 `use_simprior` → supports w=0.85 @N0002 (conf 0.585,
  uncertain); H0002 `use_gca_cal` → N0003_h0002 (training, ep8 28.8 @13:55).
  Synthesis for N0002 booked ≤2: adopted refine card + new shuffled-exemplar
  control (both validated, unbooked — book at batch-2 instantiation).
- **Protocol (canonical, fixed)**: augment=false, EMA eval, seeded loaders,
  cudnn deterministic, 32ep/1800s. **Seed 20260830 forever**.
- **Server**: `ssh cac-server`, python `/data/miniconda/envs/cac/bin/python`,
  GPU RTX3060. Chain `train-b1` (N0002→N0003) in tmux; code synced via
  `tar ... | ssh ... tar -C` (no rsync on either side). Smoke per node in
  tmux `smoke-<node>` + `capture-pane` polls; kill session after.
- **Gotchas (same-session doc rule)**: (1) `scripts/curve.py <id>` builds the
  path as `tree/N0001_champion/<id>/…` — works for nested nodes, broken for
  the root (use the events path directly for N0001). (2) Final `idea.md`
  MUST keep one machine-readable booking line `1. **Hxxxx** — <text>`
  (runner regex) or `tested_hypotheses` stays [] — happened on both batch-1
  nodes, repaired via agent appends (N0003 fix synced to server pre-completion).
  (3) `tar cf - --exclude=… <dir>` — excludes must precede the dir arg.
  (4) Server python needs repo offline-HF setup for ad-hoc scripts
  (`cac.hub.setup_hf_env()` first) or Hub connect fails.

(End of file - session block)

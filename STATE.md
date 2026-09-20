# STATE — session 2026-09-20 (batch 1+2 post-reset; N0005 training)

- **Live parent**: **N0002_h0001 = 22.5641 @ep30** (H0001 `use_simprior` CONFIRMED;
  chain: 23.293 → 22.564). All falsifier bars bind to 22.5641 (next 0.30 bar = 22.26).
- **Ledger**: H0001 `use_simprior` supports w=0.85 → conf 0.585 (uncertain).
  H0002 `use_gca_cal` contradicts w=0.80 (23.2088 @ep32; margin 0.084) → conf 0.420.
  H0003 `use_padapt` contradicts w=0.95 (**25.5661 @ep13, +3.00 worse**; active but
  harmful) → conf 0.405. H0004 `use_simbank` (N0005_h0004) — training.
- **Batch-2** (on N0002): N0004_h0003 padapt REFUTED (mechanism corruption of
  Condenser K/V; 0 synthesis bookings — objectness-modulated re-encode failed the
  new-falsifier gate per §11). N0005_h0004 simbank (h2 1/8 token similarity bank)
  in chain `train-b2`.
- **Diagnostic (local/research/perimage_diagnostic.md)**: error heavy-tailed +
  count-correlated — top 1% val images ≈30% of Σ|Δ|, gt>500 (17 imgs) mean |Δ|≈512,
  dense images severely UNDER-counted; global count calibration has NO headroom
  (best oracle affine on val worsens MAE). => matching/evidence levers over
  calibration/suppression. N0002's gain was broad, not tail.
- **Protocol (canonical, fixed)**: augment=false, EMA eval, seeded loaders,
  cudnn deterministic, 32ep/1800s. **Seed 20260830 forever**. Open question
  for user only: whether to authorize an augment=true protocol A/B.
- **Server**: `ssh cac-server`, python `/data/miniconda/envs/cac/bin/python`, RTX3060.
  Code sync via `tar | ssh tar` (no rsync). Smoke per node in tmux + capture-pane.
  Watchdogs: none; session names `train-b1/b2`, `smoke-<node>`; kill watchdog first.
- **Gotchas**: (1) `scripts/curve.py <id>` path assumes a DIRECT child of
  N0001_champion; for nested nodes pass `N0002_h0001/<id>` (e.g.
  `curve.py N0002_h0001/N0004_h0003`). (2) Every final idea.md must keep one
  runner-parseable line `1. **Hxxxx** — <booked text>` or tested_hypotheses stays
  [] (bit both batch-1 nodes; repaired). (3) `tar` excludes must precede the dir
  arg. (4) Server ad-hoc scripts need `cac.hub.setup_hf_env()` first (offline HF).
  (5) info.json writes only via run_node/`TrajectoryTree` API — N0002 tested list
  repaired via the API after the runner-bug no-op, journaled.

(End of file - session block)

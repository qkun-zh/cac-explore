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
  TIMEOUT-null (48.12, budget blow — token-max full-grid einsum past τ_max;
  uninterpretable, not a verdict). N0006_h0005 verify (UpCount-style P×V gate,
  H0005) TIMEOUT 23.5802 @ep31, 32/32ep, +1.016 vs 22.56 (bar 22.26 missed by
  1.32) — REFUTED used-and-harmful (prop2 2.24, ver2 3.86, temp→0.021, train
  2.946 vs 2.618); +9.8s overrun marginal/environmental (N0005 parity), best
  plateau-trustworthy; evidence contradicts w=0.85, H0005 conf 0.415.
  Synthesis books use_h2pool (12.5k h2-pooled 3-key bank) + use_simcal (3k
  similarity calibration) next; hires deferred.
- **Batch-3** (on N0002): N0007_h0006 h2pool DONE 47.8439 @ep32 (+25.28 —
  catastrophic joint-interference: shared simprior.qproj reuse polluted the
  confirmed readout; train 13.93 vs 2.62 stall from ep1; same train~14/val~48
  attractor as N0005) — REFUTED, contradicts w=0.95, H0006 conf 0.405.
  N0008_h0007 simcal DONE 27.6371 @ep9 (+5.07, NaN wall ep10-32 step 2249:
  undetached S recompute + std Jacobians; miss decided healthy ep6-9) —
  REFUTED, contradicts w=0.90, H0007 conf 0.410. Family expanded: ANY second
  gradient path into simprior projections is dead (4x: H0004-null, H0005 +1.02,
  H0006 +25.28, H0007 +5.07+NaN). New rule: NEVER share projections / second
  grad paths with the confirmed readout.
- **Batch-4** (on N0002, decoder-side only): N0009_h0008 hires TIMEOUT 26.4535
  @ep31 (+3.89, engaged-harmful: detach insufficient, shared-loss recentering
  dragged temp 11.8x to floor 0.0033) — REFUTED, contradicts w=0.90, H0008
  conf 0.410. N0010_h0009 densexp DONE 24.0500 @ep32 clean (+1.49,
  null-and-dragging: expert never engaged yet temp 9x collapse + broad drift;
  operating-point theory too narrow, runaway spans 4 nodes) — REFUTED,
  contradicts w=0.85, H0009 conf 0.415. Bans now: 2nd-grad-path, shared
  projection, ANY trainable post-decoder additive (even null), global
  calibration. N0002 stands as local optimum.
- **Open questions for user**: (1) augment=true protocol A/B? (2) test-set
  eval of N0002? (3) consolidate vs new scope?
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

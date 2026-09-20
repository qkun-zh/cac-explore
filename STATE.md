# STATE.md

One session block; REWRITE it each session open (archive stale content to
journal/ first). Read AGENTS.md before anything below.

## Session (2026-09-19T15:35)
- **Mode**: Free-Research. Cycles 1-2 closed. **Cycle 3**: N0006_h0012 [H0012
  solo] **DONE** — pre-registered bar 0.40 missed: 21.4952 @ep28 vs champion
  21.4589 (+0.036; mid-run led ~0.13 ep20-27 then tail flattened). Ledger:
  `contradicts w=1.0`, H0012 conf 0.400. Feedback x3 + synthesis written; booked
  H0014/H0015 (below). **N0007_h0013 [H0013 subpixel-up solo] RUNNING** in tmux
  `runN6` (started ~16:08, ETA ~16:38; may touch the 1800s budget -> mark
  `timeout` honestly if it stops at ep31).
- **Tree**: N0001_champion (canonical **23.293 @ep26**). v2-harness children
  (all same-seed paired, augment=false): **N0008_h0014 22.9697 (H0014 SUPPORTS
  0.6, -0.32, bar cleared by 0.024 — fragile)**; **N0009_h0015 23.6818 (H0015
  contradicts 0.4, +0.39)**; **N0010_h0016 22.8831 (H0016 SUPPORTS 0.6, -0.41,
  bar cleared by 0.11; learned gate (a,b)=(0.166,0.189), mild damper)**;
  N0011_h0010 (clean H0010 solo) 23.4581 (contradicts 0.32, +0.165) — exemplar-
  gating family RETIRED (rule 11). v1 history: record only.
  Falsifier bars use live-parent semantics (anchor rule).
- **Confirmations DONE (both FAILED)**: N0010@seed31 21.538 (+0.226) and
  N0008@seed31 21.579 (+0.266) vs same-seed parent 21.3126 — seed-30 supports
  (-0.41/-0.32) do NOT replicate. H0014/H0016 -> 0.48 each. Lesson: mechanism
  effects (~0.3-0.4) sit BELOW seed spread (~+-1); single-seed verdicts screen
  but cannot graduate. Seed FIXED at 20260830 forever (user directive
  2026-09-20, AGENTS §6/§7): no new seeds, no spread runs, no cross-seed
  checks; graduation only via multiple same-seed supports per Eq.1.
- **DAY CLOSE 2026-09-19 ~23:05**: all committed+pushed. NOTE 23:10 — user
  ordered full stop: tmux `spread` killed mid-run (seed20260832 partial
  discarded in scratch, seed20260833 never started), GPU idle, no sessions.
  Seed-spread quantification is UNFINISHED (have 2 points: 23.293/21.313).
  Next session: re-queue spread or decide methodology (A/B/C/D) first.
- **H0016 mechanism separated**: post-hoc damping diagnostic (frozen champion,
  n=1286): raw 23.2909, learned-gate 23.3336, uniform x0.91 24.498 → uniform
  account REFUTED; gate selective but co-adaptation-dependent (neutral evidence,
  no conf move). **Confirmations QUEUED** (tmux `confirm`, scored within-seed
  vs the 21.3126 seed+1 parent, support iff <=21.0126): N0010@seed20260831,
  N0008@seed20260831 (~1h).
- **Server queue after N0007**: (1) smoke N0008+N0009 (new run_node no-ops
  status on smoke); (2) tmux `python scripts/repro_run.py N0001_champion --out
  /data/cac/repro/champion_r2` — **noise-floor diagnostic (determinism on);
  no tree mutation**; (3) run N0008 then N0009 (`--budget-seconds 1800`).
- **Memory**: H0009 0.320, H0010 0.400 (still never cleanly solo-tested),
  H0011 0.400, H0012 0.400 (bar missed), H0013 0.500, H0014/H0015 0.500.
  H0003 0.770 confirmed, H0005 0.215 refuted. Calibration 0.176 @25 tests —
  recompute after N0007.
- **Harness review + fixes (this session, all committed+pushed)**:
  (1) local tree/ was blind to completed N0003/N0004 (info.json stale vs the
  server where run_node wrote them); champion carried migration 19.647 ->
  reconciled from each node's own result.json via machinery writes.
  (2) run_node smoke/--off no longer flips status to done.
  (3) runner honors `deterministic=true` (cudnn deterministic, benchmark off)
  as the pre-migration engine did unconditionally; old-vs-new engine/model/data
  diff: loss, schedule, dataset, model all semantically identical — remaining
  old-vs-new deltas are EMA eval (we use it, old raw), cudnn mode, Lightning
  loop. Historic 19.647 is NOT reproducible in this harness; **21.459 is the
  honest baseline**.
  (4) New `scripts/repro_run.py` replicates a node config into a scratch dir
  (no tree writes) -> noise floor for verdict bars (0.30) never measured before.
  (5) **PROTOCOL v2 (2026-09-19 16:45)**: runner never passed `cfg.augment` or
  `cfg.seed` to FSC147DataModule since the migration -> every v1 run (incl. the
  21.459 baseline) trained with augmentation OFF + unpinned loader RNG; the
  historic pre-migration engine trained augment-on (old train.py L189). Fixed:
  augment/seed/num_workers wired; AGENTS rule 13 = append-only construction
  order (new component built after all parent modules; dropout=0 so no other
  training RNG). **The v1 chain was aborted**; a fresh `N0001_champion` v2 run
  becomes the canonical baseline; v1 numbers (21.459, all child results) stay as
  historical record. Do not silently re-run v1-refuted hyps (rule 11).
  (6) **v2 augmented baseline came out 23.482 @ep16 (+2.02 WORSE than v1)** —
  augmentation hurt at the fixed 32-epoch budget in this harness. A 2x2 protocol
  isolation is IN FLIGHT (tmux `diag`, repro_run --set on the champion):
  known (noaug,EMA)=21.459, (aug,EMA)=23.482; queued (aug,raw=use_ema=false)
  then (noaug,raw). Historic 19.647 likely needs BOTH augment and raw-eval (old
  engine had no EMA) — protocol decision + re-anchored canonical baseline
  immediately after the cells land; no node runs until then.
  **2x2 DONE**: (noaug,EMA)=21.459 (v1), (aug,EMA)=23.482, (aug,raw)=23.605,
  (noaug,raw)=22.782. Augmentation costs ~2.0 at 32ep; EMA helps under noaug;
  historic 19.647 NOT reproducible from the migrated config (README says 30ep;
  the claim predates the micro-tune) -> **canonical protocol = augment=false,
  EMA kept, seeded loaders + determinism**; configs updated (champion +
  N0008-N0011). GPU queue: tmux `canon` = canonical baseline (run_node) ->
  same-seed replicate -> seed+1; tmux `nodes` watchdog = N0008 -> N0009 ->
  N0010 -> N0011 under canonical. Compare everything against the NEW canonical
  baseline number (do not use 21.459 once the new one lands).
- **Paper backlog (researcher 2026-09-19, links fetched-verified; class-agnostic
  counting literature ONLY)**: booked: SAFECount `use_safe_enhance` (2201.08959,
  zhiyuanyou/SAFECount) = H0014; BMNet+ `use_dyn_exemplar_gate` (2203.08354,
  flyinglynx/Bilinear-Matching-Network) = H0015. Un-booked next: DAVE-lite
  `use_verify_mask` (2404.16622, jerpelhan/DAVE); LOCA-lite prototype adapt
  (2211.08217, djukicn/loca); SQLNet-lite size prompt (2311.10011,
  HCPLab-SYSU/SQLNet). Rejected on fit: CounTR / GeCo / PSECO.
- **Gotchas**:
  - run artifacts gitignored (`**/run/`, tmp_ideas_round2/); never `git add -A`
    blindly. result.json lives at the NODE ROOT, run_node.log under run/latest/.
  - Server copy /data/cac is NOT a git repo — sync src/scripts/tree via
    tar-over-ssh; Python = /data/miniconda/envs/cac/bin/python; tensorboard only
    in that env (EventAccumulator works there).
  - `--book H --solo` = one-hyp inference; `--book H` = mandated + 1 compatible
    adjoin; `--new` auto-mandates. Read idea.md BEFORE trusting a node's hyps.
  - 32 epochs expected; the 1800s budget has only ~30-60s headroom — a run can
    stop at ep31 and must be recorded `timeout`.
  - smoke no longer flips node status (restores pre-run status).
  - Protocol pins now live in DURABLE docs (internalized 2026-09-19 pm):
    AGENTS §6 anchor rule + paired-contrast doctrine, §5 rule 14 (journal ONLY
    via scripts/journal.py), §7 canonical-protocol + GPU-queue/kill-order rows,
    conformance protocol pins on the runner wiring. STATE keeps live status
    only; anything load-bearing belongs in AGENTS/code/tests.
  - HF offline-first (hub.setup_hf_env() at top of entrypoints).
  - CAC = Class-Agnostic Counting (FSC147 few-shot: count ANY class given
    exemplars), NOT crowd/people counting. Hypothesis inspiration must come from
    the class-agnostic counting literature (LOCA/CounTR/DAVE/GeCo/PSECO/
    SAFECount line), never crowd-counting papers. (User correction 2026-09-19.)

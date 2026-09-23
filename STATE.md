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

# STATE — session 2026-09-21 (server down, no-GPU research day)

- **No training run** (server unreachable). Three local analyses → one fused
  draft, all in `local/research/` + `local/ideas/` (gitignored, zero ledger/tree
  writes; `memory/index.json` timestamp-only touch restored).
- **runaway_meta.md**: temp is the highest-leverage scalar (dW/dtemp ∝ 1/temp²);
  below ~0.02 top1≈consensus locks in. One attractor, 3 flavors
  (interference-collapse N0006/N0008, starvation-stall N0007, loss-coupling
  N0009/N0010). Escapes: two-phase frozen-readout, temp clamp/floor, frozen
  readout (all argued regime-compliant: subset-freezing is stricter than
  head-only; temp clamp lives in node model.py, not §10 machinery).
- **frozen_addon_survey.md**: 9 methods (Side/Ladder-Side-Tuning, adapters,
  ControlNet, LoRA, Progressive Nets, LP-FT, EWC, L2-SP, CLIP temp clamp).
  Top pick: Ladder Side-Tuning + temp freeze (≥0.05 floor). Sharpest rule:
  zero-init is necessary but NOT sufficient — **any trainable temp dies
  before smoke** (N0009 proved it).
- **n0002_success_anatomy.md**: N0002's +0.73 lives in D10 (gt≥139, 59% of
  net; top-3 images = 47%); hurt is diffuse mid-count under-deepening;
  per-image oracle headroom −1.53 (21.03).
- **Draft ready**: `local/ideas/sidetune_frozen.md` — `use_sidetune` hypothesis
  (novelty 0.518 green, validate clean) + shuffled-exemplar zero-training
  control sketch (N0002 booking, never run) + go/no-go checklist. Honest flag:
  0.05 floor sits above parent 0.0393 → step-0 not bit-identical (1.27×
  softening, inside 2× guard).
- **Server-return queue**: preflight → run shuffled-exemplar control (minutes,
  inference-only) → book `use_sidetune` via `hypo --new --solo` → Idea/Coding
  subagents → smoke → train.

# STATE — session 2026-09-21 (foundation prep; GPU still OFFLINE)

- **Directive (user, overrides default loop)**: leaderboard numbers FIRST, then a
  paper from those numbers. Priorities: (1) no training cheating, (2) real
  frontier-paper backing, (3) publishable novelty. Old head-on-head exploration
  frozen until the v2 baseline exists.
- **Protocol v2 (user-authorized)**: canonical = **augment=true** + standard
  train-time augmentation + test-split evaluation. Reference:
  `configs/protocol_augment.toml`. The 22.5641 (v1, augment=false) stays as
  historical parent for future bars but every comparison now re-instantiates
  against the v2 baseline (AGENTS §6 bars semantics).
- **Local groundwork DONE (no GPU needed)**:
  1. `src/cac/data/fsc147.py` — `_augment_scale_flip`: seeded random scale
     [0.6,1.25] w/ center placement back to 384 + random hflip, box/density
     joint affine, sum-conserving. Verified on CPU: deterministic at seed,
     box-vs-density centroid err <1px on both pad/crop branches. Dataset now
     emits `ids` (for per-image dumps).
  2. `scripts/eval_test.py` — scores a node's `run/latest/best.pth` on the
     FSC147 test split (augment=false, EMA weights = canonical): writes
     `test_result.json` (mae/rmse/bias/tail-summary + checksums) and
     `test_perimage.json` (preds/gts/ids). Accepts node id or path; mirrors
     run_node env (hub.setup_hf_env). Verified importable on CPU env.
  3. `configs/protocol_augment.toml` — read-only v2 canonical reference.
- **Server-return queue (GPU up ⇒ execute in order)**:
  1. preflight + smoke per node
  2. v2 baseline: `repro_run.py N0002_h0001 --out /data/repro/baseline_aug_v2
     --set augment=true` (budget 1800s) → v2 canonical val baseline
  3. **THE innovation card (user directive: skip test-split focus, val is the
     currency):** `run_node.py N0011_h0010 --set augment=true` — H0010
     count-anchored similarity calibration (`use_counttau`, +2 params, step-0
     byte-identical). Verdict = same-seed v2 pair vs the v2 baseline: val MAE
     ≤ baseline−0.30 (bar 22.26 pre-v2, re-instantiated) AND gt>500 dense-tail
     mean |Δ| < v2 baseline (pre-v2 511.7). Diagnostic read: w_g>0 (mass on),
     |log(τ'/τ)|<2 (no floor-collapse), qproj/kproj/temp trajectories match
     parent (harness gate).
  4. eval_test is optional/after (user demoted: val≈test, don't burn the queue).
  5. **STATUS 2026-09-21: GPU ONLINE** (new VM a7m6si2jjnzw48rdsnow.deepln.com:50020
     via install_key.py; local/address_and_password.md updated; server has no
     .git so sync = tar-over-ssh excluding .git/run/local). Code committed
     256f8b5 + synced; server conformance OK; v2 pair RUNNING via
     `setsid nohup chain_v2.sh` (tmux unavailable in apt): repro baseline_aug_v2
     then innov_counttau_v2, logs /data/repro/logs/. Poll once per message,
     never sleep-loop. Minor: fsc147.py:128 numpy non-writable warning
     (cosmetic; add .copy() later).
  6. **V2 BASELINE LANDED (journaled): 21.4362 @ep32** (32/32, budget_hit false,
     1730s, /data/repro/baseline_aug_v2/repro.json). Augment alone: -1.13 vs v1
     22.5641. **H0010 bars re-instantiated: innov val MAE <= 21.1362 AND
     gt>500 dense-tail mean |Δ| below v2 baseline tail.** innov_counttau_v2
     auto-started (cfg override augment=true confirmed, smoke ok, pinned-temp
     patched model synced before its start).
  7. **Next cards queued in research (not booked):** H0011 draft
     `local/research/h0011_msq_draft.md` (use_msq: single-step h2 query lane,
     GeCo2 PrototypeAttentionBlock semantics verified from repo source, gradual
     residual fusion; books only after H0010 verdict).
  8. **BREAKTHROUGH #1 2026-09-21: H0012 CONFIRMED (supports w=0.8, conf 0.58).**
     use_cellcal (+1 param, per-cell self-normalized energy gain on confirmed
     evidence): v2 val MAE 21.4362 -> **20.8861** (d=-0.55, first bar clearance)
     and dense tail 423.40 -> **421.13**. w_c=+0.1658; bias/RMSE improved too.
     Caveat: sparse gt<50 slice +0.32 (P2 guard +0.10 missed — texture
     amplification); dense-block still reshuffles (975/3425/6969/3665 down,
     3433/3488/840 up). **New live parent N0013_h0012 (20.8861); next bars
     re-instantiate at <=20.5861 + dense < 421.13.** Next card direction:
     kill sparse false positives while keeping dense gain (F6
     negative/background-prototype family — the survey's untried lever that
     directly targets false positives). Futility rule 16 now guards all cards.
  9. **BREAKTHROUGH #2 + HONESTY VERDICT 2026-09-21: H0014 MECHANISM REFUTED
     (contradicts w=0.85, F1 quiet-null) DESPITE 19.3431 (d=-1.56, dense -90).**
     use_peakcal NEVER engaged (w_p=0.0017, gain~=1): the discount did nothing.
     Attribution (inferred, not ablated — GPU discipline): the win rides on
     temp-pin HYGIENE (N0015 temp pinned exactly 0.07 vs N0013 temp drifted to
     0.0413 = 7th collapse-corpus instance; w_c 0.148 vs 0.166 similar).
     Rule established: never credit a null mechanism for hygiene's win; pin is
     mandatory hygiene on every card from here. **Live parent N0015_h0014
     (19.3431, peakcal = dead code in it); next bars <=19.0431 + dense <330.81.**
     Sparse guard missed again (5.814 vs 5.45) — sparse false positives remain
     THE open target. Commit: honesty over headlines.
  10. **SERVER RELEASE 22:22 2026-09-21 — SESSION CLOSED.**
     H0021 exkern REFUTED (contradicts w=0.85, futility-halt ep24 + F1
     quiet-null w_x=-0.04): 21.928 (+2.59), dense +123.6. CountingDINO transfer
     fails under joint training. CountingDINO pipeline itself REPRODUCED
     (ViT-S cached, training-free val 39.70; aligned tail table: cdino beats
     live by 170-370 on 935/865/5059/3481/3482/2850/3483/7656 — paper-grade
     evidence that the failure is training dynamics, not features).
     FINAL NUMBERS: val 19.3431 (N0015) / test 19.066 (N0015) — both sub-20.
     Local backup /home/qkun/cac_backup/: N0015_best.pth (125MB, verified),
     N0015 val+test perimage/results, baseline v2 val perimage, cdino preds.
     Git pushed clean (conformance green). NEXT SESSION: paper phase (numbers +
     diagnosis corpus + sub-cell physics + collapse corpus + futility
     machinery) OR new program if user lifts a constraint. GPU queue EMPTY.
- **Paper framing (numbers-first)**: lightweight frozen-backbone counting
  (CounTR-class) + mechanism contribution (exemplar-selection entropy collapse
  corpus from the v1 refutations). FSC147-adjacent SOTA (LOCA 17.13 val) is out
  of reach under the frozen-32M regime; the paper cuts are same-protocol wins +
  the collapse diagnosis.

# STATE — session 2026-09-22 (rescue; decoder→similarity cards; N0027 in flight)

- **Directive (user)**: achieve val < 18; act autonomously; never mention paper unless goal hit; budget open.
- **Live parent unchanged: N0015_h0014 val 19.3431 / test 19.066.** Bars ≤19.0431 + dense <330.81 + sparse ≤5.45.
- **Precision (settled this session)**: same-seed draws {19.34, 20.70, 21.59} mean 20.55 sd~1.13; champion is a lucky draw; ±2.25 same-seed floor exceeds remaining 0.30-bars. 3-seed still needs user approval (seed-fixed rule).
- **H0024 CountAnchor (N0025) REFUTED** (futility ep24, val 22.2127 +2.88, dense 454.05, sparse 5.779, test 19.437; contradicts w=0.85 conf 0.415). Feedback ×4 + synthesis (0 bookings). Response×post-out magnitude fork closed.
- **H0025 CapFilm (N0026) REFUTED** (futility ep24, val 23.223 +3.88, dense 508.54 +177.7, sparse 6.104, test 19.587; contradicts w=0.85 conf 0.415). Feedback ×4 + synthesis (0 bookings). Decoder FiLM consumption site closed at this form.
- **H0026 GemeCal (N0027) BOOKED + RUNNING**: geometry ME scale f=(me/32)^w_g on SimPrior ev post-gain pre-out, +1 param, use_geme; novelty 0.601; smoke ok (cfg c48cbec8d36450f4, mdl f8b0470a4542fd92); futility_bar=19.0431; log /data/repro/logs/n0027_run.log. ep16 WARN best 24.0491 (need 0.313/ep) — on track for ep24 HALT unless late surge.
- **Research (no booking)**: h0027/h0028/h0029 drafts all NO STRONG CANDIDATE / NO-GO until N0027 lands. TT-norm rejected for training (post-decoder/global-cal bans). ExDiv = expected-negative fallback only.
- **Offline probes** (`local/research/offline_probes_20260922.md`): CounTR shrink worsens; gt-free pred-binned affine oracle 17.93 (−1.40, overfit risk); gt-binned scale 17.06; wipes 11/38 confirmed, cdino wins 11/11 wipes but loses globally; path to <18 needs train-time count-conditional calibration (−2.3…−7.5 oracle band), not post-hoc shrink.
- **Queue after N0027**: verdict+feedback+synthesis → if fail, ExDiv only if forced else keep searching legal count-conditional train-time actuator; if confirm, re-parent bars at new live.
- **Gotcha**: local python3 has no torch/cac — all GPU/eval/smoke on server via `/data/miniconda/envs/cac/bin/python`; sync = tar-over-ssh excluding .git/run/local.


## 2026-09-22
- Focus: training-free N0029_tf_champion; eval gate subset286.
- Archive: `archive_local/README_ARCHIVED.md`.

# STATE — session 2026-09-23 (TF rapid-goal update)

- **User directive:** success bar = **subset286 MAE ≤ 20.0** (rapid iteration); run the full TF works autonomously. Champion/board rows remain full val 1286 only.
- Docs updated same-session: `tree/N0029_tf_champion/{constraints_and_goal,baseline_board,HANDOFF_TF}.md`.
- **Server rotated:** `ssh -p 46315 root@beuqr6fzapk1bhxosnow.deepln.com` (password in `local/address_and_password.md`); old hzpcqeuyl…:52662 entry replaced. Host key scanned into known_hosts; BatchMode preflight pending password path (no sshpass yet).
- Local conformance OK at session start; git clean vs origin (`f1d5c55`).
- Rapid gap: subset286 best **26.339 → 20.0 = −6.34**. Structural queue unchanged (P2 transductive / TTA / dense-bin focus — no more fs micro-tuning).
- **Docs language lock (user 2026-09-23):** all project documentation must be precise English only; Chinese forbidden (constraints_and_goal.md §5b).


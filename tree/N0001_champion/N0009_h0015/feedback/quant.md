# Quantitative feedback — N0009_h0015 (H0015 solo, dyn exemplar gate)

Scope: numbers only. No verdict (ledger already records the outcome).
Child = `tree/N0001_champion/N0009_h0015` (H0015 SOLO, `use_dyn_exemplar_gate`).
Parent = canonical N0001_champion re-run under protocol v2
(noaug + EMA + seeded loaders + cudnn deterministic, seed 20260830).

Provenance: child `result.json` is identical local vs server. Parent canonical
numbers below are read from the SERVER copy (`/data/cac/tree/N0001_champion/`,
`result.json` + TB `events.out.tfevents.1789812726...23038.0` via
EventAccumulator in the cac env) — the LOCAL `tree/N0001_champion/result.json`
(21.4589@ep32) and local `run/latest/tb` are stale v1 artifacts, NOT the
canonical baseline. Child local TB (`...1789819679...29630.0`) is byte-identical
in name and values to the server copy (synced). All TB values below are
`val/mae` per-epoch scalars (4dp as logged); result.json full precision quoted
in the table.

## 1. Final table (head-to-head, same seed 20260830, same protocol)

| field | child N0009_h0015 | parent N0001_champion (canonical) |
|---|---|---|
| best val MAE | 23.681793212890625 @ep23 | 23.293153762817383 @ep26 |
| last val MAE (ep32) | 23.8680 | 23.3364 |
| epochs done | 32/32 | 32/32 |
| budget_hit | false | false |
| elapsed (runner train) | 1698.27 s | 1694.12 s |
| elapsed_s (wall) | 1703.5 s | 1699.4 s |
| config_sha256 | 3fd5536c2d575d19 | 679f37c26bc72103 |
| model_sha256 | dd4cbfdaa685e386 | eaf2a8b93c280cba |
| best.pth sha256 (server) | df439c05…eaeb9b1 | 60eb392b…09fc17b82 |
| best.pth size (server) | 125,516,815 B | 125,383,139 B (+133,676 B child) |
| val RMSE at best-MAE ep | 90.8771 @ep23 (own RMSE best 90.8044 @ep19) | 90.8030 @ep26 (own RMSE best 90.6424 @ep24) |
| train loss_epoch, ep32 | 3.0339 | 2.8345 |
| status | done | done (canonical re-run) |

Headline deltas (child − parent): best **+0.3886** (worse); last **+0.5316**.
Wall delta: **+4.1 s** (+0.24%) — negligible gate overhead.

Config diff parent→child is exactly one switch (verified in both hparams.yaml
snapshots on server): `use_dyn_exemplar_gate: false → true`; every other key
identical (seed 20260830, deterministic, augment=false, AdamW lr 1e-3 wd 0.08,
cosine eta_min 1e-6, MSE+0.4·L1, batch 16, 32ep). Single-switch test holds.

## 2. Epoch-by-epoch val MAE, child vs canonical parent (delta = child − parent)

| ep | child | parent | delta |
|---|---|---|---|
| 1 | 139.6814 | 146.9976 | −7.3162 |
| 2 | 70.4448 | 69.0238 | +1.4210 |
| 3 | 50.5364 | 49.4376 | +1.0988 |
| 4 | 43.9792 | 39.5353 | +4.4439 |
| 5 | 34.9486 | 34.1235 | +0.8251 |
| 6 | 31.2465 | 30.6235 | +0.6230 |
| 7 | 28.7373 | 28.5269 | +0.2104 |
| 8 | 27.2465 | 27.1416 | +0.1049 |
| 9 | 26.1466 | 26.0856 | +0.0610 |
| 10 | 25.4242 | 25.3461 | +0.0781 |
| 11 | 24.9511 | 24.7792 | +0.1719 |
| 12 | 24.6700 | 24.3675 | +0.3025 |
| 13 | 24.4100 | 24.2335 | +0.1765 |
| 14 | 24.3145 | 24.0380 | +0.2765 |
| 15 | 24.1824 | 23.8607 | +0.3217 |
| 16 | 24.0414 | 23.9168 | +0.1246 |
| 17 | 23.9911 | 23.8016 | +0.1895 |
| 18 | 23.9199 | 23.7184 | +0.2015 |
| 19 | 23.7936 | 23.6163 | +0.1773 |
| 20 | 23.7874 | 23.5276 | +0.2598 |
| 21 | 23.7303 | 23.5090 | +0.2213 |
| 22 | 23.7103 | 23.4600 | +0.2503 |
| 23 | 23.6818 | 23.3879 | +0.2939 |
| 24 | 23.6820 | 23.3307 | +0.3513 |
| 25 | 23.6865 | 23.3257 | +0.3608 |
| 26 | 23.7008 | 23.2932 | +0.4076 |
| 27 | 23.7183 | 23.3194 | +0.3989 |
| 28 | 23.7575 | 23.3075 | +0.4500 |
| 29 | 23.7910 | 23.3070 | +0.4840 |
| 30 | 23.8237 | 23.3221 | +0.5016 |
| 31 | 23.8506 | 23.3341 | +0.5165 |
| 32 | 23.8680 | 23.3364 | +0.5316 |

Reading:

- Leads exactly once: ep1 (−7.32). From ep2 on the child never leads again
  (31/32 epochs behind).
- Worst early gap ep4 (+4.44): parent drops 49.44→39.54 while child stalls
  50.54→43.98 — the gate costs one full epoch of early descent that is never
  recovered.
- Closest approach after ep1 is ep9 (+0.061); mid-run (ep7–ep23) the gap sits
  in a +0.06…+0.32 band — persistent, same-sign, never crossing.
- Best@23 then drift: child best 23.6818 @ep23, then 9 straight non-improving
  epochs, monotonic upward ep23→32 (+0.186 total:
  23.6820 / 23.6865 / 23.7008 / 23.7183 / 23.7575 / 23.7910 / 23.8237 /
  23.8506 / 23.8680). No late recovery.
- Parent tail: best@26, drift ep26→32 only +0.043 with a flat bounce
  (23.3194 / 23.3075 / 23.3070 / 23.3221 / 23.3341 / 23.3364) — plateau, not
  divergence. Tail gap widens +0.35 (ep24) → +0.53 (ep32).
- RMSE tells the same story at smaller amplitude: child best 90.8044 @ep19 vs
  parent best 90.6424 @ep24 (Δ +0.162); child RMSE plateaus ~90.8–91.5 while
  parent reaches lower and holds it.
- Train fit gives the gate no credit either: ep1 child train loss starts much
  lower (38.31 vs 91.30, Δ −52.99) yet by ep9 the sign flips, and from ep11 on
  the child train loss is almost uniformly HIGHER (ep23 +0.06, ep32 +0.20:
  3.0339 vs 2.8345). Extra capacity, worse late train fit — optimization
  friction, not overfitting from fitting harder.

## 3. Bar distances

- Live-parent anchor rule (AGENTS §6): bar = parent_best − 0.30 = **22.9932**.
  Child best 23.6818 misses by **+0.6886** (needs −0.69 to reach the bar).
- Head-to-head: **+0.3886** worse than parent best — larger in magnitude than
  the 0.30 bar width, in the wrong direction.
- Last-epoch gap **+0.5316** exceeds the best-epoch gap: the reported +0.3886
  is the flattering endpoint; the run ends further behind than its best.
- Scale: paired-contrast precision under this harness is ~0.02, so +0.3886 is
  ~19× precision — not a noise-scale wiggle. (Cross-seed level noise is ~±1
  MAE per AGENTS §6, so a single-seed level claim would still need multi-seed;
  the paired same-seed gap is what it is.)
- Literal booked text (idea.md) cites the v1 anchor 21.459 → line 21.159;
  under the falsifier-semantics rule the live line is 22.9932 (both numbers
  stated here). The child misses both: +2.5228 vs the literal line, +0.6886
  vs the live line.

## 4. Wall-clock

- Child 1703.5 s vs parent 1699.4 s: **+4.1 s (+0.24%)**, ~53.2 s/epoch both.
  The 256→64→256 per-exemplar MLP is compute-invisible at this budget.
- Headroom: both runs clear the 1800 s ceiling with ~96–100 s to spare;
  32/32 epochs, `budget_hit=false` both — full-horizon comparison, no
  stopped-clock asymmetry.
- Runner accounting is consistent: `elapsed_s − elapsed` = 5.2 s (child),
  5.3 s (parent) — same overhead on both sides.

## 5. Internal consistency checks

1. TB argmin == result.json: child TB min 23.6818 @ep23 == best_mae
   23.681793212890625 @ep23 ✓; parent TB min 23.2932 @ep26 == best_mae
   23.293153762817383 @ep26 ✓.
2. Horizon complete, no budget stop: 32 val/mae scalars each, budget_hit
   false, elapsed < 1800 ✓.
3. Single-switch: server hparams.yaml files differ ONLY in
   `use_dyn_exemplar_gate` ✓.
4. Model hash moves exactly where expected: parent model_sha256 eaf2a8b9 is
   IDENTICAL between the stale local v1 record and the canonical server run
   (parent architecture untouched); child dd4cbfda differs (gate module
   added) ✓.
5. best.pth cross-check: distinct sha256; size delta +133,676 B ≈ 33,088 fp32
   params × 4 B = 132,352 B plus optimizer/housekeeping — matches a
   256→64→256 MLP (256·64+64 + 64·256+256 = 33,088 params) ✓. Child
   checkpoint is the gate run, not a copy of the parent.
6. Local-sync warning: child local TB filename/values == server copy ✓, but
   LOCAL parent `result.json` (21.4589) and local parent TB
   (`...1789796237...`, curve 157.32→21.4588@ep32) are pre-canonical v1 —
   any local-only read of the parent is the wrong baseline; use the server
   canonical ✓.
7. Log hygiene: `train/mae` scalars are NaN on both runs (logged, never
   filled — pre-existing, symmetric); `hp_metric` −1 placeholder both; LR
   schedule identical (cosine 1e-3 → ~3e-6) ✓.
8. Init-at-identity held at step 0 by construction (2·sigmoid(0)=1.0, last
   layer zero-init) yet ep1 already differs on both train (−52.99) and val
   (−7.32): gradients through the gate bite from the first step, so "starts
   at champion" lasts exactly zero updates — consistent with the ep2+
   permanent deficit, noted here as arithmetic, not mechanism.

## 6. So what

- The +0.3886 is a full-horizon, same-seed, single-switch deficit that grows
  toward the tail (+0.53 at ep32): there is no epoch subset, early-stop, or
  last-vs-best choice that rescues the comparison.
- The tail shapes diverge qualitatively (child monotonic +0.186 over 9 dead
  epochs vs parent flat +0.043): the gate's cost concentrates late, and the
  train-loss sign flip (child worse from ep11) rules out "fits harder,
  generalizes worse" as the whole story.
- The bar arithmetic is not close (miss by 0.69 ≈ 2.3× the bar width, ~19×
  paired precision): any follow-up on this switch would need a new falsifier
  and a mechanism addressing the ep4 descent stall plus the late drift, not a
  re-run of the same gate.

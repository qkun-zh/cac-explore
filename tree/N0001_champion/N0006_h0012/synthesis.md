# N0006_h0012 — Synthesis (H0012 solo on N0001_champion)

Synthesis Agent, AGENTS.md §4 step 8. Sources read directly: `AGENTS.md`, `STATE.md`,
`feedback/{quant,qual,causal}.md`, `result.json`, `info.json`, `idea.md`, `model.py`,
`config.toml`, `memory/hypotheses.jsonl` (H0012 rows, lines 37 and 41),
`memory/index.json` (H0012 block, lines 377–398), plus a fresh
`discovery calibration` run. No commits, no node creation, no writes to `memory/`.
Confidence math is the machinery's (`memory/index.json`); nothing here recomputes it.

## 1. Outcome

At seed 20260830 the solo count-temperature child **N0006_h0012** completed
**32/32 epochs with `budget_hit = false`** (1771.0 s wall of the 1800 s τ_max, 29.0 s
headroom; trainer loop 1765.6 s vs the parent's 1699.1 s) and finished with best val MAE
**21.49515724182129 at ep28**; parent **N0001_champion** stands at
**21.458858489990234 at ep32**. That is a best-vs-best delta of **+0.0363 in the wrong
direction** (child marginally worse) and a last-epoch delta of **+0.2129** (child ep32
21.6718 vs parent ep32 21.4589), against a pre-registered support bar of
21.4589 − 0.40 = **21.0589** — the observed best sits **+0.4363 above the bar**, so even
oracle early-stopping at the child's own global best (ep28) cannot reach it; the run added
**+24,705 params** (31,349,476 vs 31,324,771, +0.0788%, under the 32 M cap).

## 2. Consolidated findings (deduped)

1. **Bar and endpoint.** Child best 21.4952 @ep28 vs champion 21.4589 @ep32 =
   +0.0363; distance to the 21.0589 bar +0.4363; last-epoch delta +0.2129; 32/32
   epochs, budget not hit (quant §1, §3; causal §2).
2. **Curve shape: transient lead, then an overfit-shaped tail.** Child led 12/32
   epochs with a 9-epoch window ep20–28 (mean Δ ≈ −0.19, peak −0.3563 at ep23) and
   trailed ep3–19 (up to +0.7915 at ep8); it peaked at ep28 then gave back +0.1766 by
   ep32 while the champion was still falling monotonically to its best at ep32
   (quant §2; causal §2). Train loss was lower for the child in 17/32 epochs and at
   the tail (ep32 2.5294 vs 2.7062) — better train fit with worse EMA val at the end is
   a capacity/overfit signature, not an optimization failure (qual §5, §6).
3. **What the component actually is.** `use_count_temp` is an image-descriptor
   attention temperature, not a count prior: `tau = clamp(1 + MLP(GAP(fine), e_mean),
   0.25, 4.0)`, one scalar per image shared by all 4 heads and all 9216 query positions,
   softmax over K=3 exemplar tokens; no count head, count input, or count target attaches
   to `z`, and only `out["density"]` feeds the loss (qual §1). Zero-init makes the
   switch-on path identity vs the champion at the first smoke step only — the runner
   smoke-trains 2 epochs × 4 batches before the timed 32-epoch fit (qual §2).
4. **The information channel was already consumed by GCA.** `CountPrior` reads exactly
   `GAP(fine)` + `e_mean` with the same MLP shape (384→64→1, zero-init last layer) as GCA,
   which is directly count-supervised through the `w_cnt = 0.4` L1 on the density sum;
   H0012 therefore tested the *marginal* value of a second count pathway on attention,
   not the premise that a fixed temperature "leaves error on both tails uncorrected"
   (qual §3; causal §3b). No GCA-off config exists anywhere in the tree, so the
   redundancy explanation is untested.
5. **Not identified in the other direction either.** TB carries no `z`/`tau`/entropy or
   per-image scalars (`hp_metric, lr-AdamW, train/loss, train/loss_epoch, train/mae,
   val/mae, val/rmse, epoch`), so whether τ collapsed to identity, saturated at a clamp
   bound, or tracked anything is unknown; a bias-dominated constant solve is possible
   (qual §5; causal §3a, §4).
6. **Run hygiene: clean ablation, not an RNG-paired A/B.** Switch-off reproduces the
   champion graph and the diff surface is exactly the component plus `use_count_temp`
   (qual §4), but `CountPrior` is constructed before `GCA`, so the child's global RNG
   offsets GCA's hidden-layer init and the shuffle/flip streams (qual §4; causal §1 —
   data order itself is unaffected, the datamodule seeds its own generator). Consistent
   with this, τ ≡ 1 at init yet ep1 train loss is 65.98 (child) vs 50.30 (champion), so
   per-epoch deltas mix mechanism with trajectory divergence (quant §2). Both runs
   predate the `deterministic=true` engine fix (commit 72bf8e4) (qual §4).
7. **Cost.** Wall +66.5 s (+3.90%) is ~50× the +0.0788% parameter share and orders of
   magnitude beyond the added arithmetic; it is not attributable to the mechanism at
   n=1 without an interleaved control, and it is not on the causal path (quant §4;
   causal §3c). The child ran with only 1.61% budget headroom vs the parent's 5.31%.
8. **Noise floor still unmeasured.** The queued same-seed `repro_run.py` bounds engine
   determinism, not seed-to-seed spread; the ep20–28 lead (≈0.19–0.36) is the same order
   as plausible seed variation, so no sub-0.30 delta should be read as signal yet
   (quant "So what"; qual §6; causal §5 cross-cutting).
9. **Artifact integrity.** `result.json.best_mae` == min over 32 TB `val/mae` scalars;
   `best_epoch` 28 == argmin; server `best.pth` holds exactly the 4 `count_prior`
   tensors with the expected shapes; server `result.json` byte-identical to local;
   checksums re-derived (quant §5).

## 3. Mechanism verdict (ledger-consistent)

The ledger stands as written: `memory/hypotheses.jsonl:41` records `contradicts` w=1.0
on 2026-09-19T16:05:07+0800 from N0006_h0012; H0012 is **confidence 0.400**
(memory/index.json:377–398, status `uncertain`), exactly as stated in the node inputs.
Verdict language: **H0012's pre-registered bar failed decisively** — this implementation,
solo on N0001_champion, 32 epochs, seed 20260830, does not produce a ≥0.40 val-MAE
improvement (observed best +0.0363 worse). What this run does **not** license is
"H0012's mechanism is false": τ/count/attention-entropy were never measured, and the
tested channel duplicates GCA's already count-supervised inputs (causal §0, §3a, §3b;
qual §1, §5).

**Explicitly untested by this node:** (i) true count-supervision of the prior — no
direct count objective attaches to `z`, and the MLP re-read a signal already supervised
through GCA (qual §3; causal §3b); (ii) per-head or per-position temperature — one
image-global scalar over a 3-way softmax at 9216 positions has little expressive room
(qual §1, §6; causal §5 ranks this last); (iii) any behaviour without GCA (causal §3b);
(iv) whether τ collapsed/saturated or encoded count at all (qual §5; causal §4 proposes a
cheap pre-registered τ-audit on the server checkpoint as diagnostic, not evidential).

## 4. Calibration (verbatim, `PYTHONPATH=src python3.13 scripts/discovery.py calibration`, run this session)

```text
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      9       0      0%       0.36
[0.50,0.75)     18      10     56%       0.55
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall         27      10     37%      -
reliability error (weighted |rate − pred_conf|): 0.126
direction accuracy (decisive tests, c≠0.5): 15/15 = 100%

=== Current Standings ===
hyp       confclass       tests
H0001    0.664uncertain       2
H0002    0.664uncertain       2
H0003    0.770confirmed       4
H0004    0.664uncertain       2
H0005    0.215refuted         4
H0006    0.336uncertain       2
H0007    0.269uncertain       3
H0008    0.269uncertain       3
H0009    0.320uncertain       2
H0010    0.400uncertain       2
H0011    0.400uncertain       1
H0012    0.400uncertain       1
H0013    0.500uncertain       0
```

Interpretation: reliability error is now 0.126 (was 0.176 in STATE.md; under the 0.2
gate) with perfect 15/15 direction, but the [0.25,0.50) bin still over-predicts
(predicted 0.36 vs 0% observed over 9 tests), so the H0009–H0012 uncertainty band is
where the ledger is most optimistic.

## 5. BOOKING RECOMMENDATIONS

**Parent choice: N0001_champion.** Verified this session: `discovery parent` returns
N0001_champion (score 0.697 vs N0006_h0012 0.647; N0007_h0013 still `proposed`/in flight
and cannot displace it on current numbers). Both bookings are `--solo` single-hypothesis
children on the champion, using only the pre-validated candidate texts (A/B), verbatim.
Execution note for the Lead: re-run `discovery parent` at booking time; the commands below
hardcode N0001_champion as instructed.

### Booking 1 of 2 — (A) SAFECount dual-normalized similarity (`use_safe_enhance`)

Chosen because it is the top-ranked fetched paper-backlog item (STATE.md:36–38) and it
attacks exactly the interface H0012 failed on: instead of one globally shared scalar it
injects a **per-location, per-exemplar match decision** into the feature the decoder sums,
and it does **not** re-read `GAP(fine)+e_mean` — the channel that qual §3 and causal §3b flagged
as already count-supervised by GCA. The novelty gate shows it is unrelated to every
refuted mechanism (top sim 0.360 vs H0009; no structural twin), so no `contradicts`
booking applies.

```bash
PYTHONPATH=src python3.13 scripts/discovery.py hypo N0001_champion --new "IF we bolt a similarity-aware enhancement block onto the fused fine feature map that projects the fine map and the exemplar tokens into a shared score space, softmax-normalizes each query-exemplar similarity map jointly across the exemplar axis and the spatial axis, and adds a similarity-weighted mix of value-projected exemplar tokens back onto every spatial location as a residual before the map enters the condenser IN the CountingHead of the champion counter THEN final val MAE is at least 0.30 lower than the champion without the enhancement BECAUSE the condenser gives every location the same exemplar attention distribution, so background patches that weakly match any prototype still pass feature evidence to the density head, while a dual-normalized similarity map selects which prototype matches at each location and sharpens the locations where that prototype matches, injecting the match decision into the feature the decoder sums and leaving the residual features untouched where no prototype matches, which cuts false-positive density on clutter. DISPROVED IF final val MAE is not at least 0.30 lower than the champion N0001 seed of 21.459, that is, final val MAE is above 21.159" --solo
```

### Booking 2 of 2 — (B) BMNet+ dynamic per-exemplar channel gate (`use_dyn_exemplar_gate`)

Chosen because it is the **per-exemplar steering** mechanism the lineage has never tested
cleanly: H0009's image-global channel gate was refuted (22.6076, Δ −1.149), and H0011
showed that gate's benefit was inert capacity; H0010 (per-exemplar scalar) was only ever
run joint-confounded (N0004) or condition-severed (N0005). B gives each exemplar its own
per-channel mask before the condenser keys — a strictly finer intervention than anything
refuted — and it reads each exemplar token, not `GAP(fine)+e_mean`. It is a **new id**
(new 0.30 bar vs the 21.459 seed), not a retry of H0009/H0010: novelty gate top sim 0.531
vs H0010, below 0.82, and no structural twin. So quality gate (c) requires no
`contradicts` event for it.

```bash
PYTHONPATH=src python3.13 scripts/discovery.py hypo N0001_champion --new "IF we add a small exemplar-conditioned gating MLP that maps each exemplar embedding to its own per-channel weight vector and multiplies that vector into the channels of that exemplar's token before the token enters the condenser cross-attention IN the CountingHead of the champion counter THEN final val MAE is at least 0.30 lower than the champion without the gate BECAUSE every exemplar token is a pooled ROI whose channels mix object appearance with that box's background and illumination, and one fixed or image-global gate cannot choose a different channel subspace for each exemplar, so a per-exemplar channel mask lets the condenser matching keys carry the object-bearing channels of that specific prototype and down-weight the context channels, which reduces both missed matches for unusual exemplars and false matches on textured background. DISPROVED IF final val MAE is not at least 0.30 lower than the champion N0001 seed of 21.459, that is, final val MAE is above 21.159" --solo
```

### Not booked this node — (C) DAVE-lite verify-and-suppress (`use_verify_mask`, fallback)

C is the only false-positive-suppression mechanism and near-zero params, and its text
passes format + novelty (exit 0, top sim 0.384) today. It is deferred only because
K_SYNTH=2 and A/B rank above it in the fetched backlog (STATE.md:36–49) and better match
the H0012 consolidation (finer-grained exemplar steering over a globally shared scalar).
Recommended as the first booking next synthesis or if either A/B child fails to smoke.

### Registry entries needed (Lead adds after booking returns the new ids)

`--solo` bookings do not consult `src/cac/expt/mechanisms.py`, but future Q_t adjoins do.
Add to `_reg()` in `src/cac/expt/mechanisms.py`:

- A's new id: `{"components": {"safe_enhance"}, "switches": {"use_safe_enhance"}, "requires": set()}`
- B's new id: `{"components": {"exemplar_gate"}, "switches": {"use_dyn_exemplar_gate"}, "requires": set()}` — sharing the `exemplar_gate` component with H0010 so two exemplar-token gating mechanisms can never be co-composed.

Switch names follow the STATE backlog contract. No ledger evidence is due for H0012 (it
already has `contradicts` w=1.0 at `memory/hypotheses.jsonl:41`), and A/B are new
mechanisms, not opposites of refuted hypotheses, so no `contradicts` event is booked
(AGENTS §4.8c).

## 6. Quality-gate checklist (AGENTS §4 step 8)

- **(a) Each booking passes format gate — PASS.** I re-ran the format validator on both
  exact texts this session: `cac.expt.hypothesis.validate` returned `errors: []` for A
  and B (C too). Novelty gate re-verified today: `novelty_check.py` exit 0, top sims
  A=0.360 (H0009), B=0.531 (H0010), C=0.384, no structural twins. Texts are used verbatim.
- **(b) ≤ K_SYNTH = 2 new hypotheses — PASS.** Exactly two bookings (A, B), one hypothesis
  each, both `--solo`.
- **(c) Opposite-of-existing books as `contradicts` on the existing id, never a duplicate
  — PASS.** No booking re-encodes a refuted mechanism; H0012's contradicted verdict is
  already in the ledger; B is a new id (novelty gate: 0.531 < 0.82, no structural twin),
  explicitly not a retry of H0009/H0010.
- **(d) Misattributed reasoning remapped or discarded — PASS.** B's premise (image-global
  gate cannot choose per-exemplar channel subspaces) matches H0009's refutation and
  H0011's inertness; A's premise describes the exemplar softmax's normalizing behaviour
  (qual §1) — "same distribution" is loose wording for "a full convex combination over
  the 3 prototypes at every location", but it is a proposed mechanism premise, not a
  misattributed measurement, and the text is kept verbatim per instruction; C's premise
  cites no existing result. No remaps required.

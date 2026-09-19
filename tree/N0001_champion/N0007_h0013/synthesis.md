# N0007_h0013 — Synthesis (H0013 solo on N0001_champion)

Synthesis Agent, AGENTS.md §4 step 8. Sources read directly: `AGENTS.md`, `STATE.md`,
`idea.md`, `model.py`, `config.toml`, `result.json`, `info.json`,
`feedback/{quant,qual,causal}.md`, `memory/hypotheses.jsonl` (H0013 rows, lines 38 and 44),
`tree/N0001_champion/result.json`, plus a fresh `discovery calibration` and a fresh
`discovery parent` (returns N0001_champion, score 0.730). Both gates re-run on the booking
text this session: `cac.expt.hypothesis.validate` → `errors: []`; `novelty_check.py` exit 0
(top sim 0.489 vs H0014, no structural twin). No commits, no node creation, no writes to
`memory/`. Confidence math is the machinery's; nothing here recomputes it.

## 1. Outcome

At seed 20260830 the solo subpixel child **N0007_h0013** completed **32/32 epochs with
`budget_hit = false`** (1708.1 s wall of the 1800 s τ_max, 91.9 s headroom) and finished
with best val MAE **22.454483032226562 at ep32** (= last epoch; `config_sha256`
2300d4dbbde4aef7, `model_sha256` 559d216c252b657e, both recomputed by quant). Parent
**N0001_champion** stands at **21.458858489990234 at ep32**. Best-vs-best delta
**+0.9956245 (child ~1.0 MAE worse)** against a pre-registered support line of
21.4589 − 0.30 = **21.1589**: the child sits **+1.2956245 above the line** (~4.3× the bar),
and **no epoch of the child is within 1.29 of it** (child best 22.4545 is only champion
quality at ~ep21.5 — champion first reaches ≤22.4545 at ep22). The falsifier "final val MAE
is not at least 0.30 lower" evaluates to 22.4545 vs 21.1589. Ledger:
`memory/hypotheses.jsonl:44`, `contradicts` w=1.0 (2026-09-19T16:33:42+0800) → H0013
confidence **0.400**, status `uncertain`, `n_tested: 1`.

## 2. Consolidated findings (deduped across quant/qual/causal)

**Identified (verified measurements):**

1. **Bar and endpoint.** Child 22.4545 @ep32 vs champion 21.4589 @ep32 = +0.9956; miss of
   the 21.1589 line = +1.2956; 32/32, budget not hit; both bests at ep32 (quant §1/§3,
   qual verdict, causal §2).
2. **Curve shape: early lead, then a stable ~+1.0 deficit — a fit/regularization shift,
   not an optimization failure.** Child leads ep1–11 (ep1 −21.87, ep11 −0.12), crosses at
   ep12 (+0.12), then diverges monotonically to a flat tail (ep25–32 mean **+1.0147**,
   sd 0.0355, ep32 +0.9956, no closing trend). Train loss is lower for the child nearly
   everywhere (ep1 23.30 vs 50.31; ep32 2.470 vs 2.706) while val is worse; the val RMSE
   gap exceeds the MAE gap (ep32 **+4.904**: 84.675 vs 79.771), i.e. excess error sits in
   fewer, larger misses (quant §2, qual §3, causal §2).
3. **The implementation is exactly the stated change; the ablation surface is clean.**
   One config key (`use_subpixel_up = true`, config.toml:27) + one component: grouped
   depthwise 3×3 conv `(512,1,3,3)` + `PixelShuffle(2)` (model.py:74–77), **5120 params**
   (3,479,938 → 3,485,058), replacing the *first* bilinear upsample at S/16→S/8; the second
   upsample (S/8→S/4, model.py:90) stays bilinear in both arms. Switch-off is structurally
   the champion path, so the failure is the mechanism's, not the interface's (quant design
   note, qual §1/§4, causal §1). Schedule invariant: LR sequence and train/loss step
   indices bit-identical to champion (quant §1/§5).
4. **RNG-stream confound (explicit, causal — verified on the server, not inferred).**
   `top_up` is constructed *mid-`FineFuser`, before* `ExemplarEncoder`, `Condenser`,
   `DensityDecoder` (and `GCA` after), so it consumes **5120 global RNG draws** with the
   shared seed. At seed 20260830, `fuser.top/lat/fuse/refine` stay bit-identical while
   `exemplar.proj.weight`, `exemplar.tr.layers.0.self_attn.in_proj_weight`,
   `cond.proj_in.weight` and `decoder.block.0.weight` **all differ** between parent and
   child. The same offset reaches the **data stream**: `runner.py:117` builds
   `FSC147DataModule(...)` **without `seed=`**, so `DataLoader(shuffle=True)` draws from the
   global RNG (val loader is `shuffle=False`, so the metric is unaffected) — first-epoch
   permutations differ (parent `[2318, 1003, 2169, …]` vs child `[2699, 1786, 1514, …]`)
   while `torch.initial_seed()` stayed 20260830. The contrast is therefore a **bundle**:
   resampler function + re-init of most trained modules + different training batch order.
   (This also corrects N0006's "batch order is identical" line; it applies to every node
   that constructs an extra module before fit.) (causal §1; quant §2 confound note.)
5. **The replaced operator is a fixed low-pass; the claim's premise is inverted.** Bilinear
   scale-2 `align_corners=False` is a non-negative partition-of-unity 2×2 filter (mass-
   preserving, separable triangular low-pass); the learned block is a signed, unconstrained
   3×3-per-parity map with no sum-to-one constraint, and its claimed "detail" is re-smoothed
   by the surviving bilinear at model.py:90 anyway. Remaining count anchors are global only
   (L1 sum term, GCA bias), so high-frequency density is a pure MSE liability against the
   smooth GT (gauss_sigma=1.5) (qual §1).

**Not identified (do not read as established):**

6. **The named mechanism was never measured.** No map sharpness/localization observable was
   recorded; count-level MAE = `|sum(pred) − count|` cannot distinguish "sharper but worse"
   from "blurrier with correct mass"; no per-image predictions were saved. A kernel audit
   on the two server `best.pth` files is cheap (minutes, CPU/GPU-lite) but descriptive
   unless pre-registered (causal §2–§3; qual "Implications").
7. **Attribution of the +1.0.** The verified bundle (finding 4) means the run does not
   separate (i) the resampler function, (ii) the verified re-init of exemplar/condenser/
   decoder/GCA, (iii) the different batch order. Nothing in the run ranks them (causal §0,
   §4).
8. **Generality.** Not licensed: "learned upsampling is bad", "subpixel at this junction is
   bad per se", or even "the champion's bilinear smoothing prior is load-bearing" — the
   tested object is one placement (`after GroupNorm`, S/16→S/8, random init, no residual)
   under one protocol; both endpoints are non-converged (both improve at ep32), so part of
   the gap may be convergence speed (causal "What this run does/does not license", §2).
9. **Identity-at-init regularity is observational.** Cross-node (n=1/cell, different
   lineages): identity-at-init H0012 **+0.036**; non-identity H0009 **+1.149**, H0010 joint
   **+1.923**, H0011 **+1.35** conditioned, H0013 **+0.996**. The ordering tracks step-0
   function perturbation, not parameter count (5 k params cost ~1 MAE). Not established:
   n=1 per cell, joint confounding for H0010, noise floor unmeasured (qual §5, causal §4).
10. **Noise floor is still unmeasured.** Same-seed `repro_run` bounds determinism, not
    seed spread; ~1.0 is far outside any plausible seed noise, but sub-0.3 deltas are not
    interpretable until measured (qual §3 caveat, causal §5, quant "So what").
11. **Non-findings (checked, benign).** Capacity is not the cause: +5120 params = 0.15% of
    3.5 M trainable and the child trains *better*; wall-clock +3.6 s (+0.21%) is not
    binding (~33 epochs fit the 1800 s budget) (qual §3, quant §4, causal §3).

## 3. Mechanism verdict (ledger-consistent)

The ledger stands as written: `memory/hypotheses.jsonl:44` records `contradicts` w=1.0 for
H0013 from N0007_h0013; **H0013 is confidence 0.400**, status `uncertain` (H0013 confidence
0.5 → 0.4 via Eq.1; machinery-derived, not recomputed here).

Verdict language: **H0013's pre-registered bar failed decisively** — this implementation,
solo on N0001_champion, seed 20260830, 32/32 epochs, ends 1.2956 above the 21.1589 support
line with no epoch within 1.29 of it. What this does **not** license is "subpixel/learned
upsampling is refuted as a mechanism": H0013 is **contradicted, not confirmed-refuted**
(0.400 > 0.25), the run is a verified bundle (finding 4), and the mechanism's own
observable was never measured (finding 6). Any re-encoding/retry of H0013 must book as
`contradicts`-evidence on the existing id **with a NEW falsifier** (Hard Rule 11) — never
a silent retry.

**Identity-at-init question.** `top_up` was **default-random-initialized** (no custom init;
kaiming-uniform a=√5, bias ±1/3 → zero-mean non-identity with ≈1/√3 gain in variance),
unlike N0006_h0012 (zero-init → exact identity at step 0) and the booked N0008_h0014/
N0009_h0015 (identity-at-init, smoke pending). Is that a distinct failure mode? It is
**treatable as a candidate distinct failure mode** — a step-0 function-perturbation tax /
fragile-basin regularity (finding 9) — but **not identified by this node**:
(i) H0013's own curve is early-lead/late-regression, the opposite of the early-deficit/
recovery a pure init-damage story predicts, so its own random init is at most a
second-order contributor (qual §2); (ii) the bundle supplies two other candidate causes
(finding 4); (iii) n=1/cell across lineages, noise floor unmeasured. Establishing it needs
a pre-registered, stream-matched identity-at-init contrast (e.g. bilinear + zero-init
residual at the same site); per Rule 11 that books on H0013 with a new falsifier, and per
this synthesis it should wait for the harness fix (causal §4) and for N0008/N0009's
identity-at-init results.

## 4. Calibration (verbatim: `PYTHONPATH=src python3.13 scripts/discovery.py calibration`, run this session)

```text
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      9       0      0%       0.36
[0.50,0.75)     19      10     53%       0.54
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall         28      10     36%      -
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
H0013    0.400uncertain       1
H0014    0.500uncertain       0
H0015    0.500uncertain       0
```

Interpretation: reliability error holds at **0.126** (under the 0.2 gate) with perfect
15/15 direction, but the `[0.25,0.50)` bin still over-predicts (0.36 predicted vs 0% over
9 tests); H0013's 0.5→0.4 contradiction moved the overall confirm rate 37%→36% without
changing the miscalibration — the H0007–H0013 uncertainty band remains the ledger's
optimistic corner.

## 5. BOOKING RECOMMENDATIONS

**Parent: N0001_champion** — verified this session, `discovery parent` → N0001_champion
(score 0.730 vs N0006_h0012 0.686; N0007_h0013 0.435). One booking, one hypothesis,
`--solo` on the champion, text verbatim (format+novelty re-verified this session).

### Booking 1 of 1 — (C) DAVE-lite verify-and-suppress (`use_verify_mask`)

Chosen because it attacks a **different failure mode** than the refuted perturbation family:
false-positive mass on distractor/textured background, suppressed *at the density map*
before the loss, rather than a re-encoding of any refuted mechanism. It reads the fused
fine feature + an exemplar prototype (it does **not** re-read `GAP(fine)+e_mean`, the
H0012 channel qual/causal flagged as already count-supervised by GCA), is near-zero params,
and is the top un-booked paper-backlog item (STATE.md "paper backlog"). Novelty clear: new
id, top sim 0.489 vs H0014 < 0.82, no structural twin — no `contradicts` event applies to C.

```bash
PYTHONPATH=src python3.13 scripts/discovery.py hypo N0001_champion --new "IF we add a verify-and-suppress stage that pools the fused fine feature at the top-scoring peaks of the preliminary density map, scores each pooled peak by cosine similarity against an exemplar prototype pooled in the same fine-feature space, and multiplies the density map by a learned-threshold verification gate before the loss IN the CountingHead of the champion counter THEN final val MAE is at least 0.30 lower than the champion without the verifier BECAUSE the decoder is trained only to deposit mass at salient regions, so distractor objects and textured background that match nothing in the exemplars still receive spurious mass that inflates count error, while a verification gate whose offset is learned through the density loss suppresses mass exactly at peaks whose appearance disagrees with the exemplar prototype and leaves verified peaks intact, removing false positives without touching the frozen backbone. DISPROVED IF final val MAE is not at least 0.30 lower than the champion N0001 seed of 21.459, that is, final val MAE is above 21.159" --solo
```

Coding-agent note (implementation, not text): make the gate identity-at-init (multiplier ≡ 1
at step 0) and construct the stage after all pre-existing `CountingHead` modules, with any
new params drawn from a **private `torch.Generator`** — that keeps the global RNG stream
(and hence shared-head init and DataLoader shuffle) champion-identical (findings 4, 9).

Registry entry for future Q_t adjoins (Lead, after the new id returns):
`{"components": {"verify_mask"}, "switches": {"use_verify_mask"}, "requires": set()}`.

### Deliberately NOT booked this cycle — identity-init H0013 re-test, constructor placebo

The follow-ups the causal/qual reports rank (constructor placebo for N0007; bilinear+zero-
init-residual subpixel re-test booked on H0013 with a new falsifier) are **not booked now**:
- The stream confound is not yet fixed (Lead is preparing the `runner.py` seed fix) and the
  reference champion may need a re-run under it; a re-test booked now would repeat the same
  bundle it is meant to remove.
- N0008_h0014/N0009_h0015 are identity-at-init and pending smoke/run; their results are
  direct evidence on the identity-at-init failure-mode question at no extra booking cost.
- Noise floor (`repro_run`) is still pending; the new-falsifier text should be authored
  once the champion's seed spread and the fixed-stream reference exist.
- Kernel audit / map-sharpness eval on the existing two checkpoints is a measurement, not a
  hypothesis: no booking needed (causal §3).

If K_SYNTH headroom is wanted later, the re-test must book on **H0013** (rule 11) — e.g.
`--book H0013 --solo` with a falsifier that pre-commits the identity-at-init claim — and its
text must pass `novelty_check.py` before booking.

## 6. Quality-gate checklist (AGENTS §4 step 8)

- **(a) Format gate — PASS.** Booking text run this session through
  `cac.expt.hypothesis.validate` → `errors: []`; markers IF/IN/THEN/BECAUSE/DISPROVED in
  order; falsifier is numeric (21.159 bar). Novelty: `novelty_check.py` exit 0, top sim
  0.489 (H0014) < 0.82, no structural twin. Text used verbatim, unmodified.
- **(b) ≤ K_SYNTH = 2 new hypotheses — PASS.** Exactly one booking (C).
- **(c) Opposite-of-existing books as `contradicts` on the existing id — PASS.** C is a new
  mechanism, not a re-encoding of any refuted id, so no `contradicts` event is due. H0013's
  `contradicts` w=1.0 is already in the ledger (line 44); the identity-init re-test (the
  only H0013-family candidate) is deliberately deferred, avoiding a rule-11 silent-retry
  risk this cycle.
- **(d) Misattributed reasoning remapped/discarded — PASS.** The one-line "identity-at-init
  is a distinct failure mode" regularities are carried only as observational candidates
  (finding 9), not as causes; quant's "early lead cannot be attributed" is preserved as the
  bundle finding; the cross-node table keeps its n=1/lineage caveats; nothing discarded was
  load-bearing for the booking text.
- **Confound avoidance (per this node's findings):** C avoids the `GAP(fine)+e_mean` channel
  (it reads fine features + an exemplar prototype at density peaks and multiplies the
  density map pre-loss; no second count pathway duplicates GCA). The RNG-shifted
  shared-init confound is avoidable **by construction** (private generator + late
  construction + identity-at-init gate), which is the required coding-agent pattern stated
  above; C can run regardless, but for a cleanly *identified* contrast the Lead should land
  the pending harness fix (`seed=` into `FSC147DataModule` at `runner.py:96/117`) and
  re-reference the champion under it. **Dependency: identification depends on the fix (or
  on the private-generator pattern); booking itself does not.**

# Causal feedback — N0011_h0010 (H0010 clean solo, per-exemplar scalar reliability gate)

Role: causal-identification critique + family closure. NOT a re-verdict: ledger
already records `contradicts w=1.0` on H0010 (conf 0.3200,
`memory/hypotheses.jsonl:49`). Numbers verified from
`tree/N0001_champion/N0011_h0010/{result.json:5-6,info.json:7,config.toml:27,model.py:158-176}`,
parent `tree/N0001_champion/{result.json:5-6,config.toml,model.py:143-168}`,
`feedback/{quant.md,qual.md}`, `memory/hypotheses.jsonl:31-49`, and the
pre-registration `tree/N0001_champion/N0009_h0015/feedback/causal.md:155-202`.

## 0. Result in one line

Child best **23.4581 @ep25** (32/32, `budget_hit=false`, 1702.7 s, `status: done`)
vs canonical parent **23.2932 @ep26** (32/32, 1699.4 s) → **+0.1649 worse**.
Live bar under AGENTS §6 anchor rule (0.30 below live parent): **≤22.9932**,
missed by **+0.4649** (≈1.5× bar width, ≈8× paired precision 0.02).
Child worse in **30/32** epochs; train fits BETTER (−0.0852) while val MAE and
RMSE both worse (+0.1649 / +0.9639 best-vs-best) — overfit signature on 257 params.

Provenance: canonical-protocol (augment=false, EMA eval, seeded loaders, cudnn
deterministic, seed 20260830), same-seed paired, full 32-epoch observation, no
budget stop. Fidelity is airtight per qual.md §1: 1 config line + ~5 model lines,
`Linear(256,1)` zero-weight/bias-3.0 (~0.953-uniform init), verbatim from
N0004, OFF restores champion exactly, rule-13 append-last holds. Full tables,
hash and stale-file checks: quant.md §§1–6; fidelity + GCA fan-out read: qual.md
§§1–4. Nothing below re-litigates the verdict.

## 1. Pre-registration check: does outcome (ii) obtain? YES — with one honest qualifier

The N0009 causal pre-registered three joint readings (causal.md:162-185),
with H0015's side fixed at fail (+0.3886, best@23, tail-drift):

- **(ii)** as written: "N0011 fails (misses bar AND lands at/near-or-below N0009,
  i.e. clearly worse than parent by ~0.3+ with a similar early-stall/drift
  signature) × H0015 fails" → retire the family.
- **(iii)** as written: "N0011 misses the bar but is clearly less harmful than
  N0009 (ordering parent ≥ N0011 ≫ N0009, e.g. within ~0.1 of parent while N0009
  is +0.39 behind…)" → gradient-of-harm, optimization-drag account, keep family
  alive only in K×1 form.

Observed N0011: bar missed decisively (+0.4649 beyond the line); level **+0.1649**,
i.e. squarely BETWEEN the (ii) level clause (~0.3+) and the (iii) level example
(~0.1). Strictly, the literal "~0.3+" sub-clause is NOT met (+0.165 < +0.30),
and the train-loss sign differs from N0009's (N0011 fits train BETTER −0.0852;
N0009 fit train WORSE late +0.20 — drag vs discount, qual.md §4 here and N0009
causal §2). So a pedantic literalist could claim neither (ii) nor (iii) fires cleanly.

The decision-relevant answer is still **YES, outcome (ii) obtains**, for the
reason the pre-registration's §4 logic makes explicit: the (ii)/(iii) split was
designed to answer "is there a surviving K×1 form?" — and the K×1 best-case has
now itself decisively contradicted solo. The load-bearing clauses of (ii) all hold:

1. **Bar decisively missed**: needs ≤22.9932, lands 23.4581 (−0.4649 swing needed).
2. **Clearly worse than parent at ~8× paired precision** (+0.1649 vs ~0.02) —
   a real degradation, not noise (quant.md §2).
3. **Same family signature**: persistent full-horizon deficit (30/32 behind, mean
   Δ ep20–32 +0.166, tail widening ep29–32 all ≥+0.20), no crossing, no late
   recovery, best epochs adjacent (25 vs 26) — the uniform-small-harm shape,
   differing from N0009 only in the train-loss sign (discount-memorization vs
   optimization-drag: two failure MODES, one failure SIGN).
4. **Double failure across independent function classes** (K×1 scalar + K×C
   per-channel) **plus** H0009's global variant (refuted ×2) — three granularities,
   three conditioning schemes, one sign.

Meanwhile (iii)'s escape hatch — "keep the family alive but only in K×1 form" —
is self-closing: the K×1 form IS what just failed. The (iii) premise (harm scales
with gate capacity: 0.389 → 0.165, ratio 2.36×) is confirmed as a GRADIENT
observation, but it now functions as a mechanism hint (see §3), not a survival
license: the family's best, minimal, direction-preserving cut still contradicts.
Verdict: **outcome (ii) obtains — YES — with the recorded qualifier that the
level (+0.165) sits between the (ii)/(iii) drafting examples, and the operative
fact is the double-independent-class failure, not the exact depth of the miss.**

## 2. Family ledger and precise evidential closure

Live ledger (all canonical-protocol unless noted):

| Hyp | Claim (family role) | Evidence | Conf |
|---|---|---|---|
| H0009 | image-global channel gate (1 mask, all exemplars) | contradicts ×2 (N0003 solo v1, N0004 joint) | 0.320, refuted-track |
| H0010 | per-exemplar SCALAR gate (K×1, direction-preserving) | contradicts (N0004 joint, confounded) + neutral (N0005, unattributable) + **contradicts w=1.0 clean solo (N0011, this node)** | **0.320** |
| H0011 | capacity control (frozen-random gate, conditioning severed) | contradicts (falsifier fired: +0.202 within 0.30 band → inert capacity) | 0.400 |
| H0015 | per-exemplar per-CHANNEL gate (K×C MLP mask) | contradicts w=1.0 solo (N0009, +0.389) | 0.400 |
| H0014 | query/similarity-side enhancement (NOT a gate) | supports w=1.0 (N0008, −0.32, fragile) | 0.600 |
| H0016 | post-decoder verification mask (NOT a gate) | supports w=1.0 (N0010, −0.41) | 0.600 |

**Dead — every pre-condenser exemplar-rescale cut tried:**
image-global (H0009, refuted ×2) → per-exemplar per-channel (H0015, +0.389) →
per-exemplar scalar (H0010, +0.165 clean solo). The conditioning axis (global →
per-exemplar) and the freedom axis (K×C → K×1, rotation → mass-rescale) are both
exhausted with the same sign; the monotonic harm gradient (coarser/richer =
worse) plus the minimal cut's failure closes the granularity escape (qual.md §2).
Under rule 11, no exemplar-side rescale variant may be re-booked without a NEW
falsifier — and §1 shows no falsifier-shaped gap remains for rescaling keys/values
before the condenser.

**Mechanism controls, not live hypotheses:** H0011 (frozen-random) already fired
its falsifier — it established inert-capacity in its lineage, which now cuts
AGAINST any "but capacity/padding" rescue of the gate family. A frozen-random
SCALAR control off N0011 (the follow-up pre-registered under outcome (i)) is
explicitly deprioritized: it would be a mechanism control on an already-refuted
claim, spending queue to split "selection information" from "scalar padding" for
a switch whose joint treatment already hurts. Likewise the decouple-GCA node
(gate condenser keys, keep `e_mean` ungated) was pre-registered as "never after
(ii)" (N0009 causal §5.4) — the GCA fan-out confound (qual.md §3 here) qualifies
mechanism sentences but cannot rescue the switch-level kill, so decoupling is
forensics, not a live direction.

**What (if anything) remains untested exemplar-side — the LOCA question:**
prototype-side ITERATION (LOCA-lite: adapt the prototype by cross-attending exemplar
tokens to image features over steps, djukicn/loca) is the one exemplar-adjacent
idea never tested here. Judgment: it is **distinct enough to survive — narrowly,
and only if booked as a new address, not a gate.** Reasons: (a) it does not
rescale calibrated matching keys by a learned per-exemplar/channel multiplier —
it REFINES the prototype representation against dense image evidence before
matching, leaving key norms/set-membership un-gated; (b) its failure mode would
differ observably (iterative refinement collapse vs multiplicative sharpness
shift); (c) it must pass novelty + rule 11 with a fresh falsifier that names the
N0011/N0009 signatures it must avoid (ep10-onward persistent deficit,
train-better/val-worse split). Any LOCA-lite proposal that reduces in
implementation to "multiply exemplar tokens by a learned function before the
condenser" dies by association — the booking must show the iterative,
image-grounded refinement step as the load-bearing mechanism, or it is a silent
re-encode and must be rejected at the format gate.

## 3. The overfit signature on 257 params: candidate causal stories, ranked by testability

Fact to explain: +257 params (one Linear(256,1)) cannot memorize FSC147, yet
train fits better (−0.0852) while val degrades (+0.165 MAE, +0.96 RMSE), 30/32
epochs behind, gap widening into the tail. Capacity is arithmetically excluded;
the lesion is positional (multiplicative, at the highest-leverage point: every
condenser query + the GCA mean) and/or dynamical (landscape reshaped from step 1).

Ranked most-testable first (all cheap, no-GPU-first):

1. **Per-image discount memorization (cheap train dof).** The K×1 scalar is a
   per-(image, exemplar) free discount: on train scenes it learns "trust exemplar
   2, discount exemplar 1" as a 1–3-dof per-image fit that lowers train loss
   without learning transferable reliability — val prototypes get arbitrary
   discounts, calibrated keys poisoned. Predicts: gate-value variance across
   exemplars correlates with train-loss improvement per image but NOT with any
   exemplar-quality proxy (box size, occlusion, background fraction); gate
   entropy collapses on train, scatters on val. Test: read gate statistics from
   the N0011 checkpoint (train vs val gate-value distributions + per-image
   gate-spread vs per-image MAE-delta) — pure log/checkpoint forensics, no GPU.
2. **Key-norm → softmax-sharpness shift (matching-path optimization drag).**
   Queries are LayerNormed, keys/values un-normed (model.py:121-125), so scalar
   rescaling directly moves K-way (K=1..3) attention sharpness and hence gradients
   into condenser + exemplar encoder from step 1 (identity lasts zero updates).
   Predicts: condenser attention entropy child-vs-parent diverges early (by
   ep3–10, when the +0.08…+0.24 gap establishes) and stays shifted; gate-mean
   drift coincides with entropy shift. Test: attention-entropy readout
   child-vs-parent on val across epochs (needs stored attentions or a short
   repro — repro_run, no tree mutation) + gate-mean trajectory from checkpoint.
3. **GCA-path contamination (aux-channeled harm).** Gated `e` feeds `e_mean` →
   GCA → density bias (model.py:175,221-224); the gradient into the gate flows
   through both heads. Direct forward effect is small (0.02/(Hf·Wf) scaling,
   ~5% uniform at init), so this is the underdog as PRIMARY carrier — but it
   cannot be excluded by arithmetic alone since only `out["density"]` feeds the
   loss. Predicts: ablating the GCA fan-out (ungated `e_mean`) closes part of
   the +0.165. Test: the decouple-GCA node — deliberately NOT recommended (§2:
   forensics only, never after (ii)); listed here for completeness of the
   ranking, lowest priority because even full attribution to GCA would not
   resurrect the switch (the joint treatment is what the ledger kills).
4. **Attention dilution (mass-spreading) — weakest, listed to kill it.**
   Would predict flatter attention + higher RMSE-through-count-spread; but the
   scalar gate SHARPENS selection (down-weighting = concentration), and RMSE
   concurs with MAE rather than diverging, so dilution's signature is absent.
   Ranked last: untestable-without-new-instrumentation and disfavored by the
   observed concentrated-harm shape.

Net: (1) and (2) are the live accounts and both are checkpoint-forensic
testable without spending the queue; (3) is real-but-small and explicitly
deprioritized; (4) is disfavored. None rescues H0010 — they explain HOW the
minimal gate harms, which is exactly what a retired family needs on record.

## 4. What this licenses for the NEXT bookings

Live directions (all downstream of the exemplar keys — query/similarity-side or
decoder-side, leaving exemplar tokens alone):

- **H0014 confirmation** (query/similarity-side, supported N0008 −0.32, fragile
  margin +0.024): the highest-evidence live direction. Next booking should be a
  same-seed replicate/confirmation with a mechanism-separating readout (dual-norm
  ablations: exemplar-axis vs spatial-axis normalization) rather than a new
  similarity variant.
- **H0016 confirmation + mechanism separation** (decoder-side, supported N0010
  −0.41, margin +0.11): confirm, then run the **uniform-damping diagnostic** —
  a constant (unconditioned) density-damping control that splits "verification
  information" (H0016's cosine-vs-prototype claim) from "any late suppression
  helps." This is the H0011-logic control done RIGHT: on a SUPPORTED hypothesis,
  where a control still has decision value.
- **Genuinely new families** at new addresses (query-side, decoder-side,
  loss-geometry) with fresh falsifiers that name the retired signatures.

**What should NOT be booked (rule 11):** any pre-condenser exemplar modulation —
scalar, per-channel, global, value-only, logit-bias, schedule/init-revised gate,
decoupled-GCA re-encode — without a NEW falsifier that (a) cites N0011 (+0.165)
and N0009 (+0.389) by number, (b) states which of stories §3.1–3.3 it defeats and
how the readout would show it, and (c) books as `contradicts`-evidence on the
existing H0009/H0010/H0015 ids rather than a new id. Silent retries are
fabrication-adjacent; the format gate must reject them.

## does / does not license

- **Does license:** outcome (ii) obtains — YES (bar missed by +0.4649, +0.1649
  worse at ~8× paired precision, 30/32 behind with no recovery, double
  independent-class failure K×1 + K×C plus H0009's global variant) — with the
  recorded qualifier that +0.165 sits between the (ii)/(iii) drafting examples
  and the operative fact is the solo K×1 best-case contradicting, which
  self-closes (iii)'s keep-alive hatch.
- **Does license:** retiring the pre-condenser exemplar-gating family under rule
  11 — H0009 (0.320 refuted-track), H0010 (0.320 after clean solo), H0015 (0.400)
  are all contradicts-side; H0011's inert-capacity finding cuts against rescue;
  no rescale-before-condenser variant remains live.
- **Does license:** directing the queue to query/similarity-side (H0014
  confirmation) and decoder-side (H0016 confirmation + uniform-damping
  mechanism separation) plus genuinely new addresses — with LOCA-lite
  prototype iteration admitted ONLY as a new-address booking with a fresh
  falsifier, never as a gate re-encode.
- **Does not license:** any verdict or confidence move on H0014/H0016 from this
  node (different mechanisms; the sibling gains only corroborate that the
  harness registers improvements, sharpening the gate family's negative signal).
- **Does not license:** the narrow claim "no exemplar-touching intervention could
  ever help under any schedule/init" — optimizer-in-budget (32ep cosine) and the
  GCA fan-out remain unseparated rivals; what is retired is the family of
  pre-condenser exemplar-side multiplicative modulations under this
  head/optimizer/budget, not every conceivable exemplar-adjacent idea.

# Causal feedback — N0009_h0015 (H0015 solo, dynamic per-exemplar channel gate)

Role: causal-identification critique + what the result licenses. NOT a re-verdict:
ledger already records `contradicts w=1.0` on H0015 (conf 0.400). Numbers verified
from `tree/N0001_champion/N0009_h0015/{result.json:5-6,info.json:7,config.toml:27,
model.py:120-139,199-203}`, parent `tree/N0001_champion/{result.json:5-6,
config.toml,model.py}`, `feedback/{quant.md,qual.md}`, `memory/hypotheses.jsonl:43,47`,
and queued `N0011_h0010/{idea.md,model.py:158-176}`.

## 0. Result in one line

Child best **23.6818 @ep23** (32/32, `budget_hit=false`, 1703.5 s, `status: done`)
vs canonical parent **23.2932 @ep26** (32/32, 1699.4 s) → **+0.3886 worse**.
Live bar under AGENTS §6 anchor rule (0.30 below live parent): **≤22.9932**,
missed by **+0.6886** (≈2.3× bar width, ≈19× paired precision 0.02).
Last-epoch gap +0.5316 > best-epoch gap: the miss widens toward the tail.

Provenance: both numbers are canonical-protocol (augment=false, EMA eval, seeded
loaders, cudnn deterministic, seed 20260830), same-seed paired. Per instruction,
local `tree/N0001_champion/result.json` (23.2932) and its tb are now the
CANONICAL run — used here; any stale v1 tb artifact is ignored. Child local TB
is byte-identical to the server copy (quant.md:8-16). Full epoch table, RMSE,
train-loss, and hash checks: quant.md:18-176; fidelity read: qual.md:5-50.

## 1. Identification: single-switch test with a coupled fan-out

**Yes — single switch, verified by direct diff:**

- Config parent→child is exactly one line: `use_dyn_exemplar_gate = true`
  (child config.toml:27; parent file ends at `xscale_size`, config.toml:23-26).
  Server hparams snapshots differ ONLY in this key (quant.md:39-42). All else
  identical: seed 20260830, deterministic=true, augment=false, AdamW lr 1e-3
  wd 0.08, cosine eta_min 1e-6, MSE+0.4·L1, batch 16, 32ep.
- Model parent→child is one new class + three head lines: `DynExemplarGate`
  (model.py:120-139) + flag read (model.py:189), conditional construction
  (model.py:190-191), gated multiply (model.py:199-200). Parent model.py:143-168
  has no gate class/flag/hook. OFF path restores the champion exactly
  (qual.md:36-40).
- Rule 13 (append-last) holds: gate constructed AFTER fuser/exemplar/cond/decoder
  (model.py:182-191 order), dropout 0, seed-pinned data stream — shared-module
  init draws identical to the parent's. Model hash moves only where expected
  (parent eaf2a8b9 unchanged; child dd4cbfda), checkpoint +133,676 B ≈ 33,088
  fp32 params of the 256→64→256 MLP (quant.md:147-149). Wall +4.1 s (+0.24%):
  compute-invisible.
- Solo: info.json:10-12 lists `tested_hypotheses: ["H0015"]` only. No Q_t adjoin.

**Treatment contrast, precisely:** for each exemplar token e_k ∈ R^256
(post-ExemplarEncoder incl. XScale), mask g(e_k) = 2·sigmoid(MLP_256→64→256(e_k))
∈ (0,2)^256, token-wise/K-free (nn.Linear on last dim), last layer zero-init so
g=1.0 at step 0. Treated forward: e ← e ⊙ g(e) after the encoder (model.py:198-200),
before the condenser (model.py:201) whose call is `attn(norm1(tok), e, e)`
(model.py:154) — queries LayerNormed, keys/values UN-normed, so scaling is not
cancelled. Control: identical tokens ungated. Because `e_mean = e.mean(dim=1)`
(model.py:203) is computed from ALREADY-GATED e and feeds GCA
(`Counter.forward`, model.py:250-252: n_aux, bias from e_mean, bias added to
density), the switch fans out to **(a) matching keys/values AND (b) gated
prototype mean into the GCA aux path**. Idea.md:7 scopes the mechanism to
"before the token enters the condenser" and never mentions GCA — (b) is an
unregistered side effect (qual.md:16).

**How much does (b) threaten attribution?** At the *switch* level, not at all:
"this switch, trained this way" is cleanly identified — the kill of the
operational claim stands. At the *mechanism* level, it confounds matching-path
vs aux-path harm: the design cannot separate them. Magnitude prior: GCA's
density bias is scaled 0.02/(Hf·Wf) (model.py:218), so its direct forward effect
is small — matching-path harm is the likelier carrier — but the gradient path
into the gate flows through BOTH heads from step 1, and only `out["density"]`
feeds the loss, so GCA harm cannot be excluded by arithmetic alone. What this
blocks: the narrow claim "matching-path subspace selection per se hurts."
What it does not block: the joint claim "this gated-e treatment hurts." A
decoupling node (gate keys, keep e_mean ungated) would split them; this node
cannot. Carry the qualifier on every matching-path sentence below.

## 2. The failure: clean kill of the operational claim, not of every rival

The +0.3886 is a full-horizon (32/32, no budget stop), same-seed, same-protocol
deficit at ~19× paired precision, wrong-signed vs a 0.30 bar, with no rescuing
epoch subset (behind 31/32 epochs; only ep1 leads, −7.32; worst stall ep4 +4.44;
closest approach ep9 +0.061; mid-run band +0.06…+0.32 never crossing;
quant.md:44-104). Tail shapes diverge qualitatively: child best@23 then 9 dead
epochs monotonic +0.186 to ep32 vs parent best@26 flat +0.043 bounce — gap grows
+0.35 (ep24) → +0.53 (ep32). RMSE concurs at smaller amplitude (child best
90.8044 @ep19 vs parent 90.6424 @ep24). Train fit excludes the easy overfit
story: child train starts −52.99 lower (ep1) yet flips by ep9 and is almost
uniformly HIGHER from ep11 (ep32 +0.20) — extra capacity, worse late fit.

Three testable claims, separated because they demand different follow-ups:

- **(a) Subspace-selection-is-useless:** per-prototype channel reweighting cannot
  help counting; condenser+decoder gain nothing from per-exemplar subspaces.
  Consistent with the miss, but val-MAE alone does not isolate it (GCA confound
  §1 + optimizer rival below).
- **(b) Unlearnable-in-budget / optimization-drag:** a better K×C optimum may
  exist, but identity-init 2·sigmoid(0)=1.0 lasts zero updates (ep1 already
  differs on train −52.99 and val −7.32; quant.md:158-162); from step 1 key
  norms → K-way (K=1..3) softmax sharpness → gradients into condenser/exemplar
  encoder all shift, with 256× the multiplicative dof of a scalar gate at the
  highest-leverage point. Early stall (ep4) + persistent deficit + worse late
  train fit is exactly this signature. Falsifiable without retraining the gate:
  gate statistics from the N0009 checkpoint near 1.0 → unused-but-dragging (b);
  diverged/saturated toward 0/2 with high cross-K variance + val harm → actively
  harmful learned subspaces (a)/(c).
- **(c) Actively-harmful mask:** the learned g(e_k) rotates matching direction
  (not just mass) and the tiny-K softmax amplifies small norm changes; GCA mean
  perturbed jointly. Same checkpoint read as (b) plus condenser attention-entropy
  child-vs-parent on val (collapse/flattening coincident with gate movement →
  b→a trace) would split (b) from (c).

Bottom line: **clean mechanistic kill of "this per-exemplar channel gate, from
identity init, under the canonical 32ep cosine/AdamW protocol, lowers val MAE by
≥0.30"** — identification is airtight (solo, paired, sustained). **Not** a kill
of "no channel-gating variant could ever help under any schedule/init/norm" —
(b) remains an unseparated rival, and (c) vs (b) is undecided pending the
checkpoint reads. Report both; design the next test to split them (§5).

## 3. Cross-evidence: H0009, H0011, and the never-cleanly-tested H0010

- **H0009 (image-global channel gate): refuted twice.** N0003 solo (−1.15 vs v1
  parent) and N0004 joint (−1.92); ledger `contradicts` ×2 (hypotheses.jsonl:32,36).
  Different conditioning (pooled-fine global vector → one mask for ALL exemplars)
  from H0015's per-exemplar conditioning, same verdict direction. H0015 fixes
  exactly H0009's conditioning gap (global→per-exemplar) and STILL fails — the
  failure cannot be blamed on global-vs-local conditioning. Remaining axis is
  per-CHANNEL freedom itself (qual.md:34).
- **H0011 (capacity control): refuted as a hypothesis** — frozen-random gate
  22.810 vs conditioned 22.608 (+0.202, within the 0.30 falsifier band), i.e. its
  DISPROVED-IF fired → ledger `contradicts` (hypotheses.jsonl:39). Careful
  reading: this refutes "conditioning carries the information," establishing
  inert-capacity in that lineage — it cuts against conditioning, not against
  gating per se, and is v1-protocol evidence. It weakens "gate information
  matters" but transfers no mechanics to N0009's learned per-exemplar MLP.
- **H0010 (scalar per-exemplar gate, K×1): never cleanly tested.** N0004 tested
  it jointly with H0009 (confounded `contradicts`, hypotheses.jsonl:35); N0005
  booked `neutral` (severed conditioning in the same node, no attribution,
  hypotheses.jsonl:40). Ledger conf 0.400 reflects penalty-without-verdict.
  N0011 (`use_exemplar_gate`, Linear(256,1) bias 3.0 ≈0.95 at init,
  model.py:158-176: scalar eg per exemplar, `e·eg`, direction-preserving) is the
  queued designated decider (idea.md).

**Evidential relationship H0015→H0010, precisely:** H0015's failure licenses **no
ledger update to H0010** (separate id, append-only, different falsifiers;
synthesis must not cite N0009 as evidence for/against H0010). Informally the two
share a family (pre-condenser exemplar-side gating) but denote **different
function classes**: K×1 rescales attention MASS across prototypes preserving each
token's direction (which prototype to trust); K×C rotates each prototype's
matching DIRECTION (what each prototype means) — 256× the dof, multiplicative at
every query plus the GCA mean. K×1 is approximately a restricted (uniform-mask)
point of K×C space, but *learnability runs the other way*: richer class = harder
to optimize, easier to overfit. A K×C failure is therefore compatible with both
"selection is useless" and "selection helps but K×C cannot learn it in 32ep."
It raises the family prior ("exemplar-side perturbation of calibrated keys +
GCA mean is risky") while leaving the class-specific question
("does direction-preserving selection help?") strictly to N0011.

## 4. Pre-registered joint interpretation with N0011 (written BEFORE N0011 lands)

N0011 live bar (anchor rule, same parent): **support iff N0011 best ≤22.9932**
(≥0.30 below 23.2932). Same-seed paired precision ~0.02; cross-seed level noise
~±1 MAE per AGENTS §6 — readings below are paired contrasts, not level claims.
H0015 side is fixed: fail (+0.3886, best@23, tail-drift signature).

- **(i) N0011 supports (≤22.9932) × H0015 fails:** isolates EXEMPLAR SELECTION
  from CHANNEL-SUBSPACE SELECTION. Licensed reading: direction-preserving K×1
  down-weighting of untrustworthy prototypes helps; K×C freedom hurts or is
  unlearnable in-budget (subspace rotation destroys matching / overfits).
  Mandated follow-up: H0011-style frozen-random SCALAR control + gate-value vs
  exemplar-quality analysis (do gated-down exemplars look worse?).
- **(ii) N0011 fails (misses bar AND lands at/near-or-below N0009, i.e. clearly
  worse than parent by ~0.3+ with a similar early-stall/drift signature) ×
  H0015 fails:** double failure across independent classes (K×1 and K×C) plus
  H0009's global variant → license the working conclusion **"any pre-condenser
  exemplar-side perturbation hurts under this head/optimizer/budget — leave
  exemplar tokens alone."** Retire the family (rule 11: no silent retries);
  intervene query/similarity/decoder-side only (extend supported H0014
  similarity-enhance direction, hypotheses.jsonl:46).
- **(iii) N0011 misses the bar but is clearly less harmful than N0009
  (ordering parent ≥ N0011 ≫ N0009, e.g. N0011 within ~0.1 of parent while N0009
  is +0.39 behind, or scalar tail plateaus where channel tail drifts):**
  gradient-of-harm with gate capacity → favors the OPTIMIZATION-DRAG account
  over "selection is useless." Follow-up: capacity-matched random-scalar control
  to split mechanism from padding; keep the family alive but only in K×1 form.
- **Signature qualifier (applies to any level outcome):** N0011 best@32 still
  climbing while N0009 peaked@23 → (b) over (a) regardless of level; both peaking
  mid-run with degraded tails + worse late train fit → (a)/(c)-side active harm.
  Compare train-loss sign flips, not just val argmins.

## 5. Next-test suggestions (lab-rule compliant)

1. Cheap first (no GPU, no ledger): gate-statistic + attention-entropy readouts
   from the N0009 checkpoint (§2 b/c split) to pick between paths 2–4 before
   spending the queue.
2. If joint outcome (i): scalar capacity-control node — same K×1 gate fed a
   frozen random vector (H0011 logic with a NEW falsifier on the scalar gate),
   one switch from N0011. Splits "selection information" from "scalar padding."
3. If joint outcome (ii): stop exemplar gating; next nodes test query-side or
   decoder-side changes only (e.g. extend H0014). Do not re-encode gating under
   a new id (rule 11); any K×C retry needs a new falsifier addressing the ep4
   stall + late drift (e.g. decoupled-GCA or revised schedule — not a silent
   re-run).
4. Only if the family stays alive [outcome (i) or (iii)]: decouple-GCA node from
   N0009 (gate condenser keys, keep e_mean ungated) — one switch, same parent,
   pre-registered bar — to resolve the §1 confound. Never after (ii).

## does / does not license

- **Does license:** the solo treatment — per-exemplar K×C mask on matching keys
  plus gated e_mean into GCA, from identity init, same-seed paired — degrades
  canonical val MAE by +0.39 (best@23, full 32ep, tail widening to +0.53) and
  decisively misses the ≥0.30-gain bar (needs −0.69 from where it landed).
- **Does license:** retiring the K×C channel-subspace gating variant (rule 11:
  any retry needs a new falsifier, e.g. decoupled-GCA form, never a silent re-run).
- **Does license:** a heightened family prior that pre-condenser exemplar-side
  perturbations hurt in this head/budget — to be confirmed or broken by N0011,
  not assumed; N0011's pre-registered joint readings in §4 govern.
- **Does not license:** any verdict, ledger entry, or confidence move on H0010
  (scalar K×1 gate) — different function class, never cleanly tested; N0011 is
  the designated decider.
- **Does not license:** the narrow claim "channel-subspace selection could never
  help," nor any cross-seed level claim — optimizer dynamics in the fixed 32ep
  cosine budget (key-scale → attention-sharpness → gradient path) plus the GCA
  fan-out remain unseparated rivals, and level noise (~±1) dwarfs the 0.30 bar.

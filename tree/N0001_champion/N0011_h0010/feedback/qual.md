# Qualitative feedback — N0011_h0010 (H0010 SOLO, per-exemplar scalar reliability gate)

Result: child best val MAE **23.4581 @ep25** (`tree/N0001_champion/N0011_h0010/result.json:5-6`)
vs canonical parent **23.2932 @ep26** (`tree/N0001_champion/result.json:5-6`),
delta **+0.1649 worse**; live bar (parent − 0.30) = **22.9932**, missed by +0.4649.
Ledger `contradicts w=1.0`, H0010 conf 0.3200 (`memory/hypotheses.jsonl:49`,
`tree/N0001_champion/N0011_h0010/feedback/quant.md:19-25`). This note is code
fidelity + behavior reading only; numbers from `feedback/quant.md`.

## 1. Fidelity: gate matches the pre-registration and the N0004 reference; OFF restores champion

Pre-registered mechanism (`tree/N0001_champion/N0011_h0010/idea.md:7`, identical to
ledger H0010 text `memory/hypotheses.jsonl:33`): "compute a per-exemplar reliability
gate as sigmoid of a linear read-out of each exemplar token and rescale that exemplar
embedding before it enters the condenser cross-attention." Falsifier: ≥0.30 below
champion with the gate removed.

Implementation (`tree/N0001_champion/N0011_h0010/model.py:158-162`):

- Flag read `self.use_exemplar_gate = _get(cfg, "use_exemplar_gate", False)` (`model.py:158`).
- Conditional construction (`model.py:159-162`): `nn.Linear(embed_dim=256, 1)`,
  `nn.init.zeros_(weight)`, `nn.init.constant_(bias, 3.0)` → sigmoid(3.0) = 0.9526,
  near-uniform ~0.953 identity at init.
- Hook (`model.py:170-172`): `eg = torch.sigmoid(self.exemplar_gate(e)).squeeze(-1);
  e = e * eg.unsqueeze(-1)` — per-exemplar scalar (`[B,K,1]` broadcast over 256 dims),
  placed after `self.exemplar(...)` (`model.py:169`) and before `self.cond(fmap, e)`
  (`model.py:173`). Direction-preserving by construction (scalar rescale, no channel rotation).

Reference match — the gate was lifted verbatim from N0004
(`tree/N0001_champion/N0004_h0010/model.py:162-165` construction,
`model.py:173-175` hook: identical `zeros_` + `bias 3.0` + `sigmoid(...)·e` lines).
N0004 additionally carries a channel gate (`model.py:158-160`, `model.py:176-178`)
that N0011 deliberately omits — that is the point of this CLEAN solo: N0004 tested
H0010 jointly with H0009 (confounded `contradicts`, `memory/hypotheses.jsonl:35`),
N0005 booked `neutral` on severed conditioning (`memory/hypotheses.jsonl:40`), so
H0010 at conf 0.400 had never been solo-tested until now.

OFF restores champion exactly:

- Parent `tree/N0001_champion/model.py:143-168` (CountingHead) has no gate class, no
  flag, no hook; the ungated path `e = exemplar(...); cond = cond(fmap, e)` is
  line-identical to the child's flag-off path.
- Parent `tree/N0001_champion/config.toml:23-26` ends at `xscale_size` with NO
  `use_exemplar_gate` key, so `_get` defaults to False. Child adds exactly one config
  line (`tree/N0001_champion/N0011_h0010/config.toml:27` `use_exemplar_gate = true`;
  server hparams snapshot confirms the flag, `run/latest/tb/t/hparams.yaml`).
- Rule-13 append-last holds: gate constructed AFTER fuser/exemplar/cond/decoder
  (`model.py:151-162` order), dropout 0, seed-pinned data stream — shared-module init
  draws identical to the parent's. Child guards both construction (`model.py:159`)
  and forward (`model.py:170`).

Diff surface, total: 1 config line + ~5 model lines (flag read + 3-line construction +
2-line hook). +257 params (256×1+1), wall +3.3 s (1702.7 vs 1699.4,
`feedback/quant.md:76-78`) — compute-invisible. The +0.165 is representational /
optimizational, not budgetary.

## 2. Family gradient: global > per-channel > scalar — finer cuts harm less, none help

Scoreboard under the canonical protocol (same-seed paired, augment=false, EMA,
seed 20260830):

| Variant | Granularity | Δ vs 23.2932 |
|---|---|---|
| H0009 image-global channel gate (N0003 solo v1 −1.15; N0004 joint v1 −1.92; refuted ×2, `memory/hypotheses.jsonl:32,36`, cf. `tree/N0001_champion/N0009_h0015/feedback/causal.md:118-120`) | 1 mask for ALL exemplars | ~−1…−2 (v1 scale; direction refuted) |
| H0015 per-exemplar per-channel MLP mask, K×256 dof (`tree/N0001_champion/N0009_h0015/result.json:5-6` 23.6818 @ep23) | per-exemplar × per-channel | **+0.3886** (`feedback/quant.md:58-72` here) |
| H0010 per-exemplar scalar, K×1 dof (this node, 23.4581 @ep25) | per-exemplar scalar | **+0.1649** |

Scalar harms **2.36× less** (0.3886/0.1649 = 2.357; `feedback/quant.md:63-64`).
Siblings that do NOT gate the exemplar stream help under the same harness (N0008
22.9697 Δ−0.32, N0010 22.8831 Δ−0.41), so the harness registers gains — the gate
family is specifically negative (`feedback/quant.md:70-72`).

Reading: a monotonic gradient of harm with gate capacity / blast radius. The coarser
and richer the perturbation of the exemplar stream, the worse the damage; the minimal
direction-preserving K×1 cut is the family's *best* result and still a decisive
contradict. That pattern favors a LOCATION account over a granularity account:
post-ExemplarEncoder+XScale tokens are calibrated matching keys into the condenser's
tiny-K (K=1..3) cross-attention (`model.py:121-125`: `attn(norm1(tok), e, e)`,
queries LayerNormed, keys/values un-normed, so scaling is not cancelled) — any
multiplicative pre-condenser perturbation shifts key norms → softmax sharpness →
gradients from step 1. Finer gates disturb the calibrated keys less, hence harm
less, but no granularity helps because the intervention point itself is wrong.
H0015 already fixed H0009's conditioning gap (global→per-exemplar) and still failed,
isolating channel freedom as suspect (`tree/N0001_champion/N0009_h0015/feedback/qual.md:34`);
this node now fixes channel freedom (K×C→K×1, rotation→mass-rescale) and still fails,
closing the last granularity escape. What remains is not a finer mask but a different
address (query/similarity/decoder-side).

## 3. GCA side effect: the N0009 dual fan-out applies here too; low threat to the verdict

Yes — same coupling, line-for-line. `e_mean = e.mean(dim=1)` (`model.py:175`) is
computed from ALREADY-GATED `e`, and `Counter.forward` feeds it to GCA
(`model.py:187-191` GCA class; `model.py:221-224`: `n_aux, bias` from `e_mean`,
bias added to density). `idea.md:7` scopes the mechanism to "before the token enters
the condenser" and never mentions GCA — identical to the unregistered fan-out the
N0009 causal flagged (`tree/N0001_champion/N0009_h0015/feedback/causal.md:47-59`:
switch fans out to "(a) matching keys/values AND (b) gated prototype mean into the
GCA aux path"; `feedback/qual.md:16` there concurs).

Threat assessment (same split as `causal.md:61-72`):

- At the SWITCH level, no threat: "this switch, trained this way, solo, same-seed"
  degrades val by +0.165 (~8× paired precision 0.02) — the kill of the operational
  claim stands regardless of path.
- At the MECHANISM level, matching-path vs aux-path harm is unseparated by this
  design, as in N0009. Magnitude prior cuts mildly toward matching-path as carrier:
  GCA's density bias is scaled `0.02/(Hf·Wf)` (`model.py:190`), so its direct forward
  effect is small; and at init the gate is 0.953-uniform, so GCA sees a ~5% uniform
  downscale of the prototype mean — near-identity, no init shock (ep1 transient
  +36.11 washes out by ep3 +0.18; `feedback/quant.md:48-54,86-88`). But the gradient
  path into the gate flows through BOTH heads from step 1 and only `out["density"]`
  feeds the loss, so GCA-channeled harm cannot be excluded by arithmetic alone.
- What this blocks: the narrow claim "condenser matching-path selection per se
  hurts." What it does not block: the joint claim "this gated-e treatment hurts,"
  which is what the ledger records. A decoupling node (gate condenser keys, keep
  `e_mean` ungated) would split them — exactly the follow-up the N0009 causal
  pre-registered and then restricted to outcome (i)/(iii) only
  (`causal.md:187-202`); under the observed outcome (ii) it is not warranted (§5).

Net: the confound is real but small-forward-effect at init and learned-joint later;
it qualifies mechanism sentences without rescuing the hypothesis.

## 4. Learning behavior: better-train/worse-val with 257 params = inductive/optimization harm, not capacity

Curve facts (`feedback/quant.md:28-54,89-90`): child worse in **30/32** epochs
(better only ep8 −0.047, ep9 −0.139); mean Δ ep8–32 +0.131, ep20–32 +0.166; gap
persistent (+0.08…+0.24), widening into the tail (ep29–32 all ≥+0.20); best epochs
adjacent (25 vs 26); no crossing, no late recovery. Train: child fits BETTER
(2.7493 vs 2.8345, −0.0852 at ep32) while val MAE and RMSE both worse
(+0.1649 / +0.9639 best-vs-best) — textbook overfit signature, and RMSE concurs so
there is no MAE/RMSE conflict.

With 257 parameters the capacity story is arithmetically unavailable: 257 scalars
cannot memorize FSC147. The harm must therefore be inductive/optimization — the
gate's multiplicative position at the highest-leverage point (every condenser query
+ the GCA mean) reshapes the loss landscape from step 1 (identity lasts zero
updates: ep1 already differs), letting the head fit train prototypes slightly better
(per-image scalar discounts are cheap train degrees of freedom) while degrading the
calibrated keys the val condenser depends on. Contrast N0009, whose 33k-param gate
showed worse LATE train fit (+0.20) = optimization friction
(`tree/N0001_champion/N0009_h0015/feedback/quant.md:100-104`); here the scalar gate
is cheap enough to fit train yet still poisons val — the family spans both failure
modes (drag at K×C, mis-generalizing discount at K×1) with the same sign. Either
way the lesion is where the gate sits, not how big it is.

## 5. What remains open exemplar-side: nothing pre-condenser; the address changes

Per the N0009 causal's pre-registered joint readings
(`tree/N0001_champion/N0009_h0015/feedback/causal.md:156-185`), this outcome —
N0011 misses the bar AND lands clearly worse than parent (+0.165, 30/32 behind,
uniform small harm, no late recovery) with H0015 also failed — is outcome **(ii)**:
double failure across independent classes (K×1 and K×C) plus H0009's global variant.
The ledger note on this node already books it as such (`memory/hypotheses.jsonl:49`:
"license retiring the pre-condenser exemplar-gating family; direction is
query/similarity (H0014) and decoder-side (H0016)").

Strictly, every pre-condenser exemplar-rescale cut tried is now killed: image-global
(H0009, refuted ×2) → per-exemplar per-channel (H0015, +0.389) → per-exemplar scalar
(H0010, +0.165). Untested variants that a future proposal might name — decoupled-GCA
(gate keys, keep `e_mean` ungated), frozen-random scalar capacity control (H0011
logic with a new scalar falsifier), value-only gating, attention-logit bias instead
of key rescaling, schedule/init revisions — are either (a) explicitly deprioritized
by the (ii) reading (decouple-GCA "never after (ii)," `causal.md:202`), (b) mechanism
controls on an already-refuted claim rather than live hypotheses, or (c) new
mechanisms at a NEW address that must pass novelty + rule 11 with a fresh falsifier
addressing the ep10-onward persistent deficit and the train-better/val-worse split
— never a silent re-encode of gating. The live exemplar-adjacent directions are
downstream of the keys: H0014-style query/similarity enhancement (supported, N0008
−0.32) and H0016-style decoder-side verification (supported, N0010 −0.41), which
leave exemplar tokens alone.

- Fidelity holds at the switch level (Linear(256,1), zero-weight/bias-3.0 ~0.953-uniform init, per-exemplar scalar hook exactly as pre-registered in `idea.md:7` and verbatim from `N0004_h0010/model.py:162-165,173-175`; OFF restores the champion; diff is 1 config line + ~5 model lines), with the gated-`e_mean`-into-GCA fan-out (`model.py:175,221-224`) as the same unregistered side effect the N0009 causal flagged.
- Behavior shows uniform small harm (worse 30/32, +0.08…+0.24 from ep10, tail widening) with better-train/worse-val (−0.0852) on 257 params — inductive/optimization harm at the key-scaling address, not capacity; the K×1→K×C harm gradient (0.165→0.389) closes the granularity escape.
- Joint outcome (ii) with H0015 licenses retiring pre-condenser exemplar gating per rule 11; no exemplar-side rescale variant remains unkilled — next work goes query/similarity (H0014) and decoder-side (H0016), not finer masks.

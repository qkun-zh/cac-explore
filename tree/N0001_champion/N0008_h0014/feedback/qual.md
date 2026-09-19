# qual.md — N0008_h0014 (H0014 SOLO, SAFECount-style enhance) qualitative / mechanism-fidelity read

Result under review: child best val MAE **22.9697 @ep32** (32/32, budget not hit),
vs canonical same-seed parent **23.2932 @ep26** → delta −0.3235, clears the
semantic bar (0.30 below live parent = 22.9932) by **+0.0235**
(`memory/hypotheses.jsonl:46`; canonical baseline `journal/events.jsonl:46`;
replicate determinism ~0.02 `journal/events.jsonl:47`; seed spread ±1
`journal/events.jsonl:48`). First ledger `supports` in the lab — by a knife edge.
This file is the mechanism-fidelity + learning-behavior reading, not the verdict.

## 1. Fidelity to SAFECount (arXiv:2201.08959, §3.2, Eqs.1–7)

The `SafeEnhance` block (`tree/N0001_champion/N0008_h0014/model.py:136-179`)
reproduces the paper's SCM→FEM skeleton faithfully at the structural level:

- **Shared projection + shared LayerNorm (paper Step-1 / Eq.1).** Paper projects
  query and support with a 1×1 conv followed by a *shared* layer norm, then
  compares. Implementation: `self.q = Linear(d_fine→d_s)` for the fine-map tokens
  and `self.kv = Linear(embed_dim→d_s)` for exemplar tokens through one shared
  `self.norm = LayerNorm(d_s)` (`model.py:165-167`), compared as
  `einsum("bnd,bkd->bkn", q, kv)` (`model.py:173-175`). The shared-norm detail —
  the paper's explicit device for putting both sides in the same distribution —
  is preserved.
- **Dual normalization (paper Eqs.2–4).** `R_EN = a/sum_k a` (softmax over the
  exemplar axis, Eq.2) and `R_SN = a/max_n a` (spatial max-norm per exemplar,
  Eq.3), `R = R_SN * R_EN` (Eq.4) at `model.py:176-178`. The exp-shift
  `s − max(s)` is a pure numerical stabilizer: it cancels in both divisions, so
  the computed R is exactly the paper's product. **Yes — the spatial
  normalization is per-exemplar-map over spatial positions** (`amax(dim=2)` over
  HW for each `(b,k)`), matching the paper's `max_{dim=(2,3)}` over the stacked
  exemplar maps; ENorm is over the exemplar axis (`sum(dim=1)`), matching the
  paper's `softmax_{dim=0}`. The normalization geometry is the most faithful
  part of the port.
- **Shared key/value, residual add, zero-init out (paper Eqs.5–7).** Keys and
  values are literally the same projection (`kv` feeds both the score at
  `model.py:175` and the aggregation at `model.py:179`), mirroring the paper's
  shared-projection design; fusion is residual `fmap + out(mix)`
  (`model.py:179`); `out` is zero-initialized (`model.py:169-170`).

Adaptations the coding agent made, and whether they change the meaning:

- **Token dot-product instead of conv-kernel comparison (Eq.1) + einsum instead
  of flipped-kernel conv (Eqs.5–6): FORCED, approximately meaning-preserving for
  the score, meaning-changing for the aggregation.** Both sides here are already
  tokens (flattened fine-map positions; transformer-pooled single-vector
  exemplars), so a dot product is the only available comparison — fine. But the
  paper's FEM convolves R with the *flipped, spatially structured* 3×3 support
  feature (Eq.5, `flip` preserving support spatial structure) and the paper's
  ablations show flipping alone is worth ≥1 MAE. Here the mixed content is a
  single 64-d vector per exemplar with no spatial extent, so every query
  location receives the same K-vector mixture with position-varying weights.
  The paper's "clear boundaries between densely packed objects" mechanism, which
  operates at 3×3-support resolution, has no counterpart here — match resolution
  is per-query-location only.
- **Scale 1/√(d_s) instead of 1/√(H_S·W_S·C): CHANGES the temperature.**
  `model.py:175` scales by `1/sqrt(64) = 1/8`; the paper's scale is
  `1/sqrt(3·3·256) ≈ 1/48`. Combined with LayerNormed (unit-variance) inputs,
  scores here live in a ~6× hotter regime than the paper's, i.e. a much sharper
  ENorm softmax. The dual-norm *form* is exact; its *operating point* is not.
- **Dropped final `layer_norm(f_Q + h(f_R))` (Eq.7) → residual only, `h` reduced
  to one Linear: a real deviation, benign by construction.** No output norm
  means the enhanced tokens can drift in scale — but the condenser trains
  jointly from scratch around it, and zero-init makes the drift grow from zero.
- **Asymmetric, cross-level inputs: the largest unflagged semantic change.**
  The paper compares same-level ResNet features (query map vs ROI-pooled support
  map). Here the query is the fused fine map (128-d, from h2+h3) while the keys
  are transformer+XScale-pooled exemplar embeddings (256-d, from h3 only,
  `model.py:221`) — contextualized prototype vectors, not support feature maps.
  The block therefore matches *locations against prototypes*, closer in spirit
  to vanilla attention (which the paper's Tab.5e shows trails SAFECount by ~6
  MAE) than to SCM on raw support features.
- **One block pre-condenser vs the paper's 4 stacked blocks in the trunk** —
  capacity/iteration depth deliberately traded away for pluggability; fine for a
  first solo, but the paper's dose–response (Tab.5c) says one block is the
  weakest effective dose.

Net fidelity grade: normalization geometry exact; projection-sharing and
residual structure preserved; comparison kernel, score temperature, aggregation
content (pooled prototypes, no spatial support structure, no flip), output norm,
and input symmetry all adapted away from the paper. It is "SAFECount-flavored
dual-normalized prototype attention," not SCM+FEM as published.

## 2. Identity-at-init: holds by construction, no init damage visible

`nn.init.zeros_(out.weight/bias)` (`model.py:169-170`) forces the residual
branch to contribute exactly 0 at step 0, so the forward pass at initialization
is bit-identical to the champion. The randomly initialized `q/kv/norm`
(`model.py:165-167`) can only steer the trajectory via gradients from step 1
onward — they cannot damage the starting point. Consistent with this, the
child's first-epoch val (148.38) does not collapse relative to the parent
history (champion v1 run: 157.32 at ep1; TB curves, see §5 caveat) — there is no
smoking gun of init damage. (Strictly, step-0 identity is an architectural
guarantee, not something val curves can prove; the curves only rule out gross
damage.)

## 3. New evidence, not re-read pooled scalars — with one scoping correction

- The block reads the **flattened fine map** `fmap [B, HW, 128]` (`model.py:220`)
  and the **per-exemplar tokens** `e [B, K, 256]` (`model.py:221`) and computes
  a genuine position-wise interaction: a full `HW × K` score tensor
  (`model.py:175`) that exists nowhere else in the champion. The GCA aux reads
  only `GAP(fine)` + `e_mean` (`model.py:238-240`) — global pooled scalars. So
  the similarity block consumes strictly finer-grained evidence (per-location
  match strengths) than the channel GCA does. Both sides (`fmap`, `e`) are
  tensors the condenser already consumes, so it is a new *read* (pairwise
  interaction) of existing tensors, not new raw data — worth stating precisely.
- Writes: the enhanced map feeds **only** the condenser
  (`fmap = safe_enhance(fmap, e)` → `cond(fmap, e)`, `model.py:222-224`); the
  `fine` tensor passed to the decoder (`model.py:225`) and to GCA (`model.py:273`)
  is untouched (permute/flatten in `model.py:220` copies, the residual lands on
  the copy). So the match decision enters just the 64-d condenser branch of the
  decoder's 192-d input, raw fine features flow through unmodified — narrower
  than "injecting the match decision into the feature the decoder sums"
  (`memory/hypotheses.jsonl:42`): half of the decoder input never sees it. GCA's
  path is bit-identical to the champion's.

## 4. Pluggability: clean — OFF restores the champion line-for-line

Diff surface of child vs `tree/N0001_champion/model.py` is minimal and fully
gated on `use_safe_enhance` (default `False`, `model.py:212`; parent config has
no such key, `tree/N0001_champion/config.toml:22-26`; child sets it
`true` at `config.toml:27`):
(1) new `SafeEnhance` class (`model.py:136-179`, ~33k params: 8256 q + 16448 kv
+ 128 norm + 8320 out ≈ 33.2k, as the docstring claims at `model.py:161`);
(2) two lines conditional construction (`model.py:212-214`);
(3) two lines conditional application (`model.py:222-223`).
Backbone, FineFuser, ExemplarEncoder, Condenser, DensityDecoder, GCA, Counter
are untouched; deleting the switch restores the parent forward path exactly, as
the module docstring (`model.py:17-18`) claims. Append-only construction order
(AGENTS.md rule 13) keeps shared-module inits identical to the parent's.

## 5. Learning behavior: late-settling trajectory; asymptote claim exceeds what the logs support

Child val/MAE from its TB (tag `val/mae`, 32 pts): 148.38 → 69.44 → 49.44 →
40.00 (eps 1–4) → 27.32 (ep8) → 24.28 (ep12) → 23.52 (ep16) → 23.01 (ep20) →
23.04 (ep24) → 22.97 (ep28) → **22.9697 (ep32, best = final epoch)**; tail
ep24–32 spans 23.05–22.97, i.e. a flat plateau with a marginal (−0.002)
last-epoch best. "Still descending" is technically true and practically a
plateau: the run settled ~ep28 and crept. The canonical parent peaks earlier
(best@26, 23.2932 per `journal/events.jsonl:46`), and the ledger records the
child tail running ~sustained-0.3 below the parent tail rather than a
single-epoch spike (`memory/hypotheses.jsonl:46`) — that persistence is the
strongest point in the support claim's favor.
Reading: best-at-final-epoch + flat tail + parent-best-earlier says the block
changes the *optimization path* (slower to settle, keeps inching down where the
parent turned up) more clearly than it demonstrates a new *asymptote* — at
32 epochs neither curve has proven its floor, and a 0.002 last-epoch dip is not
a trend. Two honest caveats: (a) the margin (+0.0235) is at the harness
replicate noise floor (0.0194, `journal/events.jsonl:47`) — mechanism gain vs
lucky draw is undecidable on this single seed; (b) within-child RMSE *rises*
over the tail (90.08 → 90.87, TB `val/rmse`) while MAE falls — late MAE gains
come with growing large-count errors, i.e. error variance widens even as the
mean improves. **What the similarity block is "doing" per location cannot be
measured from these logs at all** — no similarity maps, R distributions, or
R_SN/R_EN ablations were recorded; any claim about sharpened match decisions
vs. learned temperature shift is speculation until a diagnostic run dumps R.

## 6. Why this one differs from the refuted gates and neutral H0012

- **H0009** (image-global scalar/channel gate on exemplar tokens in the
  condenser; contradicted ×2, `memory/hypotheses.jsonl`) and **H0010**
  (per-exemplar reliability gate rescaling exemplar embeddings pre-condenser):
  both multiply *embeddings* by content-pooled scalars — a global reweighting
  with no position-wise computation. H0011's frozen-random control further
  showed that class of gate is ~inert capacity (+0.202 within-bar).
- **H0012** (count-prior-conditioned condenser temperature; neutral,
  `contradicts w=1.0`, missed bar by +0.036): rescales attention *sharpness*
  globally per image — again no per-location match evidence, and its mid-run
  lead decaying into a flat tail reads as capacity the optimizer routes around.
- **H0014** is the first tested component that computes an explicit
  **position-wise match decision** (`HW × K` dual-normalized similarities) and
  **adds it into the summed feature stream** (residual onto every query token
  pre-condenser) rather than gating embedding magnitudes or attention
  temperature from pooled context. Mechanistically it is the only child that
  gives the decoder location-specific "which prototype matches here" evidence —
  plausibly why it is the first to clear (however narrowly) rather than hurt or
  wash out. The flip side of the same distinction: it is also the first whose
  claimed mechanism (sharpened per-location matches cutting clutter
  false-positives) is *unobservable* in the logged metrics — the gates' failure
  mode (global rescale ≈ constant) at least had a control; this one has none yet.

## implications

- The support is real per the pre-registered semantics but evidentially thin
  (margin ≈ noise floor, single seed, plateau tail): treat H0014 as
  "promising, unconfirmed" — prioritize a same-seed replicate and a seed+1
  contrast before stacking anything on it.
- The highest-value follow-up is a mechanism diagnostic, not a bigger block:
  dump R / R_SN / R_EN (entropy over K per location, background-vs-object
  separation) and ablate SNorm-vs-ENorm (paper Tab.5b says each is worth ≥4
  MAE in their setup) to test whether the dual norm is load-bearing here or
  the gain is just hotter attention from the 1/8 scale.
- If H0014 survives replication, the faithful-to-paper upgrades are concrete
  and ordered: restore the score temperature toward 1/√(H_S·W_S·C), aggregate
  spatially structured (pre-pooled) support features instead of pooled
  prototype vectors, and re-add the output LayerNorm — each is a one-line,
  single-hypothesis child.

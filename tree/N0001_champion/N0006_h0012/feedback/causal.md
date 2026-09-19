# N0006_h0012 — Causal identification critique (H0012, solo)

Scope: causal-identification critique only. This report does **not** re-verdict.
The machinery already recorded the outcome (`memory/hypotheses.jsonl:41`:
`contradicts` w=1.0; H0012 0.5→0.4) and that stands. The question here is: *was
this a valid, informative test of the H0012 mechanism, and what would have been
cleaner?*

Evidence actually read for this report:
- node: `idea.md`, `model.py`, `config.toml`, `result.json`, `info.json`
- parent: `model.py`, `config.toml`, `result.json`
- `memory/index.json` (H0012 block), `memory/hypotheses.jsonl` (H0012 rows)
- per-epoch `val/mae` extracted from both TensorBoard event files:
  `tree/N0001_champion/run/latest/tb/t/events.out.*` (parent) and
  `tree/N0001_champion/N0006_h0012/run/latest/tb/t/events.out.*` (child).
  TB tags present: `hp_metric, lr-AdamW, train/loss, epoch, val/mae,
  val/rmse, train/loss_epoch, train/mae` — **no tau, no per-image data**.
- `feedback/quant.md` and `feedback/qual.md` were not present; nothing was
  taken from them.

## 0. Headline

The run is a clean, well-isolated **bundle-level** test (add the count-prior MLP
+ tau hook or not), and it decisively fails its pre-registered bar. It is **not**
an identified test of the mechanism H0012 names (count → attention temperature,
dense-broad / sparse-sharp). The endpoint delta (+0.0363 best-checkpoint, +0.2129
last-epoch) is far below any resolvable scale at n=1 with no measured noise
floor, and the run contains zero measurements of tau, count, or attention
entropy, so a failed bar cannot be converted into "the count-conditioned
temperature mechanism is false" — only into "this implementation does not clear
≥0.40 within 32 epochs at seed 20260830".

## 1. Identification: what exactly was contrasted

**Solo node confirmed.** `idea.md` books exactly one hypothesis (H0012);
`info.json` `tested_hypotheses: ["H0012"]`. No composition/adjoin confound
exists by construction.

**Config contrast is exactly one switch.** `diff parent/config.toml
child/config.toml` yields a single functional line:
`use_count_temp = true` (child `config.toml:27`). Seed (20260830),
`deterministic = true`, AdamW/LR/wd/cosine, loss weights, batch size,
`epochs = 32`, data settings, `use_gca = true`, `use_xscale = true` are
byte-identical. Optimizer/loss/schedule invariant holds.

**Model contrast is one component plus its wiring** (child `model.py` vs
parent):
- new `CountPrior` = `Linear(384→64)+GELU+Linear(64→1)`, last layer zero-init
  (child `model.py:140-150`), +24,705 params (≈0.08% of ~31.3M; matches the
  0.099 MB model-size delta in the two smoke summaries, 125.299→125.398 MB);
- `Condenser.forward(tok, e, tau=None)` pre-scales q,k by `1/sqrt(tau)`,
  i.e. attention logits divided by `tau` (child `model.py:128-136`);
- `CountingHead.forward` computes `tau = clamp(1+z, 0.25, 4.0)` from
  `GAP(fine)` and `e_mean` (child `model.py:195-197`).
- With `z=0` at init, tau ≡ 1 and the child function is exactly the champion
  function; q/1, e/1 reproduces the parent attention call (parent
  `model.py:123`). Removing the switch restores the champion condenser.

**Precise treatment contrast:** for each image, `Y_child: attention logits
scaled by 1/tau(x)`, `tau(x) = clamp(1 + MLP(GAP(fine), mean_e), 0.25, 4.0)`,
learned end-to-end; `Y_parent: tau ≡ 1`. All shared weights initialize from the
same RNG stream in the same order, so shared modules are bit-identical at step 0.

**One uncontrolled nuisance:** `CountPrior` is constructed inside `CountingHead`
*before* `Counter` builds `GCA`, so the child's global RNG stream is offset when
`GCA`'s hidden `Linear(384→64)` is initialized. GCA's output layer is zero-init
(n_aux = softplus(0) = 0.693 in both runs), so the initial *function* is
identical, but GCA's hidden init — and hence its training trajectory — differs
between parent and child. This is small but real: the contrast is "champion +
{tau head, GCA hidden-init offset}", not a pure tau intervention. (Cheap fix for
future single-switch runs: construct nuisance modules before any conditional
module, or give each module its own seeded generator.) The data order is *not*
affected: the datamodule seeds its own `torch.Generator` from `seed`
(`src/cac/data/pl_datamodule.py:78-82`), so batch order is identical.

## 2. Per-epoch pattern, budget, and the pre-registered bar

Both runs: 32/32 epochs, `budget_hit = false` (parent `result.json`: best
21.4589 @32, 1704.5 s; child: best 21.4952 @28, 1771.0 s). Extracted `val/mae`
delta (child − parent), by 1-based epoch:

| phase | epochs | delta (child − parent) |
|---|---|---|
| early | 3–19 | child **worse**, up to **+0.79** (ep8) |
| lead | 20–28 | child better, peak **−0.356** (ep23), −0.066 at ep28 |
| tail | 29–32 | parent better, up to **+0.213** (ep32) |

Endpoint: child best 21.4952 vs parent best 21.4589 → **+0.0363** worse;
last-epoch delta +0.2129. The recorded note (`index.json:394`) said "child led
ep20-27 ~-0.13"; the TB stream gives a 9-epoch lead window (ep20–28) with mean
≈ **−0.19** and a single crossover at ep29. Same sign story, slightly larger
transient than recorded, but the signal **changes sign twice over the run**.

**Is +0.036 causally meaningful at n=1 seed / 32 epochs?** No. With one seed and
no measured noise floor (STATE records this explicitly; the queued replicate is
*same-seed*, so it bounds engine determinism, not seed-to-seed spread), a
0.036 endpoint difference is smaller than the child's own late-training movement
(21.495@28 → 21.672@32 = 0.18) and cannot be attributed to the treatment in
either direction. The mid-run lead (≈0.19–0.36) is the same order as plausible
seed variability it cannot be separated from.

**What the pre-registered 0.40 bar does here.** The bar is an anti-gaming
device: registered before the run, it prevents post-hoc lowering to claim a win,
at the cost of being blind to small effects. 0.40 is ~11× the observed point
estimate and ~10× typical late-epoch movement; it is a one-sided screen against
large false positives. Consequently this run has high sensitivity only to the
*strong* version of H0012 ("count-scaled temperature buys a large global MAE
drop") and near-zero power against effects below ~0.2–0.4 — the plausible range
for an attention-temperature reweighting. It is therefore a decisive miss of the
bar and a weak statement about the mechanism. The remedy inside the rules is
better-matched estimands/measurements for future tests, **not** moving this or
any bar (no such proposal is made here).

## 3. Threats, and what is / is not identified

**(a) tau is image-global, identity-initialized — the count prior is not
identified.** Nothing in the loss forces `z` to encode count: the only
supervision is MSE on the density map plus the existing
`w_cnt = 0.4` L1 on the density sum. `tau` is clamped to [0.25, 4.0] and can
learn an image-*independent* constant (bias-dominated solution), which makes the
run's estimand "a learned global attention scaling" as much as
"count-conditioned temperature". The direction H0012 needs — dense → broader →
τ↑ — is never checked: no tau value, correlation, or attention entropy is
logged anywhere (TB tags verified above). A failed bar therefore does **not**
discriminate H0012's count-specific mechanism from generic temperature scaling;
a passed bar would not have either. This is symmetric outcome/mechanism
identification failure, not evidence against the mechanism.

**(b) GCA already consumes the same inputs and is directly count-supervised.**
`GCA` reads exactly `GAP(fine)` and `e_mean` (parent `model.py:180-181`) — the
same two tensors `CountPrior` reads (child `model.py:196`). GCA's `n_aux` is
injected additively into `out["density"]` (child `model.py:247-250`), so the
`w_cnt` L1 on the density sum directly trains GCA as a global count predictor.
The champion's fixed-temperature condenser is thus already paired with a global
count path; the premise in H0012's BECAUSE clause ("the fixed condenser leaves
error on both tails uncorrected") is not the model that was tested. The run
measures the **marginal** value of a second count pathway feeding attention, not
whether count-conditioned temperature would help in a model without one. A null
marginal result is consistent with (i) the mechanism being wrong, (ii) the
information being already consumed by GCA, or (iii) the MLP failing to learn a
useful τ — this run cannot separate them. No GCA-off config exists anywhere in
the tree (all `tree/**/config.toml` have `use_gca = true`), so the redundancy
explanation cannot be checked against an existing arm.

**(c) Wall clock +66.5 s (+3.9%; 1704.5 → 1771.0 s) changes nothing.** Both runs
hit 32/32 epochs with `budget_hit = false`; the extra time is the mechanical
cost of +24.7k params (and the tau path) and is not on the causal path to the
endpoint. It does mean the treatment carries a small compute cost, irrelevant
here.

**Identified by this run (at seed 20260830, 32 epochs, this engine):** the
bundle {24.7k-param count-prior MLP + tau hook, on top of champion+GCA} does not
produce a ≥0.40 val-MAE gain; best-checkpoint delta +0.036 (child worse),
last-epoch +0.213.

**Not identified:** whether the count → temperature mechanism is true; the sign
or size of any sub-~0.2 effect; behaviour without GCA; behaviour with an
explicitly count-supervised prior; behaviour across seeds; the tau-specific
effect versus extra-parameter/init effects.

## 4. Measuring the mechanism directly (and why the logs can't)

Measurements that would falsify the *mechanism* independent of the MAE bar, if
pre-registered:

1. **tau vs true count.** On a held-out split, compute per-image
   `tau = clamp(1+z, 0.25, 4.0)` and correlate with GT count. Mechanism
   predicts positive rank correlation and monotone quintile means. A
   pre-registered falsifier could be: Spearman ρ ≤ 0, or mean τ in the sparsest
   quintile ≥ mean τ in the densest quintile, or Var(τ) ≈ 0 (prior collapsed to
   a constant → the run never tested image-conditioning at all).
2. **Stratified MAE by density.** Per-image AE (child and parent checkpoint on
   the same val split), binned by count/density quintile. Mechanism predicts
   negative ΔAE concentrated in the dense and sparse tails; a flat
   `ΔAE`-vs-count relationship falsifies "reduces error on both tails" even
   with the global bar failed.
3. **The intervening variable itself.** Hook the condenser cross-attention and
   measure attention entropy per image/head as a function of τ and count —
   this is the "attend broadly/sharp" claim in the code, not a proxy.

**Post-hoc extractability from existing logs: no.** TB carries only
loss/mae/rmse/lr; `exec.log` has no tau or per-image numbers; `model.py` never
returns or logs τ or z. The only artifact that could yield τ is the trained
checkpoint — `best.pth` lives in the server run dir (`/data/cac/.../run/latest`,
gitignored) and is **not** present in the local run dir (local `run/latest/`
contains only `exec.log`, `tb/`, `hparams.yaml`). Extracting τ therefore
requires a new server-side eval pass: possible and cheap, but it is a new
measurement, not log mining, and any such read on the same val split is
diagnostic (descriptive), not evidential, unless pre-registered on held-out
data with a numeric falsifier.

## 5. Cleaner next tests, ranked by identification cleanliness vs cost

Rules honored: pre-registered, single-switch, no silent retry of H0012
(rule 11). Nothing below proposes changing the 0.40 bar.

1. **τ-audit on the existing checkpoint** (diagnostic; no ledger booking, no
   training). Server-side forward pass with a hook on `head.cond` /
   `count_prior`; pre-register the expected sign and a threshold (e.g.
   ρ ≥ 0.3, monotone quintiles) *before* looking. Cost: minutes of GPU.
   Identification: high for "did τ learn count at all"; cannot change this
   node's verdict.
2. **Drop H0012 here; take the paper backlog** (SAFECount dual-norm; BMNet+
   per-exemplar channel gate = the clean solo H0010 test never run; DAVE-lite
   verify-and-suppress). Highest mission-EV per GPU-hour; single-switch,
   pre-registered bars all ≥0.30. Identification: high. Family value for
   count-conditioning: zero.
3. **Count-supervised prior as a NEW hypothesis** if the family is kept (z gets
   direct count regression instead of an indirect path through attention).
   Identification: high for "prior predicts count", medium for MAE. Cost: 1 run
   + eval. Regime friction: an auxiliary loss on z brushes against the
   "optimizer/loss/schedule invariant" rule → needs explicit user approval
   (§10-style) or must route through the existing count-loss structure.
   Add the placebo control while at it: a learned *constant* τ, or τ driven by
   a shuffled-image prior, to separate conditioning from generic scaling.
4. **GCA-off × count-temp interaction** (2 missing arms: {GCA off}, {GCA off +
   count_temp}; the other two exist as champion and N0006). Cleanest direct
   test of the redundancy explanation for *this* null. Book as one new
   hypothesis whose estimand is the interaction ("count_temp helps only
   without GCA"), with a new falsifier — this is not a silent retry. Cost:
   2 runs; check the §9 composition filter at booking (two switches), and note
   the interaction is a 4-cell comparison.
5. **Per-head / per-layer τ.** Most expressive reparameterization; also the
   most cost and variance, with no evidence the null is a resolution problem.
   Rank last.

Cross-cutting: before any sub-0.3 delta is interpreted, the lab needs
**different-seed** replicates (the queued same-seed `repro_run` measures engine
determinism, not seed sensitivity). That is a measurement-gap fix, not a bar
change.

## What this run does / does not license us to conclude

- **Licensed:** at seed 20260830, 32 epochs, 32/32, budget not hit, adding the
  `use_count_temp` bundle (24.7k-param count-prior MLP + tau hook) to the
  champion+GCA does not clear the pre-registered ≥0.40 improvement; endpoint
  +0.0363 best-checkpoint / +0.2129 last-epoch, and the mid-run sign flip shows
  no stable direction.
- **Not licensed:** "H0012's mechanism is false" — τ was never measured, so the
  count→temperature mapping, its sign, and the dense/sparse attention claim are
  untested by this run.
- **Not licensed:** "τ is harmful" or "τ helps slightly" — the endpoint delta is
  below any resolvable scale at n=1 with the noise floor still unmeasured.
- **Not licensed:** conclusions about count-conditioned temperature *without*
  GCA — the same (GAP(fine), e_mean) count signal is already count-supervised
  via GCA into the density, so this tested a redundant second pathway.
- **Licensed (design feedback):** the informative follow-ups are a
  pre-registered τ/count + stratified-MAE audit and, if the family survives, a
  GCA-off interaction test with a placebo control — and no interpretation of
  sub-0.3 deltas until seed-sensitivity (not same-seed determinism) is
  measured.

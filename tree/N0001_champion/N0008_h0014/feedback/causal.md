# Causal feedback — N0008_h0014 (H0014 solo)

Scope: causal-identification critique + confirmation design. NOT a re-verdict.
Verdict facts (taken from the task brief + ledger evidence note, NOT from local
files — see provenance caveat §6): parent canonical 23.2932 @ep26
(noaug+EMA+seeded+deterministic, seed 20260830); child 22.9697 @ep32;
paired delta −0.3235; semantic bar (0.30 below live parent) = 22.9932;
clearance +0.0235. Ledger: supports w=1.0, H0014 conf 0.500→0.600.

## 1. Identification strengths — what is actually paired

- Solo node: N0008 tests H0014 alone (info.json tested_hypotheses=[H0014];
  config delta vs parent is exactly one switch: `use_safe_enhance` false→true).
  No joint-composition attribution problem (contrast N0004).
- Identity-at-init: `SafeEnhance.out` is zero-initialized (weight+ bias), so at
  step 0 the block is the exact forward identity (`fmap + 0`). Stronger than
  claimed: it is also backward-inert at step 0 for q/kv — dL/d(mix) =
  W_out^T·g = 0 while W_out=0 — so q/kv/norm receive zero gradient on step 0
  and only activate from step 1. The component grows out of the parent
  trajectory rather than perturbing step 0.
- Same seed (20260830) + deterministic harness (cudnn deterministic,
  benchmark off) + seeded loaders + augment=false + dropout=0: per-epoch data
  order identical by construction; paired-contrast precision ~0.02 (replicate
  noise 0.019).
- Shared-module inits paired by append-last construction (rule 13): CountingHead
  builds fuser → exemplar → cond → decoder, then safe_enhance last
  (model.py:213-214). Shared modules' init draws are identical to the parent's.

What remains un-paired (small, but state it exactly):
- GCA init shift: Counter builds head fully (incl. the new SafeEnhance draws)
  before GCA, so GCA's first Linear layer init draws shift vs the parent.
  Mitigated to near-zero: GCA's final layer is zero-initialized, so GCA output
  is 0 at step 0 regardless, and grad into GCA's first layer passes through
  the zero final layer (= 0 at step 0). Effect is second-order.
- Optimizer-trajectory divergence from step 1: once W_out ≠ 0, shared modules
  (fuser, exemplar, cond) receive different upstream gradients than the parent.
  That is the treatment working as intended, not a confound — but it means the
  −0.32 is a trajectory-level effect, not a per-step ceteris-paribus effect.
- Best-epoch mismatch: parent best @ep26 vs child best @ep32 (final epoch).
  The registered metric is best-vs-best, which is honest, but §3c shows why
  the epoch asymmetry matters.

## 2. Fragility — what the bar-meet licenses and what it does not

- The paired contrast (−0.32) is exact under determinism: re-running both
  configs at seed 20260830 reproduces it to ~0.02. That part is solid.
- The BAR CLEARANCE (+0.0235) is what is fragile: it is comparable to
  same-seed replicate noise (0.019) and dwarfed by cross-seed level noise
  (~±1). A same-seed re-run of either arm could erase the clearance without
  anything being wrong; a different seed redraws both levels by ~±1.
- Licensed: "the component changes the outcome by −0.32 in this
  seed/protocol" + the ledger support event (w=1.0, conf 0.600). The support
  bookkeeping is correct per the anchor rule (semantics over literal text).
- NOT licensed: "the dual-normalized match mechanism is confirmed" or "the
  gain generalizes across seeds/protocols." One knife-edge support at
  conf 0.600 is evidence, not confirmation (>0.75 needs a second support).
- Extra sting: child best sits AT ep32 (budget edge, 1787s of 1800s). We do
  not know if the child had converged; the parent had already turned up.
  Best-vs-best with one arm still descending flatters the child (see §3c).

## 3. Alternative explanations for −0.32

(a) Genuine match-evidence effect (the booked mechanism): dual-normalized
    similarity selects per-location prototypes, sharpens matched locations,
    leaves unmatched residual untouched → less false-positive density on
    clutter. Consistent with the sustained tail gap (ep24–32 ~0.3 below parent
    per the evidence note — not a single-epoch spike). But consistency is not
    measurement: no match-quality or false-positive quantity was recorded (§4).

(b) Extra capacity / optimizer dynamics (~33k params: q 8.3k + kv 16.4k +
    out 8.3k at d_s=64): the component adds a learnable residual path around
    the condenser input. Even with identity-at-init, from step 1 it changes
    gradient flow into fuser/exemplar/cond and adds representation capacity.
    Part or all of −0.32 could be "extra conditioned parameters reshape the
    trajectory" rather than "dual normalization selects prototypes." The
    zero-init analysis (§1) rules out step-0 perturbation but NOT this:
    delayed activation is still activation for 31+ epochs. Discriminating
    requires the norm-ablated variant (§5ii).

(c) Parent late degradation vs child continued descent: parent best @ep26
    then drifted up ~+0.04 by ep32, while child best @ep32 kept descending.
    Decomposition needed from the per-epoch curves (not available locally —
    run/ holds only tb/ stubs; server logs are the source):
    −0.3235 = [child(ep32) − child(ep26)] + [child(ep26) − parent(ep26)] +
    [parent(ep26) − parent(ep32)]-sign-carefully. If child(ep26) ≈ parent(ep26)
    and the gap opens only in ep26–32, the "effect" is largely divergent tails
    (child descends / parent overfits upward), which is weaker evidence for a
    match-selection mechanism than a gap that opens early and persists. The
    evidence note's "sustained ep24–32 gap" argues against pure endpoint luck
    but does NOT separate child-descent from parent-ascent. Quant agent should
    extract both full val-MAE curves from server run_node.logs/TB and report
    the three terms.

## 4. The mechanism was not directly measured — what would confirm it

Claimed chain: dual-norm map R selects prototype-per-location → matched
locations sharpened, unmatched left alone → false-positive density on clutter
falls. None of this was logged. Confirmatory measurements, all post-hoc
extractable (server best.pth for N0008 + canonical parent both exist):

M1. R-distribution stats on val: per-location max_k R, entropy over k, fraction
    of locations with max R below threshold (untouched residual mass). The
    mechanism predicts a bimodal/sparse R (few sharp matches + large
    near-zero mass), vs uniform/dense R under ENorm-only.
M2. Per-location match-quality vs density error: correlate max_k R (or margin
    top1−top2) with |pred−gt| density at that location, stratified by
    background-clutter score (e.g., exemplar-dissimilarity or edge/texture
    density). Mechanism predicts error reduction concentrated at low-match
    (background) locations, not uniform.
M3. False-positive mass accounting: integrate predicted density on
    exemplar-dissimilar regions (clutter mask from low backbone-similarity to
    all exemplars) for child vs parent. Mechanism predicts the −0.32 comes
    mostly from reduced clutter mass, not from changed mass on true objects.
M4. Norm-ablation forward pass (no retraining): run the trained child with R
    replaced by (i) ENorm-only, (ii) SNorm-only, (iii) uniform 1/K on a val
    subset; if val error barely moves, the trained gain does not flow through
    the dual-norm structure.

Diagnostic-agent recipe (server, offline, no tree mutation): load both
best.pth under the canonical eval harness (EMA, deterministic); run val split
with hooks capturing (fmap, e, R, density); compute M1–M3 + M4 subset eval;
write a scratch report (never result.json/info.json/ledger). Cost: ~1–2 GPU
eval passes, no training.

## 5. Confirmation design (lab-rule compliant)

To move H0014 0.600 → confirmed (>0.75) requires a second support event with
the same falsifier semantics (≥0.30 below the live same-seed parent), never a
cross-seed level comparison at the 0.30 bar (rule §6: level noise ~±1).

Design (i) — different-seed solo re-test (RECOMMENDED, rank 1):
- Run canonical parent AND identical N0008 component at a new seed (e.g.,
  seed+1 = 20260831, or a pre-registered fixed alternate), both under the
  canonical protocol (noaug+EMA+seeded+deterministic, 32ep/1800s).
- Comparison is strictly within the new seed: child_new ≥ 0.30 below
  parent_new → second supports (w per Lead/machinery); miss → contradicts.
- Identification cleanliness: maximal — same solo contrast, only the seed
  changes; directly attacks the fragility (§2). Cost: 2 training runs ≈
  2×~30 min ≈ 1 GPU-hour on the single RTX3060 (sequential queue).
- Pre-register the alternate seed before running (no seed shopping — rule 1).

Design (ii) — mechanism-variant ablation (rank 2, only if (i) infeasible):
- Same seed 20260830, one run: identical component with dual-norm replaced by
  ENorm-only (or uniform-mix residual with matched ~33k capacity), tested
  against the N0008 result / canonical parent. If the variant keeps the gain,
  the gain is capacity/dynamics (§3b), not dual-norm selection; if it loses
  ≥0.30 vs N0008, mechanism attribution strengthens.
- Cheaper (1 run ≈ 0.5 GPU-hour, no new parent needed) but answers a different
  question (attribution, not generalization) and does NOT by itself supply the
  independent support event confirmation wants. Run it as a complement, not a
  substitute.

Also recommended regardless: M1–M4 post-hoc diagnostics (§4, eval-only cost)
before spending training budget — if R is dense/uniform or M3 shows no
clutter-mass reduction, deprioritize (i).

## 6. Provenance caveat (rule 5)

Local tree files are STALE w.r.t. this verdict: local
tree/N0001_champion/result.json + info.json still carry the v1 historical
numbers (21.4589 @ep32), and local feedback/ is empty (quant.md/qual.md
absent at write time). All numbers in this report use the task-briefed live
canonical figures (parent 23.2932 @ep26; child 22.9697 @ep32) as corroborated
by the ledger evidence note (2026-09-19T20:09:20+0800). Per-epoch curves for
§3c decomposition must come from server run logs/TB, not local files.

## What this run does / does not license

- DOES license: a same-seed paired causal effect of −0.32 for adding the
  SAFECount-style block (identity-at-init, solo, deterministic) — and hence
  the ledger supports w=1.0 (conf 0.600), the lab's first support.
- DOES license: treating H0014 as the lead confirmation candidate — it is the
  only hypothesis with a same-seed support under the canonical protocol.
- DOES NOT license: mechanism confirmation — no match-quality, sparsity, or
  false-positive measurement was made; capacity/dynamics (§3b) and divergent
  tails (§3c) remain open alternatives.
- DOES NOT license: generalization claims across seeds — clearance (+0.0235)
  is at replicate-noise scale and cross-seed level noise (±1) dwarfs the bar;
  confirmation requires design (i).
- DOES NOT license: stopping the noise-floor work — the in-flight same-seed
  replicate + seed+1 parent runs decide whether the 0.30 bar itself is
  well-calibrated; a second support at a new seed subsumes this.

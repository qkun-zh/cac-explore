# Causal feedback — N0010_h0016 (H0016 SOLO, DAVE-lite verify-and-suppress)

- Node: `tree/N0001_champion/N0010_h0016`, H0016 solo (`use_verify_mask`), seed 20260830.
- Result (read from `result.json`, `info.json`, TB tables in `feedback/quant.md`,
  gate values in `feedback/qual.md` — not re-read from scratch, not re-verdicting):
  child **22.8831 @ep32** vs canonical parent **23.2932 @ep26** → **−0.4101**,
  clears the live anchor line (22.9932) by 0.11; ledger `supports w=1.0`, H0016 conf 0.600.
- Learned gate (qual §2, server best.pth): **(a, b) = (0.1663, 0.1885)** —
  suppression ON, mild: 5.1% damping even at cos=+1, 12.8% at cos=−1, **~7.6pp
  match-vs-mismatch differential**. Suppression-only by construction (clamp cap moot, a>0).
- Scope: causal-identification critique + confirmation design only.
  The verdict stands in the ledger; nothing below re-scores it.

## 1. Identification: what the solo contrast actually identifies

**Design strengths (all verified in quant/qual, restated for the causal record):**

- **Solo node, one-switch diff.** Config diff vs parent is exactly one line
  (`use_verify_mask = true`, quant §2 table); OFF path restores the champion
  bit-for-bit (module not built, forward skips it — qual §4). Pluggable rule holds.
- **Identity-at-init, bit-exact by construction.** `raw_gain`/`b` are explicit-zero
  scalars with no RNG draw (`model.py:190-191`), so shared-module init draws stay
  the champion's per AGENTS rule 13; at a=0, gate = 1 − 0·sigmoid(·) = 1 for every
  cos and `dens * (1−0)` returns `dens` unchanged (`model.py:209-212`).
- **Same seed + determinism + seeded loaders + zero dropout** (both nodes
  20260830/true/false-identical, quant §2). Same-seed paired precision is ~0.02
  per AGENTS §6, so a −0.41 best-vs-best gap is ~20× the paired noise floor —
  the *existence* of a same-seed effect is well identified. (Cross-seed level noise
  ±1 MAE is the separate generalizability question — §3c/§5(i).)
- **Full 32/32 epochs, budget_hit=false** on both nodes; checksums match;
  best.pth bit-exact vs result.json on both. No truncation artifact.

**The treatment is wider than "the mask suppresses FPs."** Three coupled channels,
diverging only after step 0:

1. **Forward suppression** (the registered mechanism): density at the 92 indexed
   peaks multiplied by gate ∈ [0.87, 0.95] at the learned operating point.
2. **Gradient re-weighting of the decoder** (unregistered but inseparable):
   at step 0, dL/d(dens_prelim) = gate·upstream = upstream exactly (gate==1), so
   decoder gradients start identical — but once a≠0 (step ≥1), every decoder
   gradient through a peak is scaled by that peak's gate. Low-cos peaks get
   down-weighted loss gradients; the decoder is thus **re-tuned under a
   similarity-weighted loss**, not merely masked at eval. The −0.41 is the joint
   product of (1)+(2) and no solo run separates them.
3. **Fine-space shaping via the cosine path** (also unregistered): `fine` receives
   gradient through the pooled-peak/prototype cosine route once a≠0 (qual §1:
   `b` gets gradient only after a≠0; same holds for the `fine` route). The
   exemplar prototype itself is built from `fine` via roi_align, so the treatment
   also nudges the shared fuser representation — small (2-scalar headroom,
   one-step path), but nonzero over 32 epochs.

Additional subtlety: **peak selection is detached indexing** (`dens.detach()`,
`model.py:197`) — no gradient through *which* peaks are chosen, only through
gate values at chosen peaks. And the GCA uniform bias (~2% of count mass) is added
*after* the mask (`model.py:243` then `:292`), so ~2% of mass bypasses verification
entirely (qual §4) — negligible for identification, but a "verify everything"
variant would differ.

**Bottom line for identification:** the solo contrast cleanly identifies *the causal
effect of adding the 2-scalar verifier program* (forward + backward + representation
channels jointly) under the canonical protocol at seed 20260830. It does **not**
identify *the false-positive-suppression mechanism specifically* — channels (2)
and (3), plus the uniform-damping component (§2), are live confounds inside the
treatment. The hypothesis text claims the FP-suppression channel; the experiment
supports the package.

## 2. Does a mild damper delivering −0.41 cohere with FP-suppression?

**Coherence case (plausible, not proven):** a ~5–13% trim concentrated on 92
top-density peaks per image is aimed exactly where spurious mass lives (the
decoder deposits mass at salient regions; distractors that match nothing still
receive peaks). Trimming many small overcounts image-wide moves MAE by fractions
of a count per image — accumulating to −0.41 against a 23.3 baseline (~1.8%
relative) requires no large per-peak effect. Two corroborating signatures:
(i) the MAE/RMSE divergence in the child tail (RMSE bottoms @ep23 then +0.689
while MAE falls every epoch — quant §5) is the textbook footprint of trimming
many small overcounts while leaving a few large misses; (ii) the child fits train
*slightly better* than the parent (2.603 vs 2.834, quant §5), consistent with
removing spurious mass rather than underfitting. The gradual turn-on story also
fits: child trails 12 straight epochs (ep4–15) while `a` grows from 0, crosses at
ep16, leads monotonically after (qual §2).

**Incoherence tension (the uniform-damping confound):** even a perfectly matching
peak (cos=+1) loses **5.1%**; the selective tilt over the full cos range is only
7.6pp. If the parent systematically *overcounts* by a few percent on average, a
near-uniform 5–9% downscale of peak mass alone could explain a large share of
−0.41 with zero exemplar-selectivity. On current evidence (no (cos, gate)
histograms, no density dumps — AGENTS §9 stores none) we cannot say what fraction
comes from distractor peaks vs the ~5% floor applied to *all* 92 peaks, nor where
the val cos distribution sits relative to b≈0.19 (qual §3 makes exactly this point).

**Testable claims that separate the two stories:**

- C1 (selective-suppression): post-hoc application of the *learned* gate to the
  **parent's** frozen density outputs beats post-hoc application of a *uniform*
  0.91× damping of the same 92 peaks. (Inference-only; no training — cheapest
  discriminator, see §5.)
- C2 (loss-reweighting, channel §1(2)): freezing (a,b) at (0.1663, 0.1885) from
  init and retraining gives a *different* (smaller, if re-tuning matters) gain
  than learning them — if the gain survives with the gate fixed from step 0,
  forward suppression carries it; if it shrinks, the joint trajectory does.
- C3 (threshold meaning): sweeping fixed b ∈ {−0.5, 0, 0.19, 0.5} with learned a
  moves the gain monotonically with selectivity (C1's training-time cousin).
- C4 (fine-space shaping, channel §1(3)): detaching `fine` in the cosine branch
  (stop-gradient on peak+prototype, gate still learns) — if the gain persists,
  representation shaping is inessential.

## 3. Alternative explanations, scored

**(a) Genuine FP suppression — directionally supported, mechanistically unresolved.**
Sign, monotonicity (gate non-decreasing in cos), MAE/RMSE signature, no train-fit
penalty, and the 17-epoch widening lead all point the claimed way. But per §2 the
selective component is unquantified; "supported" here means the package, not the
mechanism. Weight: leading hypothesis, ~60% of the −0.41 prior mass, with half of
that plausibly uniform-damping rather than selective.

**(b) Optimizer/regularization artifact of 2 extra params — implausible as the
sole cause, with one honest caveat.** Two scalars add no capacity, draw no RNG,
leave shared init bit-identical, and start with zero forward effect — there is no
regularization channel in the usual sense, and −0.41 (≈20× paired precision)
from 2 scalars via pure optimization luck would be extraordinary. The paradox
*helps* H0016: H0014 needed ~33k params for −0.32; H0016 got −0.41 with 2, which
argues against a capacity/overfit-to-val story. Caveat (from §1): "2 params"
understates the dynamical footprint — the gate re-weights *every decoder
gradient* at 92 peaks per image once a≠0, so the optimizer trajectory of the full
decoder differs. That is not a 2-param regularization effect; it is channel (2),
and it is real — but it is part of the treatment, not a confound invalidating the
run.

**(c) Parent tail drift — quantified: a ~0.04 sideshow, not the driver.**
Decomposition from the quant epoch table (quant §2):
- Gap already at ep26 (parent's best epoch): 22.9917 − 23.2932 = **−0.3015**
  (already clears the 0.30 bar by 0.0015, knife-edge).
- Child's own descent ep26→32: 22.8831 − 22.9917 = **−0.1086**.
- Verdict delta = −0.3015 + −0.1086 = **−0.4101**; parent drift contributes
  **nothing** to best-vs-best (parent's best is fixed at ep26).
- Parent drift ep26→32: 23.3364 − 23.2932 = **+0.0432** — it inflates only the
  ep32-vs-ep32 gap (−0.4533), i.e. ~10% of the last-epoch gap, 0% of the verdict.
So ~74% of the verdict margin was present at ep26 and ~26% comes from the child's
continued descent while the parent drifted. The legitimate worry is adjacent, not
this: the child is **best@32 and still descending** (final step −0.0074 < paired
precision — flattening but no plateau, quant §2), so both curves are
budget-truncated and the verdict margin depends on where the 1800s wall fell.
A longer-budget (or earlier-convergence) comparison could narrow or widen it.

## 4. Relation to H0014 (SAFECount): complementary errors, compatible switches

Both are supported at 0.6 with ~0.3–0.4 gains (H0014 −0.3235, margin 0.024;
H0016 −0.4101, margin 0.11), both are similarity-to-exemplar ideas — but they
intervene at opposite ends with opposite capacity profiles:

| | H0014 (N0008) | H0016 (N0010) |
|---|---|---|
| Site | pre-condenser: rebuilds every fmap token as exemplar-mix | post-decoder: multiplies 92 peak densities |
| Capacity | ~33k params, representation surgery | 2 scalars, output-side nudge |
| Gradient path | long (through condenser + decoder) | one step (`dL/da ≠ 0` at init) |
| Error type | missed/blurred matches (injects match evidence where weak) | false positives (trims mass where match disagrees) |

The natural reading is **complementary error types**: H0014 helps where the
decoder under-deposits on true instances (recall side), H0016 where it
over-deposits on distractors (precision side). The MAE/RMSE divergence on the
H0016 tail (trims small overcounts, leaves large misses) is exactly what a
recall-side companion could address — a combination might do what neither does
alone. This is a hypothesis, not a finding: neither node measures error-type
decomposition.

**Composition filter check** (`src/cac/expt/mechanisms.py:29-48`): H0014 =
{components: {safe_enhance}, switches: {use_safe_enhance}}, H0016 = {components:
{verify_mask}, switches: {use_verify_mask}}, both `requires: {}`. No shared
component keys, no shared switch keys → `conflicts()` false → the pair is
**feasible** for a joint booking (and both parents' configs satisfy the empty
`requires`). Mechanically they do not fight: SafeEnhance rewrites fmap tokens
pre-condenser; VerifyMask reads `fine` + decoder density post-decoder. One
interaction to note: both read the fine/exemplar pathway, so `fine` would carry
two new gradient routes — and the mask's cosine prototype is pooled from the
*SafeEnhance-altered* fine space in a joint node, so the verifier's input
distribution shifts. Clean single-switch ablation discipline still holds
(turn either switch off independently).

**Combination test design (for synthesis/lead, not this node):** child of N0008
(or N0010) with both switches on, solo-testing the *conjunction* against the
better single parent (N0010, 22.8831) at the same 0.30 bar semantics — expect
sub-additive gains (both trim/reshuffle overlapping mass); a result near or below
min(single gains) indicates redundancy, a further −0.3 indicates true
complementarity. Cost: one full 32ep run (~1780s). Pre-register the bar against
N0010, not the champion, or the test is uninterpretable.

## 5. Confirmation design for H0016 (needs a second support for >0.75)

Costs calibrated to observed wall-clocks: full 32ep canonical run ≈ 1700–1790s
(~0.5 GPU-hr; N0010 1751s, N0008 1787s, parent 1699s). Single RTX3060, one card
per run — costs below are serial GPU time.

- **(i) Different-seed solo re-test — RANK 1 (the license-critical test).**
  Parent + child, both retrained at a new seed (e.g. seed+1), scored
  **within-seed** (child_newseed vs parent_newseed at the 0.30 bar; never compare
  absolute levels across seeds — AGENTS §6: level noise ±1 MAE dwarfs the 0.41
  margin). Exact comparison: same paired protocol, full 32ep both arms.
  Cost: **2 runs ≈ 3500s (~1 GPU-hr)**. Why first: the current margin's
  vulnerability is not paired noise (20× precision — safe) but seed-level
  variation (a same-seed −0.41 can vanish if the parent draws a lucky seed or the
  gate's turn-on dynamics differ). H0016 needs a second support for >0.75; only a
  fresh seed provides it. Also resolves the STATE.md open question (seed-20260830
  luck vs systematic cost) for this lineage. If the within-seed gap replicates at
  ≥0.30, H0016 confirms; if it lands 0.1–0.3, book partial/neutral with the
  two-seed distribution; if ≤0, the first support was seed luck.
- **(ii) Threshold-structure ablations — RANK 2 (mechanism discrimination).**
  (a) Fixed-b sweep (b ∈ {−0.5, 0, 0.5} frozen, a learned) — kills the learned
  offset while keeping suppression; (b) frozen-gate retrain (a,b fixed at learned
  values from init — tests C2); (c) stop-gradient on the cosine branch (tests C4).
  Each is one child run vs N0010's 22.8831 (same seed): does the gain survive?
  Cost: **~1750s each**; run (a)-at-b=0 first (single cheapest mechanism probe —
  if fixed-b=0 keeps most of the gain, the *learned* threshold is inessential and
  the H0016 text's "offset learned through the density loss" clause is weakened
  even if suppression per se holds).
- **(iii) Peak-count sensitivity — RANK 3 (robustness, not confirmation).**
  K ≈ 1% of N (92 peaks) is an arbitrary card choice; sweep K ∈ {8 (≈floor),
  92 (current), 368 (4%)} same-seed vs N0010. A gain that grows with K smells of
  uniform damping (more peaks damped → more mass removed regardless of content);
  a gain peaking near small K smells of genuine peak-selective suppression.
  Cost: **~1750s each**; run only after (i) replicates — if the effect is seed
  luck, K-sensitivity is moot.
- **(0) Zero-GPU pre-step (do before any of the above): C1 post-hoc test.**
  Apply the learned gate vs uniform 0.91× damping to the *parent's* frozen val
  outputs (inference-only, reuse best.pth artifacts): if selective ≈ uniform,
  the selectivity story is already weakened without spending a GPU-hour, and (ii)
  gets reprioritized toward the uniform-damping alternative (which would suggest
  a follow-up hypothesis: global overcount bias correction, not verification).

Order: (0) → (i) → (ii-a) → (ii-b/c) → (iii). Do not run (iii) before (i).

## What this report does / does not license

- **Does license:** treating the N0010 program effect (−0.41 same-seed, 17-epoch
  sustained lead, identity-at-init, one-switch diff) as a real same-seed
  phenomenon worth a second (different-seed) support attempt toward >0.75.
- **Does license:** a feasible H0014×H0016 combination test (no component/switch
  conflict; both `requires` empty) with the bar pre-registered against N0010.
- **Does license:** output-side, identity-at-init, capacity-minimal interventions
  as the lineage's favored design pattern over pre-condenser representation
  surgery (qual §5) — as a heuristic, not a law.
- **Does NOT license:** claiming the *false-positive-suppression mechanism*
  (vs uniform damping + loss-reweighting + fine-shaping) is confirmed — the solo
  run identifies the package, and the 5.1% floor damping at cos=+1 plus the
  missing (cos, gate) histograms leave selectivity unquantified.
- **Does NOT license:** treating −0.41 as a seed-general level claim or as
  robust to budget truncation — the margin is ~0.4× cross-seed level noise, the
  child is best@32 still descending, and ~26% of the verdict margin accrued in
  ep26–32 while the parent drifted; confirmation lives or dies on test (i).

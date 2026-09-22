# Qualitative feedback — N0022_h0021 (`use_exkern`, child of N0015_h0014)

Text-log analysis per AGENTS §9. BACKFILL of the missing required qual angle.
Sources read directly (all local, no ssh, no commits): node `idea.md` /
`model.py` / `config.toml` / `result.json` / `val_result.json` /
`val_perimage.json`; parent `model.py` (N0015, for booked line refs) and
parent per-image dump `/home/qkun/cac_backup/N0015_val_perimage.json` (+
`N0015_val_result.json`); `memory/hypotheses.jsonl` (H0021 create/evidence,
H0022/H0023 for contrast); `journal/events.jsonl` entries 52/53/57;
`STATE.md:160–174`; sibling `N0023_h0022/feedback/qual.md`.
Mechanism scalar `w_x=-0.0398` is the Lead-read checkpoint value booked in
`hypotheses.jsonl:40` (≈ −0.04 in `journal/events.jsonl:53`, `STATE.md:169`);
no local checkpoint exists for N0022 (`run/latest/` holds only TensorBoard),
ssh forbidden this session — cited, not recomputed.

Numbers used (all read this session): child val 21.9275
(`val_result.json:8`), dense 454.451 (`:19,33`), sparse 5.7913 (`:23`,
recompute from `val_perimage.json` agrees), mid 40.882 (`:28`); futility
`best 21.9284@24, n_epochs_done 24, futility_hit true` (`result.json:5–10`),
need 0.367/ep vs 0.12 threshold (`hypotheses.jsonl:40`). Parent slices
recomputed from `cac_backup/N0015_val_perimage.json`: mae 19.3259, dense
330.813, sparse 5.8136, mid 37.503; live parent booked as 19.3431
(`journal/events.jsonl:37,45`, `STATE.md:164`) — Δ0.017 between the two
parent figures, dense/sparse identical, immaterial to every bar below.

---

## 1. Did the BECAUSE hold — what the code actually did given w_x never engaged

**No — and more precisely: the BECAUSE was never exercised. The pre-registered
F1 branch fired instead.**

Booked BECAUSE (`idea.md:7`, ledger create `hypotheses.jsonl:39`): full-size
spatial exemplar kernels + ROI self-calibration "supply exemplar-structure
match evidence that pooled-vector similarity priors discard and restore the
training-free tail wins without learned suppression." Booked prediction P1
required `w_x > 0` decisively; booked failure F1 (`w_x ≈ 0`) was explicitly
pre-registered as **quiet null → REFUTE** (`idea.md:45–46`).

What the code was asked to do, and what happened:

- The forward branch is exactly the booked injection: `if self.use_exkern:
  cond_map = cond_map + self.exkern.w_x * _exkern_calibrated(...)`
  (`model.py:247–248`), switch on (`config.toml:30`), scalar zero-init
  (`model.py:359`, constructed last after peakcal `model.py:399`).
- `_exkern_calibrated` builds the field with a **complete detach chain**:
  `fd = fine.detach()` (`model.py:37`), kernel `weight .detach()` (`:62`),
  calibration pool `chd = ch.detach()` (`:66`), so `ch_cal` (`:74`) is a
  constant with respect to every parameter in the network — by design
  ("fully non-parametric… zero learned weights", `idea.md:31`).
- Consequently the path exposes exactly **one** gradient handle:
  `∂L/∂w_x = Σ (∂L/∂cond_map) ⊙ ch_cal` (broadcast `(B,1,H,W)` into the
  `(B,64,H,W)` residual at `model.py:248`). Nothing can ever teach the field
  itself to become useful; training's entire policy space is open-the-gate /
  close-the-gate.
- Measured policy: `w_x = −0.0398` (`hypotheses.jsonl:40`) — never engaged
  positive, slight negative drift. The gate's revealed preference under the
  joint objective (MSE + 0.4·L1, `config.toml:50–52`) is *closed*, with AdamW
  `wd=0.08` (`config.toml:42`) regularizing the lone scalar toward zero.
- Outcome vs the booked bars — **all three limbs failed**:

  | Limb (idea.md:3,7) | Bar | Measured | Source |
  |---|---|---|---|
  | val EMA | ≤19.0431 (−0.30 vs 19.3431) | **21.928 (+2.58)** | `val_result.json:8` |
  | dense gt>500 | <330.81 | **454.45 (+123.6)** | `val_result.json:19` |
  | sparse gt<50 | ≤5.45 | 5.791 (flat vs 5.814) | `val_result.json:23` vs parent recompute |
  | mid slice | (read) | 40.88 vs 37.50 (+3.38) | `val_result.json:28` |
  | P1 engage | w_x > 0 | **−0.0398** | `hypotheses.jsonl:40` |

  Futility HALT at ep24, need 0.367/ep vs 0.12 (`result.json:8–10`,
  `hypotheses.jsonl:40`) — correct per §5.16, best.pth verdict valid.

Honest reading of the deltas given w_x≈0: the forward differs from the parent
only by a `−0.04·ch_cal` broadcast bias plus run noise. With w_x parked at
≈0, the +2.58 val / +123.6 dense moves **cannot be attributed to the content
of the exkern field** — they are small-perturbation amplification on top of
the documented same-seed spread (magnitude noted per §5: exact same-seed
N0015 repro gave 21.5928, draws {19.34, 20.70, 21.59}, `journal:57` —
ignored for the verdict per user directive). The verdict therefore rests
where the booking put it: pre-registered bars (all failed) + F1 mechanism
read. What died is the *booking's claim that fusion would engage and help*;
the BECAUSE's causal statement about kernel evidence was never given a
nonzero dose to test.

## 2. Honest design assessment: was fusion-by-scalar the right way to inject CountingDINO evidence?

**Right vehicle for the claim that was booked (transfer-by-learned-fusion);
wrong instrument for the claim the motivation cared about (are the features
useful) — and the idea itself priced this exact gap before launch.**

What the formulation got right, mechanically:

- One zero-init scalar, +1 param, step-0 byte-identity (`idea.md:34,37`;
  `model.py:359`), solo switch `use_exkern` (`config.toml:30`,
  `model.py:222`) — perfectly pluggable, perfectly ablatable, append-only RNG
  (`model.py:395–399`), §11-compliant (no touch of `e`; `fd` and boxes only,
  `model.py:37,44–57`; §3 booking `idea.md:41`).
- Deliberate non-parametricity was an *anti-suppression* choice: "keep the
  new path NON-PARAMETRIC… so training cannot re-learn suppression through
  it" (`idea.md:24`). The coder honored it literally — the evidence path
  contains zero learned weights (`model.py:61–74`).

Why the optimizer ignored it, cited to code:

1. **Detach makes the field unlearnable by construction.** With
   `fd/weight/chd` all detached (`model.py:37,62,66`), the only way training
   can use the evidence is if its *instantaneous* correlation with the
   decoder's cond-gradient is positive. Measured: ≈0, slightly negative
   (`w_x=−0.0398`). No co-adaptation channel exists to change that.
2. **One scalar judges a 64-channel-shared bias.** The `(B,1,H,W)` field is
   broadcast identically into all 64 `cond_map` channels
   (`model.py:248` vs booked `expand(-1,64,…)` `idea.md:34` — equivalent).
   A single knob must decide whether one fixed spatial field helps every
   cond channel simultaneously — an alignment bar the loss never paid for.
3. **The joint loss never rewards the tail wins.** CountingDINO loses
   globally 39.70 vs 19.33 (`idea.md:24`); MSE+L1 is dominated by the global
   regime, not the 8-image tail the transfer targeted. The gradient through
   `w_x` integrates that global preference — gate-off is its rational answer.
   (Contrast: H0001's SimPrior also zero-inits its output conv
   (`model.py:284–286`) but carries *learnable* `qproj/kproj`
   (`model.py:281–282`), so its evidence could reshape toward the loss — it
   engaged and confirmed. The difference is substrate alignment plus
   learnability, not gradient plumbing.)
4. **The booking knew.** `idea.md:51`: "gradient pressure through `w_x`
   alone may under-use it"; F1 pre-registered quiet-null as REFUTE
   (`idea.md:46`). The card correctly identified that locking out learning on
   the evidence side reduces training to a one-bit on/off decision whose
   "off" outcome must count as failure.

Verdict on the design: **the formulation was the cleanest possible test of
transfer-by-fusion and it answered "no" cleanly.** Its flaw was framed at
booking time, not at coding time: it optimized for *cannot be harmed*
(non-parametric, identity-at-init) at the cost of *cannot be learned*, then
asked a loss that globally prefers the parent's regime to open the gate
anyway. STATE's own reading is consistent: the CountingDINO pipeline was
reproduced and its tail wins measured (training-free val 39.70; wins
170–370 on the 8 tail ids; `STATE.md:171–174`) — "the failure is training
dynamics, not features." H0021 refuted a **delivery interface**, not the
source model's evidence.

## 3. Failure classification: quiet-null (not "wrong") — contrast with H0022

**Class: quiet-null / mechanism-never-engaged — booked as F1, read as F1,
refuted on F1.**

- The mechanism's own engagement prediction failed first (P1: `w_x>0`;
  actual −0.0398). The path never received a positive dose, so the causal
  content of the BECAUSE ("kernels supply evidence that restores tail wins")
  stands **unexercised** — the refutation attaches to the *fusion vehicle*
  and to the booking as written (whose F1 branch demands refutation on
  exactly this reading), not to a tested-and-failed causal mechanism.
- **Contrast — H0022 / N0023 (`use_margin`), an engaged-mechanism
  refutation** (`hypotheses.jsonl:43`, `N0023_h0022/feedback/qual.md`):
  `w_m=+0.2666` positive, mask did exactly what booked, cellcal bypassed as
  designed — the mechanism received its full dose and the world still moved
  wrong (val +2.82, dense +127.9, wipe-rescue 0/11). There, the *remedy
  theory* is what's falsified; no successor may argue "it just didn't
  optimize." Here, by contrast, a successor may not claim H0021 showed the
  kernel evidence *harmful or useless* — it showed only that joint MSE/L1
  will not pay for it through one detached scalar.
- Both classes share the verdict mechanics (triple bars missed + futility
  HALT ep24) but they license opposite downstream readings: H0022 closes an
  *operator family* with its mechanism live; H0021 closes a *transfer
  interface* while leaving the transferred features' value untouched. The
  lineage now holds one clean instance of each refutation class.

## 4. What this rules out / what it honestly leaves open

**Rules out:**

1. **Transfer-by-weighting into `cond`, family-closed.** Any fusion of
   external / detached / non-parametric evidence into `cond_map` gated by a
   learned scalar (or learned gate) under the joint MSE+L1 objective at this
   operating point: the scalar's only signal is instantaneous alignment,
   measured ≈0 (`w_x=−0.0398`), and the design leaves no other knob. H0021 is
   the canonical instance; the ledger verdict already states it —
   "Transfer-by-fusion is dead" (`hypotheses.jsonl:40`).
2. **Silent retries of `use_exkern`.** Same path retuned (eps, r, kernel
   clamp, init, lr on the scalar) re-encodes a refuted mechanism → §5 rule 11
   requires `contradicts` evidence + a NEW falsifier or it dies on contact.
   The exemplar-kernel quiet-null is settled; **note for synthesis/closure:
   §11 (`AGENTS.md:254–260`) does not yet list exemplar kernels — it belongs
   in the don't-repeat register; this agent does not edit AGENTS.md.**
3. **"Just force the gate open" as a fix of this card.** The optimizer's
   revealed preference drifted slightly *negative*; a fixed-nonzero-w_x card
   would be testing field value directly (a different claim), needs its own
   falsifier, and starts against the measured sign.
4. **Quiet readings of this refutation.** F1 was pre-registered and read —
   no successor may argue exkern "wasn't really tested due to noise" to
   re-litigate engagement (the noise datum, `journal:57`, is noted for
   magnitude only, per directive).

**Honestly leaves open:**

1. **Whether CountingDINO-style spatial kernel evidence has value under ANY
   supervision.** Quiet-null never opened the gate; the correct statement is
   "joint MSE/L1 won't buy it through one scalar," not "the features don't
   help." The features' training-free worth is measured and intact
   (`STATE.md:171–174`; tail table `idea.md:11–22`).
2. **Distillation-style supervision** — training the head against external
   targets (CountingDINO outputs / tail-win supervision) so the evidence gets
   gradients that actually encode its value. Ledger: "different program"
   (`hypotheses.jsonl:40`). **Out of scope unless the user expands scope.**
3. **Structurally-forced / training-free injection** (fixed nonzero weight,
   inference-time fusion): untested, different claim, new card + new falsifier
   required; carry the mildly negative w_x sign as prior caution.
4. **Untouched by this card:** H0023 (`N0024_h0023`, currently training) —
   substrate-swap on the SimPrior energy field, a different substrate and an
   engaged-mechanism program; and the load-bearing parent stack
   (frozen hs(2,3) + cross-attn condenser, `AGENTS.md:260`).

## 5. Booking-quality notes: was the falsifier well-posed? booked-vs-code?

**Falsifier: well-posed, and unusually self-aware.**

- Triple conjunctive numeric limbs, all pre-registered and bound to the live
  parent (`idea.md:3,7`): val ≤19.0431, dense <330.81, sparse ≤5.45 —
  machine-checkable from `val_result.json` alone; all three failed by
  mechanism-level margins on the val/dense limbs.
- F1 quiet-null → REFUTE was **booked before the run** (`idea.md:46`) with
  the risk priced in §5 (`idea.md:51`). The card named its own most likely
  death and the falsifier did exactly its job. P1 (`w_x>0`) gave the
  mechanism read a pre-registered anchor (`idea.md:45`) — exemplary.
- Sparse-limb footnote (same nit as N0023): the 5.45 bar is tighter than the
  live parent's own 5.814 (recompute; `STATE.md:164`) — an
  unattainable-by-parent limb. Binding per §6 semantics; the child was flat
  (5.791 vs 5.814), so this limb failed on bar-tightness, not regression.
  Decisive limbs were val +2.58 and the F1 read.

**Booked text vs implemented code: match on every load-bearing item**
(checked line-by-line this session):

- Detach/non-parametric path: booked `idea.md:29–33` ≡ `model.py:37,51–52,
  56,61–66,74` (clamp [2,12], fixed r=4, Kbar mean, ×1.0 calibration,
  eps 1e-6, nan_to_num).
- Fusion scalar zero-init, constructed last after peakcal, sole addition at
  the cond_map site: booked `idea.md:34,36` ≡ `model.py:359,399,247–248`;
  `decoder in_ch` untouched (`model.py:213`).
- idea's attach refs correctly target the **parent** `model.py` (stated
  `idea.md:36`): verified `:156` (decoder in_ch), `:163` (use_peakcal flag),
  `:172–186` (cond_map site), `:312` (peakcal attach), `:315–316` (temp-pin)
  against N0015's file — no drift.
- Step-0 identity, temp-pin hygiene: `idea.md:37` ≡ `model.py:393–394`
  (peakcal pin) plus `model.py:402–403`.
- Benign deltas (none verdict-affecting): (a) degenerate-box global-kernel +
  global-mean calibration fallbacks (`model.py:58–60,70–73`) not in the
  booked math — defensive, H0013 NaN lesson; (b) coder added a temp-pin
  under `use_exkern` (`model.py:402–403`) though `idea.md:37` said the path
  has nothing to pin — redundant (config has `use_peakcal=true`,
  `config.toml:29`, pin already fires at `:393–394`), value-identical;
  (c) broadcast add ≡ booked `.expand` (`model.py:248`); (d) conv output crop
  for k=4/pad=2 off-by-one (`model.py:64`) which the idea left implicit —
  correctly handled.

**Process nits (not verdict-affecting):**

- Ledger create text lost the `<` in "gt<50" → reads "sparse gt50"
  (`hypotheses.jsonl:39`; evidence note repeats it at `:40`) — consistent
  with shell-redirection escaping at booking; intent unambiguous from
  `idea.md:7`. Markers IF/IN/THEN/BECAUSE/DISPROVED present in order.
- `idea.md` lacks the runner-parseable line `1. **H0021** — <booked text>`
  (same gotcha as N0023); `info.json` carries `tested_hypotheses: []`, and
  local `info.json` still shows `status: "proposed"` / `best_metric: null`
  despite `result.json` + `val_result.json` present — repair via the
  discovery/run API, never by hand (§5.3).
- Parent figure note: booked live 19.3431 vs backup dump 19.3259 for the
  same checkpoint (Δ0.017, dense/sparse identical) — immaterial; every bar
  above uses the booked numbers.
- Noise datum (magnitude only, ignored for verdict per directive):
  {19.34, 20.70, 21.59} same-seed draws, exact-repro +2.25
  (`journal/events.jsonl:57`) — the child's val/dense deltas sit inside that
  scale, which is precisely why the quiet-null mechanism read, not the
  headline deltas, carries this verdict.

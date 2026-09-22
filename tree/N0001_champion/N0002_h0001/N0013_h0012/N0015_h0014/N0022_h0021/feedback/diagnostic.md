# Diagnostic — N0022_h0021 (H0021 `use_exkern` exemplar-kernel) — FAILURE

- **id/parent**: N0022_h0021 ← N0015_h0014 (live, val 19.3431 @ep32 v2).
- **mechanism**: non-parametric full-size RoI exemplar kernels depthwise-convolved over
  `fine.detach()`, ROI self-calibrated to mean-1 in exemplar regions, fused into `cond_map`
  by ONE zero-init scalar `w_x` (`idea.md:26-37`; `model.py:22-76,247-248,347-359`).
- **outcome**: futility-HALTED ep24 (`result.json:5-10`: `best_mae` 21.9284, `best_epoch` 24,
  `futility_hit` true, `cfg_overrides` {augment, futility_bar 19.0431}); all triple bars missed;
  ledger verdict: `contradicts` w=0.85, H0021 conf 0.15 (`memory/hypotheses.jsonl:40`;
  confidence path 0.5 → refute).
- **constraints**: frozen backbone, head-only, seed 20260830 fixed, v2 protocol, AGENTS §5.16
  futility, AGENTS §11 no-repeat register. No commits, no ssh performed for this document.

Verified locally: slices recomputed from `val_perimage.json` reproduce `val_result.json`
exactly (sparse 5.7913 / mid 40.8822 / dense 454.4514, n=895/374/17; overall 21.9275 vs
`val_result.json:8` 21.9275); parent slices recomputed from `/home/qkun/cac_backup/N0015_val_perimage.json`
(19.3259 / 5.8136 / 37.5029 / 330.8131 — EMA-vs-booked deltas ≤2e-3, same as N0023's check).

---

## 1. Failure taxonomy

**Primary class: quiet-null (F1).** Distinct from H0022's class (engaged-mechanism-wrong):
N0023's `w_m=+0.2666` engaged at full strength and failed; N0022's `w_x=−0.0398` **never
engaged positive** — the booked P1 ("`w_x > 0` decisively at end of training, path used not
ignored", `idea.md:45`) is falsified outright (`hypotheses.jsonl:40` evidence note). The
transfer field was computed every forward and the scalar did receive gradient (it moved
0 → −0.0398, so "gradient ~0" in the evidence note is approximate — small, consistently
slightly-negative gradient, never a rewarded direction), but joint MSE+L1 never rewarded
the path, so the learned fusion ignored it. Mechanism verdict: **CountingDINO
non-parametric transfer does not survive joint training dynamics; transfer-by-fusion is
dead** — distillation-style supervision would be required, which is a different program
(see §3.2).

**Noise decomposition of the +2.59 headline** (draws {19.3431, 20.697, 21.5928}, mean
20.544, sd ≈ 1.13 — `perimage_diagnostic.md:228-229`; HANDOFF heuristic ≤±1.5 untrustworthy,
≥+2 credible — `HANDOFF.md:34-35`):
- **+1.20 is parent draw-luck** (20.544 − 19.3431): the live parent is the minimum of the
  observed same-seed draws ("favorable coin flip", `perimage_diagnostic.md:229`).
- **+1.38 is the child's deviation above the draw-mean** (21.9284 − 20.544) ≈ 1.2σ of a
  single draw — noise-consistent in isolation; under an exchangeable-draw null
  sd(child−parent) ≈ 1.60, so +2.59 ≈ 1.6σ — the level gap alone would be HANDOFF-gray.
- **What noise cannot explain — the structural reads:**
  - **Dense +123.6** (454.45 vs 330.81, `val_result.json:19` vs booked bar): the same-seed
    retrain band for N0015's dense tail is **[300, 361]** (H0020 booking,
    `hypotheses.jsonl:38`). 454.45 exceeds the band's upper edge by **+93.4** — at least
    +93.4 of the +123.6 is structural, whatever level noise does to val MAE. Note the
    honest tension: a quiet-null fusion should be near-inert, yet dense is structurally
    broken. Three candidate causes, none exculpating the card: (a) `w_x=−0.0398 ≠ 0` — the
    residual `w_x·ch_cal` was live at −4% scale every forward, and `ch_cal` is largest
    off-exemplar where the self-calibration denominator is small (`model.py:74`); (b) the
    mere presence of the gradient path perturbs the joint trajectory (rule 13 guarantees
    init-RNG identity, not trajectory identity once `w_x` receives gradient); (c) **epoch
    confound**: both futility-halt cards land at dense ≈455 at their ep24 best (N0022
    454.45, N0023 458.69) vs parent 330.81 @ep32 and band [300,361] from 32ep retrains —
    an untested ep24-vs-ep32 dense control (new read 7, §4).
  - **Wipe-set 0/11** (recomputed from `val_perimage.json`; ≥0.65 needed per the family's
    R2): 840→0.301, 851→0.478, 865→0.339, 935→0.126, 3428→0.144, 3481→0.056, 3482→0.099,
    3484→0.051, 3487→0.292, 3488→0.143, 7656→0.299 — identical story to N0023: every wipe
    stayed wiped (935 deepened 0.403→0.126). The booked transfer target never moved.
  - **cdino-win-8 R2 read** (the card's own R2, `idea.md:24,47`, never run at booking time;
    computed now from both per-image dumps): child vs parent error delta
    935 **+579**, 865 −81, 5059 −6, 3481 −2, 3482 −3, 2850 −12, 3483 +3, 7656 +13 —
    **0/8 transferred**; the only large move is 935 *worse*. F3 ("MAE bars pass but
    dense-tail unchanged → transfer failed", `idea.md:46`) is also satisfied in spirit:
    the transfer mechanism delivered nothing on its own measured target set.
  - **Mid +3.38** (40.88 vs parent 37.50, n=374, `val_result.json:28`): same direction and
    rough size as N0023's mid +4.21 — a stable slice, not draw noise.
  - **Trajectory/futility**: best stuck at 21.9284 by ep24, required slope 0.367/ep vs
    gate 0.12 (`hypotheses.jsonl:40`) — hopeless-but-running, exactly the H0011-class
    death §5.16 exists to catch; HALT saved ~8 epochs without changing the verdict
    (extrapolating at the *parent's* late slope −0.28/ep from 21.9284 gives ≈19.69 at
    ep32 — still +0.65 above the 19.0431 bar).

**Classes explicitly ruled out.**
- *Gate-too-strict*: see above — no plausible ep32 finish clears the bar.
- *Operator-wrong / implementation-vs-booking*: no — as-booked (§2).
- *Sparse conjunct*: 5.791 vs bar 5.45 is the standing parent miss (N0015 own 5.814);
  child is flat (−0.02) on sparse. Sparse never decided this verdict.
- *Draw-noise as sole cause*: contradicted by dense +93.4-over-band, 0/11 wipes, 0/8
  transfer targets, mid +3.38.
- *Mechanism-engaged-wrong (H0022's class)*: excluded by `w_x=−0.0398` — never positive.

**Taxonomy verdict: quiet-null primary** — a non-parametric externally-motivated evidence
field fused by a lone zero-init scalar under joint MSE/L1: the scalar never engaged, the
transfer targets never moved, and the run still paid a structural dense regression (≥+93.4
vs retrain band, with an open ep24-halt confound). Refutation rests on P1 (w_x) + 0/11 +
0/8 + dense-over-band — reads noise cannot produce.

---

## 2. What was actually tested — dispatch vs booking

**Booking.** Ledger one-liner `hypotheses.jsonl:39`; authored card `idea.md:7` (mechanism
math `idea.md:26-37`, predictions/reads `idea.md:43-47`); launch journaled
`journal/events.jsonl:52`: `run_node N0022_h0021 --set augment=true --set futility_bar=19.0431`,
triple bars ≤19.0431 / dense <330.81 / sparse ≤5.45.

**Implementation audit (checked line-by-line against `model.py`).**
- Steps 1–5 of `idea.md:28-33` all present in `_exkern_calibrated` (`model.py:22-76`):
  `fine.detach()` (:37), per-box roi_align at clamped size ∈[2,12] (:51-55), fixed r=4
  adaptive pool (:56), mean over K (:61), depthwise conv `groups=C` with detached weight
  (:62-63), channel mean (:65), ROI self-calibration `ch/(c+1e-6)*1.0` with `nan_to_num`
  (:66-74). Every pre-registered guard of `idea.md:37` (clamp, r=4, eps, detach, nan_to_num)
  is implemented.
- **Fusion**: booked `cond_map + w_x * ch_cal.expand(-1,64,-1,-1)` (`idea.md:34`);
  implemented `cond_map + self.exkern.w_x * _exkern_calibrated(...)` (`model.py:247-248`) —
  (B,1,H,W) broadcasts to (B,64,H,W), numerically identical to the booked expand. As-booked.
- **Defensive deltas beyond the letter of the booking** (benign, not contradictions):
  degenerate-box fallback to a whole-image pooled kernel (:58-60) and the
  `match[..., :H, :W]` crop (:64, forced by k=4/pad=2 output geometry) — both deterministic,
  RNG-free, and in the spirit of `idea.md:37`'s guard list.
- **Parent-stack integrity**: `config.toml:28-30` keeps `use_cellcal`/`use_peakcal` true
  AND the runtime dispatch keeps them active — SimPrior runs with cellcal+peakcal kwargs
  (`model.py:234-244`), then the exkern residual is added *separately* (:247-248). **No
  N0023-style exclusive-dispatch bypass**: cellcal executes on all images (parent stack
  intact; unlike `N0023_h0022` where the margin branch returned early).
- **Rule 13 / hygiene**: `ExKern` constructed LAST after `peakcal`, zero-init
  (`model.py:395-399`); temp-pin condition includes `use_exkern` (:400-403), redundant with
  the peakcal pin (:391-394) but harmless (idempotent `requires_grad_(False)`).
- **Launch vs config file**: `config.toml:34` reads `augment = false` but the run carried
  `--set augment=true` (`result.json:16`, `journal/events.jsonl:52`) — the same
  config-file-vs-override language gap flagged in the N0023 diagnostic §2 caveat; runtime
  is correct, the file alone misleads a reader.
- **Checkpoint/eval chain**: `val_result.json:40-42` — `ckpt_epoch` 24, config/model
  sha256 match `result.json:13-14`; bars evaluated on the futility-kept best.pth as §5.16
  requires.

**Is the hypothesis only partially tested? No.** Every booked clause executed: the path
ran (flag on), `w_x` was read (F1), all three bars evaluated (F2), the transfer target
images were scored (R2, now computed: 0/8), futility gate fired with best.pth retained.
There is no implementation-vs-booking contradiction of the N0023 dispatch-story kind. The
only unbooked-at-verdict item was R2's per-image read, which this backfill executes and
which confirms (does not complicate) the verdict. One wording hygiene note for the record:
the evidence note's "gradient ~0" overstates — `w_x=−0.0398` proves gradient flowed, just
never toward a rewarded positive engagement; the operational meaning (quiet-null) stands.

---

## 3. Lessons / constraints

### 3.1 For the running N0024 (H0023 `use_antimatch`) — do not contradict the N0023 diagnostic

Inherit and re-verify (N0023 diagnostic §3, all still valid): cellcal runtime-ACTIVE
(`w_c ≈ +0.15`, dispatch fixed at `N0024_h0023/model.py:178-188`), temp-pin, bars/futility
as booked, **ep24 gate: N0024 must be ≤20.00 at ep24 to finish** (a HALT in (20.0, 21.6) is
draw-contaminated), trajectory crossover watch at ep16, mechanism reads R1/R2 before bars,
±1.2/1.6 noise decomposition. N0022 adds only genuinely new constraints:

1. **Quiet-null is structurally inapplicable to N0024 — and that matters for reading it.**
   H0023 introduces **zero new parameters** (`N0024_h0023/idea.md:3`): no new zero-init
   fusion scalar exists to ignore the new field. Its substrate swap either fires and
   converts (R1) or doesn't (F1) — there is no `w_x`-style escape hatch. So if N0024's
   level comes in noisy, the mechanism reads are *more* decisive than they were here
   (N0022's F1 made its level moot; N0024's verdict lives or dies on R1/R2/w_c).
2. **The 0/11 wipe table now has two independent child entries** — N0022's ratios (§1)
   alongside N0023's. If N0024 also fails R2, three consecutive cards leave the measured
   wipe set untouched: the wipe is then firmly localized upstream of *all three*
   intervention styles tried (additive external field, sign-masked gain, substrate swap),
   which sharpens the third card's brief toward the exemplar-encoding interface rather
   than yet another cond_map residual.
3. **Dense ≈455 at ep24 is now a two-card pattern.** N0024's dense read (N0023 diagnostic
   read 5) must be compared not only to 330.81 but tagged with its halt epoch — see new
   read 7. A third ~455 would demand the ep24 control before any mechanism claim.
4. **Sparse stays a non-decider** (5.791 here vs 5.45 bar, parent 5.814) — same ruling as
   N0023 diagnostic §3.7: dense + wipes arbitrate.

### 3.2 For any future external-evidence / CountingDINO-transfer path

1. **Transfer-by-fusion is settled dead** (H0021 evidence, `hypotheses.jsonl:40`): a fixed
   externally-motivated field + one learned scalar under joint MSE(+w_cnt·L1) is a
   quiet-null design — the loss never rewards the path, `w_x` parks ≈0/slightly-negative,
   and the run still risks structural dense damage. Re-encodings must book as
   `contradicts` on H0021 with a *new* falsifier (AGENTS §5.11).
2. **The only theoretically live variant is distillation-style supervision** (aux loss
   matching teacher/tail targets) — explicitly booked in the evidence note as "a different
   program". Under the standing regime the loss is invariant (`AGENTS.md:29`:
   MSE(+w_cnt·L1)); adding a distillation term is a **regime change requiring user
   authorization**, not a next card. Do not queue it autonomously.
3. **Feature quality is exonerated; dynamics are the killer.** The CountingDINO pipeline
   itself was reproduced (training-free val 39.70; tail wins 170–370 real —
   `STATE.md:168-173`, `idea.md:11-24`), and the hybrid/router family is separately dead
   (`HANDOFF.md:37-40`). So the remaining external-evidence options are: (a) distillation
   under a user-lifted loss constraint, (b) nothing. Deterministic-inference blending is
   closed.
4. **§11 register update**: exemplar kernels (full-size RoI depthwise + ROI
   self-calibration fused into `cond_map`) are now quiet-null settled alongside the
   existing bans — no exemplar-gating, no shared projections, no exemplar-kernel fusion
   re-encodes without a new falsifier.

---

## 4. Follow-up reads on N0024 (when it lands) — coordinated with the N0023 diagnostic's six

The six reads of `N0023_h0022/feedback/diagnostic.md` §4 stand unchanged (wipe-ratio pairs;
WIPE/KEEP/MID attribution with route-fire/evg_sum/gain_on_neg; mid-slice vs 37.50 dispatch
attribution; KEEP-guard route precision; dense-block 17; checkpoint scalars w_c/w_p/temp).
**Added — only genuinely new ones from N0022's corpse:**

7. **Halt-epoch dense control** (N0022-specific, §1c): tag N0024's dense value with its
   halt/final epoch and compare against the two ep24 entries (454.45 / 458.69) and the
   32ep retrain band [300,361]. If N0024 lands dense ≈455 *and* its mechanism reads are
   mixed, pull the parent's ep24 dense from tensorboard (`run/latest/tb` if the parent
   curve logs per-epoch val slices; otherwise record the control as unresolvable) before
   attributing dense to the mechanism. This read did not exist for N0023 — its dense was
   read straight against 330.81.
8. **cdino-win-8 transfer table** (H0021's R2, `idea.md:47`, executed post-hoc here):
   per-image child-vs-parent error deltas on {935,865,5059,3481,3482,2850,3483,7656} from
   `val_perimage.json`. Baseline to beat: N0022's 0/8 with 935 at +579 (§1); parent ratios
   and N0023's wipe ratios already tabulated. This covers 5059/2850/3483 — images *outside*
   the 11-wipe set — testing whether the substrate swap converts mid-band transfer targets,
   and retroactively completes the read H0022 never ran. One-line subset filter, no new
   inference.

(No config/dispatch audit added: N0023 diagnostic §3.2's `w_c` check already settles the
N0024 dispatch-fix question — `w_c ≈ 0` ⇒ invalid card.)

---

## 5. Recommendation (sequencing)

**Run the sequence: N0024 verdict → book the third registered card → paper decision —
consistent with the N0023 diagnostic §5, with my card count stated explicitly: the HANDOFF
"3 cards no >1.5 → paper" rule (`HANDOFF.md:64`) counts exactly three post-HANDOFF cards —
(1) N0023/H0022 MarginCal (ran: +2.82, no drop), (2) N0024/H0023 use_antimatch (pending),
(3) the registered exemplar-distinctness / exemplar-encoding-interface card
(`HANDOFF.md:48` "remaining 残血 idea, 期望为负"; dense-targeted variant `HANDOFF.md:62`) —
and it does **not** count N0022/H0021: exkern ran 2026-09-21, *before* the 2026-09-22
HANDOFF, and is already folded into the history that *created* the budget
(`HANDOFF.md:48`: "H0010–H0021 除 cellcal 外全部 REFUTED" — that sentence is precisely why
HANDOFF then granted three fresh cards); retroactively charging exkern against the budget
would shrink a budget granted with exkern already known, i.e. outcome-shopping the stop
rule. Operationally: arbitrate N0024 by mechanism reads first (R1/R2/w_c, ep24 ≤20.00
gate, plus new reads 7–8), bars second; unless N0024 clears the triple bars with R1/R2 true
(which would install a new live parent and void the stop rule), book the third card as the
final unexhausted readout-side probe — its brief is strengthened by N0022 (three
consecutive intervention styles leave the wipe set untouched ⇒ localize at the
exemplar-encoding interface, and pre-register an engagement read like P1 so a quiet-null
cannot pass as "waiting longer") — and only after *that* card also fails to deliver a
**>1.5 net drop vs 19.3431 (i.e. val ≤ 17.8431)**, trigger paper closure
(`HANDOFF.md:64`). N0022's failure is itself paper-grade corpus (quiet-null transfer +
structural dense + reproduced teacher tail wins) but does not substitute for card 3 of 3.

# Synthesis — N0024_h0023 (H0023 `use_antimatch` AntiMatch substrate-swap) — REFUTED (futility-halted, mechanism-wrong)

## 1. Consolidated verdict

Booked: H0023 `use_antimatch` — a fixed stop-gradient per-image substrate swap: on
images whose whole-image top1 cosine mean is negative, the two evidence channels fed
to the confirmed zero-init `out` projection are replaced by the mean-1 fine-energy
field `mhat` (both channels), *before* the parent cellcal gain, with **zero new
parameters** and every other parent line byte-identical (ledger create
`hypotheses.jsonl:42`, mirrored `idea.md:9`); triple bars vs live
19.3431 / 330.81 / 5.45 + R2 ≥7/11 wipe-rescue with non-negative routed post-gain
sum. Design intent: anti-phase similarity is the wipe's measured substrate (wiped-set
top1 mean −0.258, 66.7% negative cells, post-gain sum −5301), so swapping to an
in-phase energy field on exactly that anti-match minority converts the identical
confirmed cellcal gain from negative-deepener to positive mass source "without
touching the healthy majority."

Node history (must note): the original draft attach used an exclusive
`if use_antimatch:` branch that dropped cellcal+peakcal for all images; audit caught it
pre-launch ("exclusive if drops cellcal+peakcal … dispatch fix + smoke before launch",
`journal/events.jsonl:59`), fixed to `**am` kwarg composition through all four parent
dispatch arms (`model.py:178-188`), so cellcal is source-faithful as booked — contrast
sibling H0022, where exclusive dispatch *was* the booked replacement.

Ran: N0024_h0023, v2 protocol, seed 20260830, launch `--set augment=true
--set futility_bar=19.0431`. Futility HALT at ep24 (best 23.0618, need 0.5251/ep vs
margin 0.12, saved ~8 epochs; `result.json` futility_hit true, n_epochs_done 24/32).
Verdict on the futility-kept best.pth:

| Bar / read (booked) | Threshold | Child | Parent | Result |
|---|---|---|---|---|
| val EMA MAE | ≤19.0431 | **23.0618** (best @ep24) / 23.0572 (`val_result.json`) | 19.3431 | **FAIL** (+3.72 vs parent, +4.02 vs bar) |
| dense gt>500 MAE | <330.81 | **491.81** (n=17) | 330.81 | **FAIL** (+161.0, +48.7%) |
| sparse gt<50 MAE | ≤5.45 | **5.9717** (n=895) | 5.8136 | **FAIL** (+0.16; bar was a standing parent miss) |
| mid 50–500 (read) | — | 42.6369 (n=374) | 37.50 | +5.13 (+13.7%) |
| R1a routed `evg_sum` ≥0 | ≥90% of routed | **531/531 = 100%** (mean +18804.2, construction identity ≤0.04%) | — | **PASS** |
| R1 route fires on booked 11 wipes | ≥90% | **7/11 = 63.6%** | — | **FAIL** |
| R1c route prevalence (minority ~2–3%) | booked minority | **531/1286 = 41.29%** | — | **FAIL** (20× over) |
| R2 wipe rescue | ≥7/11 at ≥0.65; mean 0.241→≥0.55 | **0/11**; mean **0.2002** (fell); 0/11 ≥0.50 | — | **FAIL** (deciding) |
| R3 no new wipes + dense ≤3% | 0 new; dense ≤+3% | **19 vs 11 wipes (8 new)**; dense +48.7% | — | **FAIL** |
| Mechanism | w_c alive ~+0.15 | **w_c=0.02822391** (alive, 5.2× below parent 0.147648); gain_on_neg 1.0186; temp 0.07 pin; w_p 0.00643 | — | alive-but-weak |

Triple conjunctive gate **0/3**; overall incl. R1/R2/R3 **1/8** (R1a only — and R1a
is construction-true for any routed image). Headline signature: **the swap fired
construction-true (+18.8k, 0/531 negative) and every count bar still failed** —
REFUTED as booked, mechanism-wrong (F3 primary, F2 co-fired amplifier), not a quiet
null. AE: Δ total **+4798.55**; KEEP-27 **+3275 (68.3%, 0/27 improved)**, wipes +605
(12.6%), mid +776 (16.2%), sparse +141 (2.9%); 8 new dense wipes alone +1674.8 (34.9%).
Noise decomposition (diagnostic): +3.72 ≈ **+1.20 parent draw-luck** (same-seed draw
mean 20.544) + **≈+2.52 child deviation (~2.2–2.3σ)** — beyond the HANDOFF ≥+2 line;
HALT at 23.06 is above even the worst draw (21.59) by +1.47 (not draw-contaminated).
Ledger verdict already booked: `contradicts` w=0.85 from N0024_h0023,
2026-09-22T12:48:03 (`hypotheses.jsonl:44`); journal `events.jsonl:62` — this file is
the closing backfill.

## 2. Distinct contributions of the 4 feedbacks

| Feedback | Unique load-bearing claims |
|---|---|
| quant | Full bar-verdict table (0/3 triple + R1b/c FAIL, R2 0/11, R3 8-new-wipes) with exact cross-checks (val_perimage recompute 23.0572 = val_result; tb @ep24 = result best; parent dump 19.3259 vs booked 19.3431 Δ0.017 immaterial; n-matched 1286-pair dumps, 0 id/gt mismatches); slice deltas dense +160.998 / mid +5.134 / sparse +0.158; **paired AE partition: total +4798.55 with KEEP-27 carrying 68.3% and 0/27 improving, 8 new wipes +1674.8 (34.9%)**; all-24-epoch trajectory (ahead of N0023 through ep7, crossover ep8, behind every epoch after, +0.8987 at ep24); futility arithmetic (need 0.5251 > 0.12, ceiling 20.0031, +3.06 above); R2 per-image table (7/11 routed; of those 4 ratios *decreased*; unrouted 840/935/3482/3487 also stayed; 935 fell 0.403→0.139); gate-fire by bucket (KEEP 12/19=0.63, WIPE 13/19=0.68, mid 194/353=0.55, sparse 312/895=0.35) |
| causal | Root audited link-by-link as a 4-arrow chain: (A) swap executed exactly — pre-swap `ev_sum_pswap` all <0 on routed-7 → post-swap `evg_sum` mean +18823 (parent same-7 sum **−62k**, ≈+81k swing), construction identity 18432·exp(w_c·ln2) predicted 18796.1 vs observed min 18796.83 (**≤0.04%**); (B) despite that, **0/11 rescued**, ratios 0.05–0.48, mean fell 0.241→0.200 — within-checkpoint, draw-invariant; (C) cost landed on dense (+161), 8 new wipes (6/8 routed), and the healthy majority (41.29% over-fire; w_c collapsed 0.148→0.0282, gain 5× weaker — PLAUSIBLE two-regime gradient stats on shared `out`/`w_c`, not ablated); (D) F3 consumption-interface veto by elimination — sign fixed, gate fired, gain alive, counts unmoved ⇒ locus **below evidence/gain layer** at `out→cond→decoder` (`model.py:224-226,181-188,128-140`); F2 coarse gate is co-fired amplifier (explains magnitude of damage, cannot explain B — fixing the gate leaves correctly-routed 7 still at 0/11); F1 did not fire |
| diagnostic | Failure taxonomy: **mechanism-wrong, F3 primary + F2 dominant secondary**; quiet-null / operator-wrong / gate-too-strict / draw-noise-as-sole-cause explicitly ruled out (HALT beyond every observed draw; dispatch bug fixed pre-launch, w_c≠0 proves fix held; evg_sum never negative); noise decomposition (+1.20 luck / +2.52 ≈2.2σ); dispatch-vs-booking adjudication: **fully tested as booked**, two honest residues (learned gain ~5× weak — outcome of intervention, not operator error; minority-scope premise empirically false at 41% — legitimate refutation via R3, not under-testing); **two-card closure statement** (H0022+H0023 both as-booked, both ep24 HALT, both 0/11 → wipe localized below evidence/gain layer); §11 register check; **pre-registered N0024 watch items from sibling now all answered** (mid 42.64 with cellcal retained ⇒ family-caused, kills two-regime F2-successor; w_c alive-but-weak third state beyond the binary engagement check); HANDOFF budget ruling (§4) and recommendation |
| qual | BECAUSE split three ways: Claim A measurement half held / causal half **FALSE** (anti-phase is correlate, not substrate — sign replaced, wipes stayed); Claim B arithmetic half true-by-construction / outcome half **REFUTED**, "identical gain" source-true only (runtime w_c collapsed 5×); Claim C construction half true / empirical half **decisively false** (41% of all val, 63% of KEEP — majority is what got touched); failure-mode table (F1 NO, F2 YES, F3 YES primary, F5 suggestive-untestable); hard `top1<0` gate design autopsy — precision≠prevalence (71% within-band conditional ≠ whole-split scope; free pre-launch prevalence read never taken), endogenous statistic, maximal blast radius per false positive; two-card synthesis localizing locus at consumption interface / count-bearing absolute magnitude (mhat sum pinned ≈18432 whether scene holds 40 or 2000); booked-vs-code line-by-line MATCH (diff-verified; dispatch fix correct); **4th DISPROVED disjunct vacuous in exactly the world that occurred** (conjunctive "rescue fails AND sum<0" is F1-only; sum component construction-true for any weights — extends H0022's mis-aimed-limb lesson: construction-true engagement reads belong in report-only R-reads); §11 call to extend register to substrate-swap-on-aggregate-sign family |

Consensus across all four: the card was tested **as booked** (qual/audit/diagnostic
agree; the pre-launch exclusive-dispatch bug was found and fixed, and w_c≠0 proves the
fix held); the mechanism **engaged** construction-true (R1a 100%, identity ≤0.04%);
the headline level is ≈half draw noise while every decisive read (0/11 under +18.8k,
R3 19-vs-11, dense +131 above retrain band, mid ≈42 with cellcal retained, sparse
worse-than-parent) is draw-independent — refutation rests on those reads, not the
level. Feedback disagreements resolved: (i) sparse FAIL as verdict-driver → footnote
only (standing parent miss; qual/diagnostic over quant), same ruling as siblings;
(ii) F-label confusion (dispatch parenthesized coarse-gate numbers under "F3") →
`idea.md:35` books F2=route-too-coarse / F3=KEEP-or-wipe-not-rescued; all three number
sets verified, no number depends on the label; (iii) qualitative/diagnostic's earlier
"artifacts absent" provenance caveats → **superseded**: all four dumps
(`result.json`, `val_result.json`, `val_perimage.json`, `val_attr.json`) are now local
and every child number was recomputed file-backed this session (causal §7 correction
table); hand-written confidence figures → discarded, CLI 0.4150 governs (§3).

## 3. Confidence check (ledger-reported, not recomputed)

`python3 scripts/discovery.py prove H0023` reports: **H0023 conf = 0.4150 (uncertain),
tests = 1** — after the single evidence event `contradicts w=0.85` (N0024_h0023,
2026-09-22T12:48:03); posterior ~Beta(1.00, 1.85), mode 0.0000. Eq.1 posterior staying
above the <0.25 refuted threshold after one event is the same situation as
H0021/H0022 (both also 0.415); REFUTED-class status here follows the verdict
semantics (futility refutation + all bars missed + decisive mechanism reads), while
the machinery books the posterior. Eq.1 math lives only in `src/cac/expt/` — no hand
recomputation performed. Calibration standings: H0024 0.500, tests=0 (booked, not
yet tested at this backfill).

## 4. Quality gate

### (a) Format gate — candidate texts vs §6 markers

Candidate successor text (H0024 CountAnchor) was checked for marker order
`IF IN THEN BECAUSE DISPROVED`: **PASS** — markers in order, numeric falsifier bound
to live parent 19.3431/330.81/5.45, plus two numeric engagement disjuncts (alpha CV
≥0.05; wipe-median alpha ≥1.3× KEEP-median).

### (b) K_SYNTH=2 cap — candidate bookings (novelty-checked)

| # | Candidate | novelty_check | Registered duplicate? | Action |
|---|---|---|---|---|
| A | `use_canchor` exemplar-box unit-mass anchor (zero-param per-image α = clamp(1/z, 0.25, 4) on the zero-init `out` residual; boxes fix absolute mass K) | exit 0, top sim 0.483 (H0023) < 0.82 | **Booked as H0024** at 2026-09-22T13:05:16 (`hypotheses.jsonl:45`), solo on N0015 → **N0025_h0024**, launched 13:28 (`journal/events.jsonl:63-64`); BECAUSE cites this two-card closure as motivation | **Already booked** before this backfill — not re-booked here |
| B | two-regime F2-successor / better route gate / unnormalized substrate / out-decoder surgery | — | **YES / dead**: mid 42.64 with cellcal retained kills F2-successor on its own pre-registered discriminator (family-caused, diagnostic §3.5/§5); better-gate and normalizer-swap re-encode H0023 → rule 11 needs new falsifier + `contradicts` on a +3.72/0/11 card; out/decoder dies at §11 final-layer + N0009/N0010 post-decoder bans | **NOT booked** |

**Bookings created by this synthesis: 0 of 2 allowed** (H0024 already occupies the
registered third-card slot; the F2/gate/substrate/decoder candidates are dead on
pre-registered reads or rule 11).

### (c) Opposite-of-existing rule

Applied: the surviving open forks (consumption interface; count-bearing magnitude)
are *different claims* about the locus the two-card closure localized, not opposite
re-encodes of H0023's mechanism. H0024's magnitude fork books as the registered
successor card citing this closure — not as a fresh opposite of H0023. Any true
re-encode of the aggregate-sign substrate-swap family books only as `contradicts`
evidence on H0023 with a genuinely new falsifier (§5.11 / rule 11); none was
allowed to mint a new id here.

### (d) Misattributed reasoning — remap or discard

- **"H0023 conf 0.15"-style hand recomputes → DISCARDED**; CLI reports 0.4150 (§3).
- **"Cellcal was supposed to stay active" as an H0022 claim → remapped already**
  (N0023 synthesis); for H0023 it is the booked phrase and the pre-launch fix honored
  it — do not re-litigate.
- **"Artifacts absent / lead-quoted only" provenance → DISCARDED** (superseded; dumps
  local, full recompute in quant/causal §7).
- **Sparse FAIL as verdict-driver → footnote** (parent-infeasible bar; child +0.16 is
  a real regression but not the deciding read).
- **AGENTS §6 "±0.02 precision" → DISCARDED** (voided by HANDOFF §3.1 noise band
  ±1–2 MAE); diagnostic decomposition governs.
- **"The swap wasn't really tested / quiet-null" → DISCARDED** (F1 falsified by
  construction; R1a 100%; diagnostic §2 fully-tested adjudication). Conversely
  **"anti-phase is the wipe's operative substrate" → DISCARDED** (qual Claim A causal
  half; H0023 itself).
- **0/11 wipe table → kept as the family consistency read** (two independent child
  entries: N0023 and N0024); idea.md booked the 11 via the measured set, R2 is the
  booked bar.
- **Vacuous 4th disjunct / idea.md:35 R1 restatement → noted, not carried as a
  mechanism falsifier** (qual §5.2; verdict carried by the three bars + report-only
  R-reads).

## 5. Calibration (verbatim `python3 scripts/discovery.py calibration`)

Run at this backfill (2026-09-22, after H0024's create event):

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)     21       2     10%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall         21       2     10%      -
reliability error (weighted |rate − pred_conf|): 0.405
WARNING: reliability error > 0.20 — confidence at test is drifting

=== Current Standings ===
hyp       confclass       tests
H0001    0.585uncertain       1
H0002    0.420uncertain       1
H0003    0.405uncertain       1
H0004    0.500uncertain       0
H0005    0.415uncertain       1
H0006    0.405uncertain       1
H0007    0.410uncertain       1
H0008    0.410uncertain       1
H0009    0.415uncertain       1
H0010    0.420uncertain       1
H0011    0.410uncertain       1
H0012    0.580uncertain       1
H0013    0.405uncertain       1
H0014    0.415uncertain       1
H0015    0.410uncertain       1
H0016    0.415uncertain       1
H0017    0.415uncertain       1
H0018    0.410uncertain       1
H0019    0.410uncertain       1
H0020    0.500uncertain       0
H0021    0.415uncertain       1
H0022    0.415uncertain       1
H0023    0.415uncertain       1
H0024    0.500uncertain       0
```

## 6. Decision-relevant summary for the Lead

1. **What this node closes — the substrate-swap / phase fork is dead.**
   H0022 (sign-mask, cannot touch negatives) + H0023 (substrate swap, *does* flip the
   sign construction-true) = two opposite operators, both 0/11 wipe-rescue with
   mechanism engaged, both as-booked futility refutations. Evidence **sign**, gain
   **sign-domain**, and **substrate identity at the `ev` interface** are all
   insufficient — the wipe's count deficit is not carried there. §11 register now
   covers the full neighborhood: gain-domain masks (H0018 form + H0022 domain,
   already written) **plus this substrate-swap-on-aggregate-sign family** (qual §4
   call; extend `AGENTS.md` §11 accordingly when the hygiene journal runs). Any
   re-encode books as `contradicts` on the existing id with a genuinely new falsifier
   (rule 11), else it dies on contact. Hard `top1<0` whole-image routing as a scope
   selector is also dead (41% fire / 63% KEEP; precision-within-band never licensed
   whole-split scope — qual §3 design lesson).
2. **What remains open — the OR-fork this corpse feeds.** The two-card closure
   localizes the dense wipe **below the evidence/gain layer**, at exactly one of:
   **(A) the consumption interface** `out→cond→decoder` cannot turn evidence into
   density mass under pixel-MSE (F3; H0016/H0017 quiet-null precedent; §11/N0009/
   N0010 make this a weak, likely-dead frontier — diagnostic does not recommend it as
   card3), or **(B) the evidence lacks count-bearing absolute magnitude** — cosine
   top1 ∈ [−1,1] and mhat mean-1 (sum pinned ≈18432 regardless of scene count) are
   both phase-correct-but-scale-free, so the decoder never sees per-image demand.
   Mid-slice ordering parent 37.50 < N0023 41.71 < N0024 42.64 (cellcal retained in
   N0024) kills the two-regime F2-successor as a booked direction.
3. **Budget accounting — N0024 is card 2 of 3; N0022 does NOT count.** HANDOFF §5 /
   `HANDOFF.md:64` grants three cards with paper closure only if all three miss a
   >1.5 net drop vs 19.3431. Composition resolved at `journal/events.jsonl:60`:
   (1) N0023/H0022 MarginCal **+2.82 worse** — spent, REFUTED; (2) **N0024/H0023
   use_antimatch +3.72 worse — this card, spent, REFUTED**; (3) the registered third
   slot, now booked as H0024 `use_canchor`. **N0022/H0021 `use_exkern` is
   pre-HANDOFF** (ran and was refuted 2026-09-21, before the 2026-09-22 HANDOFF that
   *created* the budget with exkern already known — `HANDOFF.md:48,64`) and charges
   **0 of 3**; retroactively charging it would outcome-shop the stop rule. Both spent
   cards miss >1.5 in the wrong direction; the paper trigger is **not yet armed** —
   it fires only after card 3 also fails to deliver a >1.5 net drop (or dies honestly
   at novelty/§11). Remaining budget: **exactly one card.**
4. **Successor pointer — H0024 / N0025_h0024 tests fork (b) = the magnitude fork.**
   Booked 2026-09-22T13:05:16 (`hypotheses.jsonl:45`), solo on N0015_h0014 → node
   **N0025_h0024**, Coding Agent wrote `model.py` + `config.toml` (4 call-sites +
   flag + SimPrior block; novelty exit 0, top H0023 0.483; local conformance OK),
   launched 13:28 under v2 / seed 20260830 / `futility_bar=19.0431`
   (`journal/events.jsonl:63-64`); **currently training**. Mechanism: zero-param
   stop-gradient exemplar-box unit-mass anchor — α = clamp(1/z, 0.25, 4) multiplies
   the confirmed SimPrior residual from the zero-init `out` projection, where z is
   the mean spatially-minmax-normalized top1 similarity inside the K annotation
   boxes (each box = exactly one object ⇒ absolute mass K), cellcal / substrate /
   every other parent line byte-identical. Its BECAUSE cites **this** two-card
   closure: cosine and mhat are scale-free; the boxes inject absolute unit mass so
   the decoder sees per-image demand. Engagement falsifiers are numeric (alpha CV
   <0.05 ⇒ no spread; wipe-median alpha <1.3× KEEP-median ⇒ miss locus) — the
   report-only R-read discipline both siblings' vacuous 4th disjuncts demanded.
   Exemplar-distinctness is kept as fallback only (journal:63), not booked.
5. **Paper narrative.** N0024 is the clean engaged-mechanism-wrong instance whose
   *success* was construction-certified (+18.8k, identity ≤0.04%) and whose failure
   is therefore a localization result, not a null: phase/sign/substrate at `ev` are
   closed; the open question is magnitude vs consumption; the run also contributed
   the sparse-worse-than-parent datum, the KEEP-27 AE concentration (68.3%), the
   dispatch-fix pre-launch audit trail, and the vacuous-disjunct booking lesson.
   Per diagnostic §5 the sequence is unchanged: ✅ N0023 miss → ✅ N0024 miss
   (this document) → **card 3 (N0025/H0024, in flight)** → if no >1.5 net drop,
   paper closure per `HANDOFF.md:64` (numbers val 19.3431 / test 19.066, collapse
   corpus, two-card wipe localization, futility machinery already in hand).

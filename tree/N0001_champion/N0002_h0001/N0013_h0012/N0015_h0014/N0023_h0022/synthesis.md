# Synthesis — N0023_h0022 (H0022 `use_margin` MarginCal) — REFUTED (futility-halted)

## 1. Consolidated verdict

Booked: H0022 `use_margin` — a sign-mask-selective per-cell energy gain amplifying only
positive-margin SimPrior evidence (`exp(w_m·log1p(mhat))`), negatives routed at identity,
**as a replacement for the sign-blind cellcal gain** (ledger create text, 2026-09-22T11:31:15),
triple bars vs live 19.3431/330.81/5.45 + R2 wipe-rescue ≥7/11. Ran: N0023_h0022, v2 seed
20260830, futility HALT at ep24 (best 22.1632), mechanism fully engaged (w_m=+0.2666, temp
pinned 0.07, cellcal bypassed w_c=0 — replacement semantics, as booked). Failed: val 22.163
(+2.82), dense 458.69 (+127.9), sparse 5.703 (improved vs parent 5.814 but over bar), mid
41.71 (+4.21), R2 0/11 — 0/4 bars. Why: the mask structurally cannot convert the net-negative
anti-phase field (ev_sum −4924, 66.7% neg cells) into mass — 0/11 was decided at code-write
time — while call-site-exclusive dispatch deleted cellcal's mass-floor + dense gradient-focusing
from **all** images, causing new dense/mid collapses (3425 0.90→0.13, 3436 1.13→0.09); of the
headline +2.82, ≈+1.2 is parent draw luck (draw-mean 20.544), the rest is mechanism-supported
by noise-invariant reads (dense outside [300,361] retrain band by +98; R2 0/11; mid +4.21;
late-slope divergence). Consensus: **mechanism-wrong**, as-booked dispatch, not a quiet null.
Feedback disagreements resolved: (i) "cellcal was supposed to stay active" — FALSE for H0022,
that runtime phrase belongs to H0023's booking (diagnostic §2 wins, ledger create text
authoritative); (ii) sparse is a footnote (child improved it; bar was a standing parent miss —
qual/diagnostic win over quant's FAIL row as verdict-driver); (iii) noise vs mechanism:
diagnostic's decomposition wins — noise ≈ first +1.2 only, structural findings draw-independent.

## 2. Distinct contributions of the 4 feedbacks

| Feedback | Unique load-bearing claims |
|---|---|
| quant | Full 0/4 bar verdict table with MAE cross-checks; all-24-epoch trajectory (monotone, ep16/ep24 gate lines verified in tb); AE decomposition: other-dense +45.2%, mid250–500 +25.8%, wipes +18.6%, residual mid +13.2%, sparse −2.7%; per-image R2 table 0/11 (mean ratio 0.241→0.184) |
| qual | BECAUSE split verdict: diagnosis premise held (pre-run dump), remedy decisively failed; mask is right-local/wrong-global operator (identity-routing keeps ev⁻ negative — magnitude-repair ≠ sign-conversion); three-stack mid-train-lead/late-loss explanation (early positive-gain headroom → contrast distortion on majority → one-scalar slope cap −0.156 vs −0.28); line-by-line booked-vs-code MATCH; 4th falsifier disjunct vacuous-by-construction; explicit rules-in/rules-out for H0023 |
| causal | ROOT: call-site exclusivity deletes cellcal gain from every forward (code-confirmed) — the mass-floor (+566 KEEP) and dense gradient-focusing loss; quantified attribution table: ~60% dense / ~26pp mid250–500 / ~13pp mid50–250 / sparse slightly better; static-math sign paradox (child pre-out evidence pointwise larger yet mass lower → damage runs through dispatch deletion + co-adaptation + truncation, not mask arithmetic); confound statement: run cannot separate sign-selectivity from cellcal-removal |
| diagnostic | Failure taxonomy: **mechanism-wrong** with noise decomposition (≈+1.20 parent luck, +1.62 child ≈1.4σ; HANDOFF ≥+2 heuristic); dispatch-vs-booking adjudication: H0022 fully tested as booked ("replacement" in ledger), untested counterfactual = two-regime F2-successor (fresh falsifier required); N0024 watch list (ep24 gate ≤20.00; w_c≈+0.15 dispatch check; R1/R2 arbitration before bars; draw-contaminated HALT band (20.0,21.6)); registered sequence: N0024 verdict → third card (exemplar-distinctness) → paper decision |

## 3. Confidence check (ledger-reported, not recomputed)

`python3 scripts/discovery.py prove H0022` and `calibration` standings both report:
**H0022 conf = 0.4150 (uncertain)**, tests=1 — after the single evidence event
`contradicts w=0.85` (N0023_h0022, 2026-09-22T12:21:33). Below the refuted threshold (<0.25)
is NOT met; one more contradicting event would move it further down but REFUTED-class status
here follows the verdict semantics (futility refutation + all bars missed), while the Eq.1
posterior stays 0.415 until the machinery books it. Eq.1 math lives only in `src/cac/expt/` —
no hand recomputation performed. H0023 stands at 0.500, tests=0 (N0024 TRAINING — not touched).

## 4. Quality gate

### (a) Format gate — proposed candidate texts vs §6 markers

Both candidates below were drafted and checked for marker order `IF IN THEN BECAUSE DISPROVED`:

- Candidate A (exemplar-distinctness): **PASS** — IF … IN … THEN … BECAUSE … DISPROVED IF, in order.
- Candidate B (two-regime F2-successor): **PASS** — IF … IN … THEN … BECAUSE … DISPROVED IF, in order.

(Neither is booked — see (b)/(d).)

### (b) K_SYNTH=2 cap — candidate bookings (novelty-checked)

| # | Candidate | novelty_check | Registered duplicate? | Action |
|---|---|---|---|---|
| A | `use_exdistinct` exemplar-distinctness / exemplar-encoding interface card | exit 0, top sim 0.452 (H0019) < 0.82 | **YES** — registered direction in `HANDOFF.md:48,62` + H0018/H0019 evidence notes + h0022 F1/F3 successors + diagnostic §5 recommendation ("after N0024 verdict") | **NOT booked now** — duplicates a registered direction, so it may only ever book against an existing id/evidence slot, and the registered order is N0024-verdict-first; booking a new id now would also preempt the HANDOFF 3-card budget accounting |
| B | `use_regimecal` two-regime F2-successor (blanket cellcal on non-anti-match, margin/repair only on anti-match minority) | exit 0, top sim 0.599 (H0023) < 0.82 | **YES** — the F2-successor registered in `local/research/h0022_idea_DRAFT.md:130` ("fresh falsifier — never a silent re-encode"); also the counterfactual diagnostic §2 says the run cannot separate | **NOT booked as a new id** — must be evidence/successor on the EXISTING H0022 registration; its worth is explicitly conditioned on N0024 read 3 (mid-slice ≈37.5 ⇒ dispatch-caused; ≈41 ⇒ family-caused, kills it) per diagnostic §4.3; no new evidence event to append — H0022's contradicting evidence is already booked |

**Bookings created: 0 of 2 allowed** (zero bookings is the justified outcome — both live
candidates are pre-registered directions whose booking slot and falsifier depend on N0024).

### (c) Opposite-of-existing rule

Applied: Candidate B is structurally the opposite of H0022's global replacement (retain cellcal
on the majority). Under §4(c) it would book as evidence against/relative to the existing id
(H0022), never as a fresh duplicate id — and since it has not been tested, there is no
contradicts/supports event to append. No opposite-of-existing text was allowed to mint a new id.

### (d) Misattributed reasoning — remap or discard

- **"cellcal was supposed to stay active / byte-identical" → DISCARDED from H0022, REMAPPED
  to H0023.** H0022's ledger booking says "as a replacement for the sign-blind cellcal gain";
  call-site exclusivity was the booked design (draft §2.2, idea.md:21 source-byte-identity of
  parent blocks only). The runtime-persistence phrase belongs to H0023's create text ("leaving
  the cellcal gain … byte-identical"), and N0024's pre-launch dispatch bug/fix is H0023's
  affair (journal 12:21:57). Any feedback line implying H0022 under-tested cellcal is wrong
  and is not carried forward.
- Discarded: quant's sparse FAIL row as a verdict-driver (parent never met 5.45 either —
  footnote per qual/diagnostic); AGENTS §6 "±0.02 precision" as an operative noise claim
  (voided by HANDOFF §3.1 — diagnostic decomposition governs).
- Remapped intact: F2 two-regime successor → H0022-registered successor with fresh-falsifier
  requirement; F1/F3 exemplar-side successors → the HANDOFF exemplar-distinctness slot.

## 5. Calibration (verbatim `python3 scripts/discovery.py calibration`)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)     20       2     10%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall         20       2     10%      -
reliability error (weighted |rate − pred_conf|): 0.400
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
H0023    0.500uncertain       0
```

## 6. Decision-relevant summary for the Lead

1. This node **closes** the gain-sign-domain family (mask/positive-only gain forms) and the
   strong form of "cellcal's sign-blind deepening is the wipe cause" — it does not open a new
   bookable direction of its own; both successor directions (two-regime F2, exemplar-distinctness)
   are already registered elsewhere.
2. **Book nothing new now** (0/K_SYNTH): both candidates are pre-registered duplicates whose
   falsifiers and booking slots hinge on N0024; diagnostic §5 explicitly sequences
   N0024-verdict → third card → paper.
3. N0024 must show, to matter: **R1** route fire ≥90% on the 11 wipes with routed post-gain
   evg_sum ≥0 (the phase correction H0022 structurally could not do), **w_c ≈ +0.15** (dispatch
   fix real; w_c≈0 invalidates the card), **R2 ≥7/11** rescue; bars (≤19.0431/dense<330.81/
   sparse≤5.45) are secondary — arbitration reads-first per diagnostic §3–4.
4. Watch: N0024 must be ≤20.00 at ep24 to finish; a HALT in (20.0, 21.6) is draw-contaminated,
   not a clean kill; if mid ≈41.71 with cellcal active, the intervention family itself is
   implicated and the two-regime F2-successor dies pre-booking.
5. If N0024 clears triple bars with R1/R2 true → live parent moves, stop-rule voided, re-plan;
   otherwise proceed to the exemplar-distinctness third card, then the HANDOFF "3 cards no >1.5
   → paper" decision.

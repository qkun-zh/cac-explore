# Synthesis — N0022_h0021 (H0021 `use_exkern` exemplar-kernel fusion) — REFUTED (futility-halted, quiet-null)

## 1. Consolidated verdict

Booked: H0021 `use_exkern` one-liner (ledger create `hypotheses.jsonl:39`, authored
`idea.md:7`) — an exemplar-kernel self-calibrated residual IN the live-N0015 head-only
child at seed 20260830 under v2, THEN val ≤19.0431 AND dense <330.81 AND sparse gt<50
≤5.45, BECAUSE full-size spatial RoI exemplar kernels convolved depthwise over
`fine.detach()` plus an ROI self-calibration constraint supply exemplar-structure match
evidence that pooled-vector similarity priors discard and restore the CountingDINO
training-free tail wins (measured: cdino beats live by 170–370 on the 8 catastrophic ids,
though it loses globally 39.70 vs 19.33) without learned suppression — a fully
non-parametric path fused by ONE zero-init scalar `w_x` (+1 param), solo switch
`use_exkern`. Design intent: keep the path unlearnable-by-construction so training cannot
re-learn suppression through it; F1 (`w_x ≈ 0` → quiet null → REFUTE) was pre-registered
before the run (`idea.md:46`).

Ran: N0022_h0021, v2 protocol, seed 20260830, launch `--set augment=true
--set futility_bar=19.0431` (journal:52). Futility HALT at ep24 (best 21.9284, need
0.367/ep vs gate, saved ~8 epochs; `result.json` futility_hit true, n_epochs_done 24/32).
Verdict on the futility-kept best.pth (`ckpt_epoch` 24):

| Bar / read (booked) | Threshold | Child (`val_result.json`) | Parent | Result |
|---|---|---|---|---|
| val EMA MAE | ≤19.0431 | **21.9275** (best 21.9284) | 19.3431 | **FAIL** (+2.58 vs parent, +2.89 vs bar) |
| dense gt>500 MAE | <330.81 | **454.4514** (n=17) | 330.81 | **FAIL** (+123.6) |
| sparse gt<50 MAE | ≤5.45 | **5.7913** (n=895) | 5.814 | **FAIL** vs bar; vs parent **flat** (−0.02) |
| mid 50–500 (read) | — | 40.8822 (n=374) | 37.50 | +3.38 |
| R2 tail transfer, ΣAE on the 8 cdino-win ids | < parent | 4728.7 | 4238.5 | **FAIL** (+490.2; 935 alone +579.1 → 0/8 transferred) |
| R1 mechanism `w_x` > 0 decisive (P1) | engaged positive | **−0.0398** | 0 (init) | **FAIL** — F1 quiet-null, never engaged positive |

Every booked bar failed; the run is a **pre-registered F1 quiet-null**, distinct from
sibling H0022/N0023's engaged-mechanism-wrong class (`w_m=+0.2666` there vs `w_x=−0.0398`
here). Ledger verdict already booked: `contradicts w=0.85` from N0022_h0021,
2026-09-21T21:53:15 (`hypotheses.jsonl:40`) — this file is the closing backfill.
**Budget note: N0022 ran 2026-09-21, before the 2026-09-22 HANDOFF — it does NOT count
toward the 3-card HANDOFF budget (see §6, explicit).**

## 2. Distinct contributions of the 4 feedbacks

| Feedback | Unique load-bearing claims |
|---|---|
| quant | Full bar-verdict table (0 of the booked bars passed: val/dense/sparse/R2/R1 all FAIL) with cross-checks (val_result 21.9275 exact from val_perimage; parent dump 19.3259 vs booked live 19.3431, Δ0.017 immaterial; child slices reproduce val_result exactly); slice deltas dense +123.64 / mid +3.38 / sparse −0.022 with n-matched 1286-pair dumps; full 24-epoch trajectory (monotone to 21.9284@ep24; late slope ep16→24 −0.170/ep vs required 0.367/ep — improving but 2.2–2.4× short, halt correct); **AE decomposition: dense gt>500 +62.8%, 250–500 +27.3%, 50–250 +10.5%, sparse −0.6% (only improving bucket)**; R2 eight-id transfer table (ΣAE +490.2, 5/8 marginally better, 935 +579.1 dominates); R1 `w_x=−0.0398` cited from ledger (no local checkpoint — `run/latest/` holds tb only) |
| qual | **BECAUSE never exercised — F1 pre-registered branch fired**: the refutation attaches to the delivery interface, not to the kernel features' value; honest design assessment (right vehicle for the booked claim *transfer-by-learned-fusion*, wrong instrument for the motivation's claim *are the features useful* — and `idea.md:51` priced this exact gap before launch); line-by-line booked-vs-code MATCH (detach chain `model.py:37,62,66`, zero-init scalar constructed last `:359,399`, broadcast ≡ booked expand `:248`, parent stack untouched, benign defensive deltas only); class contrast — quiet-null (H0021) vs engaged-mechanism-wrong (H0022) license opposite downstream readings; rules-out / honestly-leaves-open lists (distillation = different program; fixed-weight injection = different claim needing a new falsifier); sparse-limb footnote (5.45 unattainable-by-parent, child flat); §11 hygiene note flagging exemplar kernels for the don't-repeat register |
| causal | ROOT audited link-by-link: not a blocked gradient (structural ∂L/∂w_x live at `model.py:248`), not dead-at-init (no saturating nonlinearity), not an excluded parameter (in AdamW with wd 0.08) — the root is **zero expected reward**: `ch_cal` is a fully-detached, information-redundant re-encoding of `(fine, boxes)` the trunk already consumes, so ∂L/∂w_x is zero-mean noise that AdamW+wd parks at −0.0398 and the path sits inert (~2% scale at inference); **noise decomposition of the +2.59: structural only +0.34…+1.38 → ≥46% draw-contaminated** (same-seed draws {19.34, 20.70, 21.59}, mean 20.544 — booked parent is the lucky tail); dense blowup is level-coupled, not exkern-specific (lineage fit dense ≈ −526.3 + 44.82·val predicts 456.4 vs observed 454.45, residual −2.0); AE attribution dense +62.8% / mid +37.8% / sparse −0.6%; wipe phenotype carried mostly by **14 new ratio collapses on previously-healthy images (+48% of ΔAE)**, old 11-wipe set only +16.7%; mechanism verdict rests on the w_x read, not the level |
| diagnostic | Failure taxonomy: **quiet-null primary**, gate-too-strict / operator-wrong / draw-noise-as-sole-cause explicitly ruled out; structural reads noise cannot produce — dense +93.4 above the same-seed retrain band [300,361], **0/11 wipe ratios** (post-hoc family read; idea.md booked no wipe set), **0/8 cdino-win transfer targets**, mid +3.38; dispatch audit = **as-booked, no N0023-style exclusive-dispatch** (cellcal+peakcal active on all images); noise decomposition (+1.20 parent draw-luck, +1.38 child ≈1.2σ, combined ≈1.6σ HANDOFF-gray level gap); new N0024 constraints inherited from this corpse: quiet-null structurally inapplicable to H0023 (zero new params, no `w_x` escape hatch ⇒ mechanism reads decisive), 0/11 second entry, dense ≈455 at ep24 now a two-card pattern (new read 7 halt-epoch control), cdino-win-8 table as new read 8 (baseline 0/8, 935 +579), sparse stays non-decider; for any future external-evidence path: transfer-by-fusion settled dead, distillation is a loss-regime change requiring user authorization ("do not queue it autonomously"), feature quality exonerated / dynamics the killer; **sequencing recommendation: N0024 verdict → third registered card → paper**, with the card count stated explicitly (N0022 excluded) |

Consensus across all four: the card was tested **as booked** (qual/audit/diagnostic
agree), the mechanism never engaged (F1), the headline deltas are at least ~half draw
noise while the structural damage (dense over band, 0/11, 0/8, mid) is draw-independent —
refutation rests on P1 + those structural reads, not on the level. Feedback
disagreements resolved: (i) sparse as verdict-driver → footnote only (child flat, bar
unattainable-by-parent — qual/diagnostic over quant's FAIL row); (ii) "gradient ~0" in
the ledger note → overstatement (causal F: `w_x=−0.0398` proves gradient flowed, just
never toward a rewarded positive engagement; operational quiet-null stands);
(iii) diagnostic's hand-written "conf 0.15" → discarded, CLI reports 0.4150 (§3).

## 3. Confidence check (ledger-reported, not recomputed)

`python3 scripts/discovery.py prove H0021` reports: **H0021 conf = 0.4150 (uncertain),
tests = 1** — after the single evidence event `contradicts w=0.85` (N0022_h0021,
2026-09-21T21:53:15); posterior ~Beta(1.00, 1.85), mode 0.0000. The Eq.1 posterior
staying above the <0.25 refuted threshold after one event is the same situation as
H0022/H0023 (both also 0.415); REFUTED-class status here follows the verdict semantics
(futility refutation + all bars missed + pre-registered F1 read), while the machinery
books the posterior. Eq.1 math lives only in `src/cac/expt/` — no hand recomputation
performed; diagnostic.md's "conf 0.15" is not reproduced by the CLI and is discarded
(§4d).

## 4. Quality gate

### (a) Format gate — candidate texts vs §6 markers

Both candidates below were drafted and machine-checked for marker order
`IF IN THEN BECAUSE DISPROVED`:

- Candidate A (distillation-style supervision): **PASS** — markers in order, numeric
  falsifier bound to live parent 19.3431/330.81/5.45.
- Candidate B (structurally-fixed nonzero exkern weight): **PASS** — markers in order,
  same triple-bar falsifier.

(Neither is booked — see (b)/(d).)

### (b) K_SYNTH=2 cap — candidate bookings (novelty-checked)

| # | Candidate | novelty_check | Registered duplicate? | Action |
|---|---|---|---|---|
| A | `use_distill` external-target distillation auxiliary loss (CountingDINO/tail-target supervision so the detached evidence gets value-encoding gradients) | exit 0, top sim 0.495 (H0021) < 0.82 | **YES in kind** — ledger evidence note books it as "a different program" (`hypotheses.jsonl:40`); it is a **loss-regime change** against the standing invariant MSE(+w_cnt·L1) (`AGENTS.md` standing regime), i.e. user authorization, not a next card; diagnostic §3.2.2: "Do not queue it autonomously" | **NOT booked** — out of autonomous scope; would also re-open a family §11 just closed in its fusion form |
| B | `use_exkernfix` structurally-fixed nonzero exemplar-kernel residual weight (force the ROI-self-calibrated field open at a training-free constant instead of asking the loss to open the gate) | exit 0, top sim 0.601 (H0021) < 0.82 | **YES** — a re-encode of the refuted H0021 path: §5.11 requires `contradicts` evidence on H0021 plus a NEW falsifier, never a fresh silent retry (qual §4.2); qual §4.3 leaves it open only as a *different claim* with its own card | **NOT booked now** — (i) no new evidence event to append (H0021's contradicting event is already booked); (ii) the HANDOFF 3-card budget is fixed (journal:60): N0023/H0022 + N0024/H0023 + third card, and the third slot is already booked as **H0024 `use_canchor`** (ledger create 2026-09-22T13:05:16) whose BECAUSE cites the two-card closure — booking another id now would preempt that accounting; (iii) measured prior caution: `w_x` drifted *negative* |

**Bookings created: 0 of 2 allowed** (zero bookings is the justified outcome — candidate
A is a user-gated regime change, candidate B is a same-family re-encode whose only live
booking slot is successor-evidence on H0021, and the exploration budget is already spent
on the registered third card).

### (c) Opposite-of-existing rule

Applied: candidate B is structurally the forced-open opposite of H0021's measured
gate-closed outcome (and any claim that "the kernel features are useless/harmful" is the
opposite of qual's reading). Under §4(c) such opposites may only ever book as
`contradicts`/`supports` evidence on the **existing H0021 id**, never as a fresh
duplicate — and since neither has been tested, there is no event to append. No
opposite-of-existing text was allowed to mint a new id.

### (d) Misattributed reasoning — remap or discard

- **"H0021 conf 0.15" (diagnostic.md:10) → DISCARDED.** `discovery prove H0021` reports
  0.4150; the 0.15 figure is a hand recompute, and §6 forbids hand confidence math —
  the CLI value governs (§3 above).
- **"gradient ~0" (ledger note) → CORRECTED, not carried forward as written.** Causal
  link F: `w_x=−0.0398 ≠ 0` proves nonzero gradient mass reached the scalar; the
  accurate claim is "never a rewarded positive direction under joint MSE/L1"
  (diagnostic §2 wording hygiene). Quiet-null verdict unchanged.
- **Sparse FAIL row as a verdict-driver → footnote** (parent itself misses 5.45 at
  5.814; child flat −0.02) — same ruling as N0023's synthesis; dense + F1 arbitrate.
- **AGENTS §6 "±0.02 precision" as an operative noise claim → DISCARDED** (voided by
  HANDOFF §3.1: noise floor ±1–2 MAE) — diagnostic/causal decompositions govern, same
  discard as the N0023 synthesis.
- **"The kernel evidence was tested and is harmful/useless" → DISCARDED** (qual §3):
  quiet-null means the BECAUSE never got a nonzero dose; the refutation attaches to the
  transfer-by-fusion *delivery interface* only. Conversely, **"exkern wasn't really
  tested" → DISCARDED** (qual §4.4): F1 was pre-registered and read; no re-litigation of
  engagement.
- **0/11 wipe table → kept but labeled post-hoc**: idea.md booked no wipe set (quant
  §4); R2 for this card is the 8 cdino-win ids (0/8). The 0/11 read is the family's
  consistency read, now with two independent child entries (diagnostic §3.1.2).
- Remapped intact: N0023-style exclusive-dispatch confusion does **not** apply here —
  parent stack (cellcal+peakcal) confirmed active on all images (diagnostic §2, causal
  C1); §11 hygiene note from qual → already executed (journal:61, AGENTS §11 entries).

## 5. Calibration (verbatim `python3 scripts/discovery.py calibration`)

Run at this backfill (2026-09-22, after N0024's evidence event) — standings therefore
include ledger events appended after N0022's own verdict:

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

1. **This node closes the shared/exemplar-kernel + transfer-by-learned-fusion family.**
   Fusion of external / detached / non-parametric evidence into `cond_map` gated by any
   learned scalar or gate under the joint MSE(+w_cnt·L1) objective is a quiet-null design
   (measured: scalar parks ≈0/slightly-negative, transfer targets never move, structural
   dense damage still accrues). §11 hygiene is **already executed and verified**:
   `AGENTS.md` §11 carries "never: transfer-by-learned-fusion of
   external/non-parametric evidence into cond (H0021 exkern quiet-null…) — distillation
   would be a different program", journaled at `journal/events.jsonl:61`, alongside the
   existing bans (exemplar gating any granularity, shared projections, frozen
   hs(2,3)+condenser load-bearing). Any re-encode books as `contradicts` on H0021 with a
   NEW falsifier (§5.11), else it dies on contact. Feature quality is exonerated; the
   CountingDINO pipeline reproduction (training-free val 39.70, tail wins 170–370,
   `STATE.md:170–173`) stands as the measured motivation the delivery interface failed to
   carry.
2. **Book nothing new (0 of K_SYNTH=2).** Candidate A is a user-gated loss-regime
   change; candidate B is a same-family re-encode with no slot left in the budget; the
   exploration slot this node would have used is already occupied by the registered third
   card (H0024 `use_canchor`).
3. **How it feeds the two-card closure (H0022/H0023).** N0022's corpse supplied the
   closure's preconditions (diagnostic §3.1): the **second independent 0/11 wipe-table
   entry**, the **dense ≈455-at-ep24 two-card pattern** (→ new read 7 halt-epoch control
   before any mechanism claim on dense), the **cdino-win-8 transfer baseline 0/8 with
   935 +579** (→ new read 8), the **engagement-read precedent** (pre-registered P1/w_x —
   exactly the read diagnostic required N0024 to inherit via R1/`w_c` so a quiet-null
   could not pass as "waiting longer"), and the **class split** (quiet-null vs
   engaged-mechanism-wrong) that made N0024's mechanism reads, not its level, decisive
   (H0023 has zero new params — no escape hatch). With N0023/H0022 and N0024/H0023 both
   landing R2 0/11 across three intervention styles (additive external field, sign-masked
   gain, substrate swap), the wipe locus is localized **below the evidence/gain layer**
   (journal:62 two-card closure; H0023 ledger note) — and H0024's booked BECAUSE cites
   exactly that closure to motivate absolute-mass anchoring at the exemplar boxes.
4. **Paper narrative.** N0022 is one clean instance of each refutation class the paper
   pairs (quiet-null transfer vs engaged-mechanism-wrong), and its failure is itself
   paper-grade corpus: reproduced teacher tail wins + a delivery interface that joint
   training structurally will not reward + structural dense regression with a documented
   level-coupling fit. Per diagnostic §5 it does **not** substitute for the third card:
   paper closure triggers only after the 3-budget cards miss the >1.5 net drop
   (val ≤ 17.8431 vs 19.3431), unless a card installs a new live parent and voids the
   stop rule.
5. **EXPLICIT — does NOT count toward the 3-card HANDOFF budget.** N0022/H0021 is a
   **pre-HANDOFF card**: it ran and was refuted 2026-09-21, before the 2026-09-22
   HANDOFF, and is already folded into the history that *created* the budget —
   `HANDOFF.md:48` ("H0010–H0021 all REFUTED except cellcal" is precisely why HANDOFF then
   granted three fresh cards) and `HANDOFF.md:64` (the "3 cards no >1.5 → paper" rule).
   The three budgeted cards are (1) N0023/H0022 MarginCal, (2) N0024/H0023
   `use_antimatch`, (3) the registered third card, now booked as H0024 `use_canchor`
   (diagnostic §5 / journal:60 accounting). Retroactively charging exkern against that
   budget would shrink a budget granted with exkern already known — i.e. outcome-shopping
   the stop rule. **N0022 charges 0 of 3.**

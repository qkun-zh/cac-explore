# Synthesis — N0026_h0025 (H0025 `use_capfilm` CapFilm decoder-side FiLM) — REFUTED (futility HALT ep24, all three bars missed)

## 1. Consolidated verdict

Booked: H0025 `use_capfilm` — a ~25k-param zero-init per-channel FiLM conditioning of the
DensityDecoder: a `CapFilm` MLP maps `GAP(fine) ‖ log(ME)` (ME = mean_K(S²/(w_k·h_k)),
exemplar-box geometry) to `(γ, β)` of length 128, applied `h ← h·(1+γ) + β` on the decoder
**block output, pre-final-1×1-head, pre-softplus** — the consumption-interface / decoder
consumption fork of the two-card closure OR, opposite H0024's residual-scale arm. One-line
config delta `use_capfilm=true`; SimPrior/cellcal/peakcal/cond_map concat byte-identical;
step-0 identity by zero-init last Linear; CapFilm appended after every parent module (rule 13).
Triple bars bound to live parent 19.3431 / dense 330.81 / sparse 5.45, plus R1–R3 engagement
disjuncts (γ CV, capacity loading, S-curve movement, cond×2 leverage re-probe); launch carried
`--set futility_bar=19.0431` (ledger create 2026-09-22T13:54:44, `hypotheses.jsonl:46`).

Ran: N0026_h0025, v2 protocol, seed 20260830, green smoke (config_sha `4b56e8e656961e0c`,
model_sha `917596dd5b2df04e`, `journal:68`). **Futility HALT at ep24** — best 23.2320
(`result.json` best_epoch 24, n_epochs_done 24, futility_hit true, elapsed 1334.5s;
need 0.5236/ep ≫ threshold 0.12; ep24 ceiling 20.0031, best +3.23 above). Full eval on the
futility-kept best.pth (pulled this session — the four feedbacks were written against
missing artifacts; slices below are now file-backed):

| Bar / read | Threshold | Child | Parent | Result |
|---|---|---|---|---|
| val EMA MAE | ≤19.0431 | **23.223** (`val_result.json`, best@ep24) | 19.3431 | **FAIL (+3.88)** |
| dense gt>500 | <330.81 | **508.54** (n=17) | 330.81 | **FAIL (+177.7)** |
| sparse gt<50 | ≤5.45 | **6.104** (n=895) | 5.814 | **FAIL** (+0.29 vs parent; bar standing parent-infeasible — footnote) |
| mid 50–500 (context) | — | 42.13 (n=374) | 37.50 | +4.63 |
| test | — | **19.587** (`test_result.json`) | 19.066 | +0.52 |
| R1 γ engagement (CV, capacity loading) | CV≥0.05; gt≥300 ≥1.5× gt<50 | **not in dump** (γ never emitted by run) | — | **indeterminate** |
| Futility ep24 | need ≤0.12/ep | need **0.5236** | — | **HALT CONFIRMED** |

Triple conjunctive gate **0/3**; overall with R-reads **0/…** (R1–R3 undecidable without the
γ dump). Verdict: **REFUTED as booked** — futility refutation + all three bars missed, level
outside every observed draw (draws {19.3431, 20.697, 21.5928}, mean 20.544, sd≈1.13;
23.232 ≈ 2.4σ above mean, +1.64 above worst draw, above the card's own contaminated band
(20.0, 21.6) by +1.63). Failure sub-class **quiet-null vs engaged-harmful INDETERMINATE**
without the γ dump — diagnostic ranks engaged-harmful first from the level (generator-path
site, fine sens 15.2), quiet-null second (co-adaptation / loss-optional escape-from-zero);
neither is claimed as measured. Sibling corridor: N0023 +2.82, N0024 +3.72, N0025 +2.88,
**N0026 +3.88** — four ep24 HALTs, four loci (sign / substrate / residual-scale / decoder-FiLM),
one corridor; N0026 is the largest gap and the first decoder-consumption miss.
Feedback disagreements resolved: (i) quant's UNMEASURABLE slices — **superseded**: eval
dumps pulled 14:57 (`journal` hygiene), every slice number above recomputed from
`val_result.json`/`test_result.json` this session; (ii) sparse as verdict-driver → footnote
(standing parent-infeasible bar, same ruling as N0023/24/25); (iii) γ engagement claims →
neither PASS nor FAIL recorded (not in dump); verdict rests on val level + futility alone.
Ledger already carries `contradicts w=0.85` from N0026_h0025 at 2026-09-22T14:57:22
(`hypotheses.jsonl:47`); this file is the closing consolidation.

## 2. Distinct contributions of the 4 feedbacks

| Feedback | Unique load-bearing claims |
|---|---|
| quant | Triple-bar + futility table with exact cross-checks (ep24 ceiling 20.0031, need 0.5236, gaps +3.229/+3.889/+4.189); artifact inventory proving slices/γ/tb absent **at write time** (§6 pending-addendum discipline honored — addendum now folded into §1); draw arithmetic (2.4σ, above worst draw); R1–R4 all gated on missing dump; sibling endpoint corridor (N0023 22.163 / N0024 23.062 / N0025 22.221, all ep24 HALT); card arbitration band (20.0, 21.6) and verdict outside it |
| qual | BECAUSE split: diagnosis premise (leverage probe: fine=generator sens 15.2, cond suppressor) held, remedy failed; line-by-line booked-vs-code MATCH at the FiLM site (post-GN pre-head, zero-init, append-after-PeakCal, no dispatch bug — contrast N0023); three honest design weaknesses (loss-optional escape-from-zero F1 risk; ME is 1-D log capacity possibly collinear with GAP = scalar-family shadow; decoder co-adaptation over 24ep); four-locus sibling table; operational rule **ship the mechanism dump with the run** (γ histogram + capacity-loading split) — this card's biggest analysis gap, not the missing bars; rules-in: H0026 GemeCal already queued, exemplar-distinctness remains encoding-layer fallback |
| causal | Construction chain C1–C8 file-backed (FiLM site, CapFilm math, step-0 identity, call-site composition preserving SimPrior cascade, RNG append order, own Linears = no second grad path, nonzero init gradient); three ranked failure hypotheses (PLAUSIBLE engaged-harmful on the generator path / co-adaptation quiet-null / ME covariate inadequacy) each with what γ would distinguish; explicit two-arm closure OR statement: H0024 residual arm measured-negative AND H0025 decoder arm measured-negative at level — both arms file-backed, **similarity-side remains open and already booked**; single load-bearing causal statement (right locus by closure, loss-optional actuator, no booked gain) |
| diagnostic | Failure taxonomy: **futility-HALT level-refutation, sub-class indeterminate**; noise decomposition (+1.20 parent draw-luck / +2.69 child ≈2.4σ ≥ HANDOFF ≥+2 heuristic → real); classes ruled out (gate-too-strict, operator-wrong/dispatch, budget-timeout, draw-noise-as-sole-cause, rule-11 re-encode — CapFilm is a new locus); failure-mode matrix (only F5 fired; F1–F4 all gate on missing γ/slices); dispatch-vs-booking adjudication **fully tested for the level verdict** with honest mechanism-read residue; recommendation: do not silently retry CapFilm (rule 11: `contradicts` on H0025 with a genuinely new falsifier); sequence unchanged → H0026 GemeCal; γ-dump follow-up reads listed (inference-only, non-blocking) |

Consensus across all four: the card **executed what it booked** (qual/audit/diagnostic agree;
config delta is exactly one line; no N0023-style exclusive dispatch; smoke green); the level
misses by mechanism-level margin outside every draw; the mechanism engagement reads are
honestly indeterminate; the verdict as booked never depended on them. Feedback disagreements
resolved as in §1 (slices superseded by the local pull; sparse footnote; no invented γ
statistics; no hand-written confidences).

## 3. Confidence check (ledger-reported, not recomputed)

`python3 scripts/discovery.py prove H0025` reports: **H0025 conf = 0.4150 (uncertain),
tests = 1** — after the single evidence event `contradicts w=0.85` (N0026_h0025,
2026-09-22T14:57:22); posterior ~Beta(1.00, 1.85), mode 0.0000. Same situation as
H0021/H0022/H0023/H0024 (all 0.415 after one contradict): the Eq.1 posterior does not cross
the <0.25 refuted threshold after a single event; REFUTED-class status here follows the
verdict semantics (futility refutation + 0/3 bars + level outside draw band), while the
machinery books the posterior. Eq.1 math lives only in `src/cac/expt/` — **no hand
recomputation performed; no confidence numbers hand-written anywhere in this document.**
H0026 stands at 0.500, tests=0 (`N0027_h0026` launched 14:57 — in flight, not touched).
No H0027 exists in the ledger (`prove H0027` → no such hypothesis).

## 4. Quality gate

### (a) Format gate — candidate texts vs §6 markers

No new hypothesis text was drafted for booking at this node, so no format-gate PASS is
claimed for any candidate. (If a successor is ever proposed, it must carry markers
`IF IN THEN BECAUSE DISPROVED` in order with a numeric falsifier bound to the live parent —
as every sibling booking did.)

### (b) K_SYNTH=2 cap — candidate bookings (novelty-checked)

| # | Candidate | Status | Action |
|---|---|---|---|
| A | Decoder-amplitude CapFilm successor (re-tune γ, re-place FiLM, per-image scalar SNDM literal, cond-FiLM option C) | **rule-11 dead**: re-encodes H0025 (`contradicts` on H0025 with a genuinely new falsifier required — none drafted); FiLM family twin of the card just refuted; cond path is the 10×-weaker suppressor | **NOT booked** |
| B | Similarity-side geometry scale | **already in flight**: H0026 GemeCal booked 14:45 via `discovery hypo --new --solo` → `N0027_h0026`, launched 14:57 (journal:71-72) — covers the only open cell of the trilogy | **NOT re-booked** |
| C | Exemplar-distinctness (encoding layer) | **pre-registered fallback** in HANDOFF + N0023/N0024 synthesis sequencing ("after the in-flight verdict"); expected-negative per HANDOFF §3.5; booking a new id now would preempt the registered order (N0027 verdict first) | **NOT booked now** |
| D | Residual-scale / response-z, gain masks, substrate swap, transfer-by-learned-fusion, evidence-layer content edits | **all closed** (H0024, H0018/H0022, H0023, H0021, H0016/17 + §11 register) | **NOT bookable** |

**Bookings created: 0 of 2 allowed** — the direction is exhausted at this node's locus
(decoder CapFilm site closed; rule 11 blocks silent retries); the one open cell
(similarity-side) is already occupied by H0026 in flight; the last weak untested family
(exemplar-distinctness) is a pre-registered fallback that must wait for the N0027 verdict
rather than mint a duplicate id now. Zero bookings is the justified outcome.

### (c) Opposite-of-existing rule

Applied: no opposite-of-existing text was allowed to mint a new id. The natural opposite of
H0025's decoder-amplitude claim ("decoder consumption can carry per-image capacity") would be
another decoder site edit — which is either a rule-11 re-encode (books as `contradicts` on
H0025 with a new falsifier, none available) or a §11/quiet-null-family repeat. H0026's
similarity-side claim is a *different claim about a different site*, already booked as its
own id before this synthesis.

### (d) Misattributed reasoning — remap or discard

- **"Slices/γ/result.json missing" as a standing claim → DISCARDED as current status**
  (superseded by the 14:57 local pull; quant's pending-addendum §6 discharged in §1). The
  feedbacks' honesty at write time is kept as process credit, not as artifact state.
- **Hand-computed confidences → DISCARDED**; CLI `prove H0025` = 0.4150 governs (§3).
- **AGENTS §6 "±0.02 precision" as an operative noise claim → DISCARDED** (voided by
  HANDOFF §3.1 noise band ±1–2 MAE; same-seed floor now framed ±2.25 by the standing
  precision-crisis note); diagnostic's draw decomposition governs.
- **Sparse 6.104 FAIL as verdict-driver → footnote** (parent 5.814 also misses 5.45;
  child +0.29 is a real regression but not the deciding read — same sibling ruling).
- **"Quiet-null confirmed" or "engaged-harmful confirmed" → DISCARDED** (γ not in dump;
  diagnostic's indeterminate taxonomy with a consistency ranking is the carried form).
- **"Decoder consumption interface structurally cannot work" → DISCARDED as overclaim**
  (one card, no slices at feedback time now discharged, no γ read — causal §5: *weakened,
  not closed*; only this exact CapFilm booking is closed).

## 5. Calibration (verbatim `python3 scripts/discovery.py calibration`)

Run at this backfill (2026-09-22, after H0025's evidence event):

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)     23       2      9%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall         23       2      9%      -
reliability error (weighted |rate − pred_conf|): 0.413
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
H0024    0.415uncertain       1
H0025    0.415uncertain       1
H0026    0.500uncertain       0
```

Note: reliability error 0.413 > 0.20 — confidence at test is drifting (23 tested, 2 confirmed,
9% vs pred 0.50). Recorded, not corrected: any η/threshold change is a §10 mechanism edit and
is not performed here.

## 6. Decision-relevant summary for the Lead

1. **What this node closes — the decoder CapFilm site.** H0025 as booked (zero-init per-channel
   FiLM from GAP‖log-ME at the block output pre-head) is REFUTED: 0/3 bars, futility HALT,
   level outside every draw. Rule 11 binding: any future decoder-amplitude card books as
   `contradicts`-evidence on H0025 with a genuinely new falsifier — re-tuning γ, moving the
   FiLM site, or SNDM-literal output β all die on contact. The consumption fork is
   **weakened by one probe, not structurally closed** (no γ read); do not write it into §11
   as a family ban on this evidence alone.
2. **Book nothing new (0/K_SYNTH).** Decoder site closed; similarity-side already in flight
   as H0026 GemeCal (`N0027_h0026`, launched 14:57); exemplar-distinctness is the registered
   encoding-layer fallback and must wait for the N0027 verdict (pre-registered order in
   HANDOFF + sibling syntheses). No candidate passed format gate + novelty + non-duplication
   simultaneously, so zero bookings is the correct output, not an omission.
3. **Trilogy status after this node:** residual post-out (H0024) REFUTED, decoder consumption
   (H0025) REFUTED, **similarity-side (H0026) RUNNING**. If N0027 also misses the triple bars,
   all three sites of the closure OR are measured-negative — that is the paper-closure
   configuration under HANDOFF §5 (numbers + wipe localization + S-curve diagnosis + collapse
   corpus already in hand), with exemplar-distinctness as the one remaining weak card if the
   user keeps the GPU open.
4. **Operational rules carried forward:** (a) ship mechanism dumps (γ/α/f histograms +
   capacity-loading splits) with every run — two consecutive cards lost their R1 read;
   (b) pull val/test dumps before writing quant feedback (this node's four feedbacks were
   written artifact-blind); (c) precision floor ±2.25 same-seed — any future claim under
   ~+1.5 net is noise-class; (d) do not hand-edit `info.json` (`status: proposed` is a sync
   gap, journal is authoritative).
5. **Paper narrative contribution:** N0026 is the clean "right locus, wrong-or-quiet actuator"
   instance — booked-vs-code MATCH, generator-path siting justified by the leverage probe,
   still 0/3 — completing the four-locus corridor (+2.82 / +3.72 / +2.88 / +3.88) that shows
   micro-interventions at evidence, residual, and decoder sites cannot overcome the global
   S-curve under frozen-backbone + MSE. γ-dump follow-up (diagnostic §5) remains open and is
   inference-only.

---

*Files written: this document only — `synthesis.md`. No commit, no ledger/`info.json` edits,
no hand-written confidence numbers. Calibration run via `python3 scripts/discovery.py
calibration` (exit 0), bin table pasted verbatim in §5.*

# Synthesis — N0027_h0026 (H0026 `use_geme` GemeCal geometry ME scale on SimPrior ev) — REFUTED (futility HALT ep24, all three bars missed)

## 1. Consolidated verdict

Booked: H0026 `use_geme` — a +1-param zero-init geometry magnitude embedding scale: per-image
`f = clamp((me/32)^w_g, 0.25, 4)` from annotation exemplar-box areas (`me = mean_K(S²/(w_k·h_k))`,
CACViT ME, stop-grad geometry), multiplying the SimPrior evidence `ev = cat[top1, cons]` **after**
the unchanged cellcal/peakcal gains and **before** the unchanged zero-init `out` projection,
threaded through all four `simprior(...)` call sites; one boolean `use_geme`, single-line config
delta; step-0 double identity (`w_g=0 ⇒ f=1`, and `out` zero-init). Triple conjunctive bars bound
to live parent 19.3431 / dense 330.81 / sparse 5.45 (`idea.md:7`), plus R0 (corr information
gate), R1 (CV(f) engagement), R2 (attenuate-on-dense loading) mechanism disjuncts
(`idea.md:156–159`); launch carried `--set futility_bar=19.0431` (ledger create
2026-09-22T14:45:17, `hypotheses.jsonl:48`).

Ran: N0027_h0026, v2 protocol, seed 20260830, green smoke (config_sha `c48cbec8d36450f4`,
model_sha `f8b0470a4542fd92`, journal launch record). **Futility HALT at ep24** — best 22.0878
(`result.json`: best_epoch 24, n_epochs_done 24/32, futility_hit true, budget_hit false,
elapsed_s 1316.5; ep24 ceiling 20.0031, gate-time need **0.3975/ep ≫ 0.12** (runner formula
`(best−bar)/(32−ep)`, `runner.py:183`; pre-update best 22.2231), best **+2.085 above ceiling**;
post-update convention from final best 22.0878 gives **0.3806/ep** — HALT under either
ordering).
Full eval on the futility-kept best.pth (ckpt_epoch 24):

| Bar / read | Threshold | Child | Parent | Result |
|---|---|---|---|---|
| val EMA MAE | ≤19.0431 | **22.073** (`val_result.json`, n=1286) | 19.3431 | **FAIL (+3.03 vs bar; +2.73 vs parent)** |
| dense gt>500 | <330.81 | **445.05** (n=17) | 330.81 | **FAIL (+114.24)** |
| sparse gt<50 | ≤5.45 | **5.876** (n=895) | 5.814 | **FAIL** (+0.062 vs parent; bar standing parent-infeasible — footnote) |
| mid 50–500 (context) | — | 41.61 (n=374) | 37.50 | +4.11 |
| test | — | **19.804** (`test_result.json`, n=1190) | 19.066 | +0.74 (corroborating only) |
| R0 corr(log me, log(1+gt)) | ≥0.20 | **not in dump** | — | **indeterminate** |
| R1 CV(f), w_g, clamp hist | ≥0.05 | **not in dump** | — | **indeterminate** |
| R2 mean f\|gt≥300 ≤0.90× gt<50 | ≤0.90× | **not in dump** | — | **indeterminate** |
| Futility ep16 / ep24 | WARN / need ≤0.12 | WARN @**24.0491** (need 0.3129); ep24 need **0.3975** gate / **0.3806** from final best | — | **HALT CONFIRMED** |

Triple conjunctive gate **0/3**; R0–R2 undecidable without the `w_g`/`f`/corr dump (never emitted
by the run — no local `run/` tb either). Verdict: **REFUTED as booked** — futility refutation +
all three bars missed, level **outside every observed draw** (same-seed draws {19.3431, 20.697,
21.5928}, mean 20.544, sd≈1.13; 22.073 is **+0.48 above worst draw**, ≈1.35σ above mean) and
**outside the card's own draw-contaminated arbitration band (20.0, 21.6) by +0.49**
(`idea.md:176`) — so the pre-registered R0/R1/R2 tiebreakers were never needed for the verdict;
they remain the F1-vs-F2-vs-F3 localizers and are honestly **not in dump**. Failure sub-class
**quiet-null (F1) vs dead-covariate (F2) vs wrong-direction (F3) INDETERMINATE** — none is
claimed. Corridor: N0023 +2.82, N0024 +3.72, N0025 +2.88, N0026 +3.89, **N0027 +2.73** — five
ep24 HALTs, five loci (sign / substrate / residual-scale / decoder-FiLM / similarity-amplitude),
one corridor; N0027 is the **best of the corridor** (+2.73, dense blowup smallest at +114) yet
still 0/3. **The three-site amplitude trilogy is now complete and 0/3**: similarity-side
(this card, H0026), decoder consumption (H0025, N0026), post-out residual (H0024, N0025) — all
futility-HALT ep24 in the +2.7…+3.9 envelope. Feedback consensus: all four say REFUTED via
futility-HALT on the measurable triple-bar line; booked-vs-code MATCH (one-line delta, four
call sites, no N0023 dispatch bug, append-after-PeakCal, step-0 identity); disagreements resolve
toward diagnostic/qual — verdict rests on R3 ∧ futility, sparse is footnote, mechanism reads
stay indeterminate. Ledger already carries `contradicts w=0.85` (15:28:56) + `neutral` slice
completion (16:33:59) from N0027_h0026 (`hypotheses.jsonl:50–51`); this file is the closing
consolidation. Successor H0027 `use_finescale` already booked (16:38:55) → **N0028_h0027** —
no new booking made here.

## 2. Distinct contributions of the 4 feedbacks

| Feedback | Unique load-bearing claims |
|---|---|
| quant | Artifact inventory proving `w_g`/f/corr/tb absent; full triple-bar table with parent-dump cross-checks (22.0730/445.0527/5.8755 vs live 19.3431/330.8131/5.8136); exact futility arithmetic (ep16 need 0.3129 → WARN; ep24 need 0.3975, ceiling 20.0031, +2.085 gap; post-update 0.3806 — HALT either ordering); R0–R3 row-by-row with **not in dump** honesty for R0–R2; draw arithmetic (+0.48 above worst, outside band by +0.49); five-node sibling corridor table (N0023–N0027) with dense deltas |
| qual | Two-joint BECAUSE reading (diagnosis: scale-free similarity pathway + S-curve + net-suppressive residual; remedy: geometry ME power law); line-by-line booked-vs-code **MATCH** (clamp [0.25,4], post-gain pre-out placement, four call sites, GemeCal after PeakCal — N0023 bug did NOT recur); honest design assessment: **cleanest of the trilogy** (published CACViT covariate, +1 param, geometry-only, learnable sign) yet same failure envelope — actuator still multiplies the weak-suppressive residual (remove → +29.9%, sens 10× below fine); sparse moved wrong-way-for-R2-story (+0.062); rules-out (a) geometry ME-on-ev as booked, (b) the entire amplitude trilogy as *primary* mass actuator (five loci, 0/3), (c) "ran out of epochs" (HALT outside band); rules-in: train-time count-conditional actuator on generator path (STATE:195 oracle band), R0 still worth measuring once, rule-11 `contradicts` on H0026 for any ME-on-similarity re-encode, **ship mechanism dumps** |
| causal | Construction chain C1–C8 file-backed (four call sites, post-gain pre-out multiply, stop-grad geometry, +1 param zero-RNG, temp-pin intact, nonzero init gradient, one-line config); measured-rows table with R0–R2 **unmeasurable** and R3 **all FAIL**; residual structural contrast (f moves only pre-`out` amplitude of a net-suppressive term — cannot touch `fine`); leverage-probe §3 reading (right covariate, wrong-lever path — same structural handicap as H0024/H0025 with a better covariate); causal chain A–E with B blocked on missing dump; single load-bearing statement for the paper |
| diagnostic | Failure taxonomy **futility-HALT level-refutation, sub-class indeterminate**; noise decomposition (+1.20 parent draw-luck / +1.53 child ≈1.35σ — smallest of corridor but real); noise-invariant structure (dense +84 past retrain-band edge 361, mid +4.11 family ~42 plateau); classes ruled out (gate-too-strict — parent-slope extrapolation still +0.81 above bar; operator-wrong/dispatch; budget-timeout; draw-noise-as-sole-cause); F1–F6 failure-mode matrix (**only F6 fully fired**; F1–F5 all gate on missing dump); dispatch-vs-booking adjudication **fully tested for the level verdict** with honest mechanism-read residue; three ranked root-cause hypotheses each with what the `w_g`/f dump would distinguish; recommendation: do not silently retry geometry-ME-on-similarity (rule 11: `contradicts` on H0026 with a genuinely new falsifier); next-card direction unchanged → count-conditional train-time actuator on generator (STATE:195) or already-booked finescale |

Consensus across all four: the card **executed what it booked** (qual/audit/diagnostic agree;
config delta is exactly `use_geme=true`; no dispatch bug; smoke green; shas in `result.json`);
the level misses by mechanism-level margin outside every draw and outside the card's own
arbitration band; the mechanism engagement reads (R0/R1/R2) are honestly indeterminate; the
verdict as booked never depended on them. Feedback disagreements resolved as in §1 (sparse
footnote; no invented `w_g`/f/corr statistics; no hand-written confidences; F1/F2/F3 stay open
until the dump is pulled).

## 3. Confidence check (ledger-reported, not recomputed)

`python3 scripts/discovery.py prove H0026` reports: **H0026 conf = 0.4150 (uncertain),
tests = 2** — after `contradicts w=0.85` (15:28:56) + `neutral w=0.00` slice completion
(16:33:59), both from N0027_h0026; posterior ~Beta(1.00, 1.85), mode 0.0000. Same situation as
H0021–H0025 (all 0.415 after one contradict): the Eq.1 posterior does not cross the <0.25
refuted threshold after these events; REFUTED-class status here follows the verdict semantics
(futility refutation + 0/3 bars + level outside draw band), while the machinery books the
posterior. Eq.1 math lives only in `src/cac/expt/` — **no hand recomputation performed; no
confidence numbers hand-written anywhere in this document.** `prove H0027` = **0.5000
(uncertain), tests=0** (booked 16:38:55 → N0028_h0027, not yet run — untouched by this
synthesis).

## 4. Quality gate

### (a) Format gate — candidate texts vs §6 markers

No new hypothesis text was drafted for booking at this node, so no format-gate PASS is claimed
for any candidate. (If a successor is ever proposed, it must carry markers
`IF IN THEN BECAUSE DISPROVED` in order with a numeric falsifier bound to the live parent —
as every sibling booking did.)

### (b) K_SYNTH=2 cap — candidate bookings (novelty-checked)

| # | Candidate | Status | Action |
|---|---|---|---|
| A | Fine-scale / geometry ME on the dominant `fine` generator branch (`use_finescale`) | **already booked**: H0027 created 16:38:55 via `discovery hypo --new` → **N0028_h0027** (journal booking record; novelty exit 0, top H0026=0.638) — the legal successor moving the covariate onto the −91.6%-leverage path | **NOT re-booked** |
| B | Pre-softmax ME logit offset (`use_me_pre`) | **surveyed NO-GO** (journal 16:33:59; `local/research/h0028_pre_idea_DRAFT.md`): CACViT's published actuator is post-softmax; per-image-scalar pre-softmax multiply is **bit-equivalent by commutativity** → rule-11 re-encode of H0026; draft states NOT recommended | **NOT booked** |
| C | Geometry-ME / any-source amplitude re-encode on SimPrior `ev`, decoder consumption, or post-out residual (the trilogy) | **rule-11 dead**: three measured-negative loci (H0024, H0025, H0026), all 0/3 + ep24 HALT; any re-encode books as `contradicts` on the existing id with a genuinely new falsifier — none drafted | **NOT booked** |
| D | Exemplar-distinctness (encoding layer), gain-domain masks, substrate swap, transfer-by-learned-fusion, evidence content edits, global/regime calibration | §11 / standing bans / registered fallback (H0019 failed; H0018/H0022 wipes; H0023; H0021; H0010/H0015/H0002) | **NOT bookable** |

**Bookings created: 0 of 2 allowed** — the successor with the higher-leverage substrate
(finescale on `fine`) is **already in the ledger and the tree as N0028_h0027**; the
pre-softmax alternative was surveyed and rejected as a commutativity re-encode; the amplitude
trilogy is closed at the booked bars; remaining weak families are §11/registered. Zero bookings
is the justified outcome, not an omission (user directive: prefer 0 given H0027 already booked
as N0028).

### (c) Opposite-of-existing rule

Applied: no opposite-of-existing text minted a new id. The natural opposite of H0026 ("geometry
ME amplitude on the similarity residual helps") is either (i) the already-appended
`contradicts w=0.85` evidence event on H0026 itself, or (ii) a *different site* claim —
finescale on `fine` — which is a different mechanism and already has its own id (H0027). No
duplicate created.

### (d) Misattributed reasoning — remap or discard

- **Any claim that R0/R1/R2 fired or did not fire → DISCARDED as unmeasurable** (no `w_g`,
  no per-image `f`, no corr field in any local artifact). F1 quiet-null vs F2 dead-covariate vs
  F3 wrong-direction remains **indeterminate**; diagnostic's consistency ranking (#1
  under-powered-on-suppressive-residual, #2 quiet-null+draw-tail, #3 dead covariate) is carried
  as PLAUSIBLE, not measured.
- **Sparse 5.876 FAIL as verdict-driver → footnote** (parent 5.814 also misses the standing
  5.45 bar; child +0.062 essentially flat — same sibling ruling; dense + val decided).
- **`info.json` `status: "proposed"` / `tested_hypotheses: []` as "never ran" → DISCARDED**:
  server sync gap (rule 3: no hand-edit); `result.json` on disk records the completed
  futility-HALTED run; journal is authoritative.
- **Hand-computed confidences → DISCARDED**; CLI `prove H0026` = 0.4150 governs (§3).
- **"N0027 ran out of epochs / gate too strict" → DISCARDED**: need 0.3975/ep ≫ 0.12; best sits
  +0.49 above the card's own contamination ceiling 21.6; extrapolating the *parent's* late slope
  from 22.0878 still ends ≈19.85 @ep32, +0.81 above the 19.0431 bar — HALT saved epochs without
  changing the verdict (diagnostic §1c).
- **A specific failure mode (F1/F2/F3/F4/F5) as settled → DISCARDED** (dump absent); only F6
  late-crossover→futility-HALT is directly readable and fully fired.
- **"Amplitude path structurally impossible" → DISCARDED as overclaim** (five cards weaken it
  heavily; causal §3: right covariate wrong-lever is PLAUSIBLE not proven — no actuator read on
  any of the three trilogy cards). Carried form: empirically inert-to-harmful as *primary* mass
  actuator under joint MSE at these five loci.
- Remapped intact: leverage-probe premises (fine sens 15.2, remove residual +29.9%) → recorded
  file facts from `N0026_h0025/idea.md` + `journal/events.jsonl:65`, not re-ablated on N0027;
  `w_g`/f/corr read list → diagnostic §5 (post-hoc, non-blocking).

## 5. Calibration (verbatim `python3 scripts/discovery.py calibration`)

Run at this backfill (2026-09-22, after H0026's two evidence events; N0028 already booked):

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)     24       2      8%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall         24       2      8%      -
reliability error (weighted |rate − pred_conf|): 0.417
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
H0026    0.415uncertain       2
H0027    0.500uncertain       0
```

Note: reliability error 0.417 > 0.20 — confidence at test is drifting (24 tested, 2 confirmed,
8% vs pred 0.50). Recorded, not corrected: any η/threshold change is a §10 mechanism edit and
is not performed here.

## 6. Decision-relevant summary for the Lead

1. **What this node closes — the similarity-side pre-out amplitude site (and the trilogy).**
   H0026 as booked (geometry ME power law on SimPrior `ev`, post-gain pre-`out`, learnable sign)
   is REFUTED: 0/3 bars, futility HALT ep24, level outside every draw and outside the card's own
   band. With H0024 (post-out residual) and H0025 (decoder FiLM) already dead, **all three
   sites of the amplitude localization trilogy are measured-negative** (+2.88 / +3.89 / +2.73),
   inside one five-card corridor (N0023…N0027, +2.7…+3.9) spanning sign / substrate /
   residual-scale / decoder / similarity loci. Rule 11 binding: any future ME-on-similarity or
   amplitude-of-suppressive-residual card books as `contradicts`-evidence on H0026 with a
   genuinely new falsifier — re-tuning the clamp, re-placing `f`, or swapping the covariate
   source alone dies on contact. Do **not** write "similarity-side dead" into §11 as a family
   ban from this evidence alone (no `w_g`/f read — weakened five ways, not structurally closed).
2. **Book nothing new (0/K_SYNTH).** The finescale successor (`use_finescale` on `fine`,
   the −91.6%-leverage generator path) is **already booked as H0027 → N0028_h0027** and is the
   in-flight next card; pre-softmax ME was surveyed NO-GO (commutativity re-encode); the
   trilogy and §11 families are closed. Zero bookings is correct.
3. **Trilogy complete → next-card direction.** Per STATE:195 and the causal/diagnostic
   recommendation: the path to val <18 needs a **train-time count-conditional actuator on the
   generator side** (offline gt-binned scale oracle −2.3…−7.5), not another scalar on
   evidence/decoder output — which is exactly what N0028_h0027 (finescale) tests. If N0028 also
   misses, the amplitude/covariate program on both branches is bracketed and the HANDOFF
   "N cards no >1.5 → paper" gate applies; exemplar-distinctness remains the registered weak
   fallback only if the user keeps the GPU open.
4. **Operational rules carried forward (third card in a row):** (a) **ship mechanism dumps
   (`w_g`/`w_f`/γ/α scalars + f/γ histograms + corr + capacity-loading splits) with every run**
   — H0024, H0025, H0026 all lost their R1-class read to a missing dump; R0 (`corr(log me, gt)`)
   is still worth one eval-time read for the paper-grade dataset fact, independent of any card's
   verdict; (b) pull val/test dumps before writing quant feedback (N0026's feedbacks were
   artifact-blind — this node's were written after the pull); (c) precision/noise framing:
   sd≈1.13 draws, any claim under ~+1.5 net is noise-class, level outside worst-draw + band is
   the refutation line; (d) do not hand-edit `info.json` (`status: proposed` is a sync gap,
   journal + `result.json` are authoritative).
5. **Paper narrative contribution:** N0027 is the clean "best-of-corridor, right published
   covariate, wrong-lever path, still 0/3" instance — booked-vs-code MATCH, +1 param, geometry-
   only, novelty-green, falsifier family mutually unsatisfiable vs H0024/H0025 — completing the
   localization table that shows amplitude interventions at evidence, residual, and decoder
   sites cannot move integrated mass under frozen-backbone + joint MSE, because the leverage
   lives in `fine`. The `w_g`/f/corr reads (diagnostic §5) remain open and inference-only; they
   do not block N0028 or the verdict.

---

*Files written: this document only — `synthesis.md`. No commit, no ledger/`info.json`/journal
edits, no hand-written confidence numbers. Calibration run via `python3 scripts/discovery.py
calibration` (exit 0), bin table pasted verbatim in §5. Evidence for H0026 was already appended
to `memory/hypotheses.jsonl` before this synthesis (contradicts + neutral); this role did not
append or edit it.*

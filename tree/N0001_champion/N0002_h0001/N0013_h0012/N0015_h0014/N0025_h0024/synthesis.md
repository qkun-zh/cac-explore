# Synthesis — N0025_h0024 (H0024 `use_canchor` CountAnchor) — REFUTED (futility-halted)

## 1. Consolidated verdict

Booked: H0024 `use_canchor` — zero-param stop-gradient exemplar-box unit-mass anchor
`alpha = clamp(1/z, 0.25, 4)` (z = ROI-align mean of minmax-normalized detached top1 over K=3
annotation boxes) multiplying the post-`out` SimPrior residual, cellcal/peakcal/substrate
byte-identical, **triple bars vs live 19.3431 / 330.81 / 5.45** (`idea.md:7,124,166`). Ran:
N0025_h0024, v2 seed 20260830, futility **HALT at ep24** (best 22.2205, need 0.3972/ep vs 0.12
threshold; ceiling 20.0031, +2.2174 above; outside the card's own draw-contaminated band
(20.0, 21.6)). Failed: val **22.2127** (+2.88 vs 19.3431, bar ≤19.0431 missed by +3.18), dense
**454.05** (+123.2 vs 330.81, band upper edge 361 exceeded by +93), mid **41.91** (+4.41 vs
37.50), sparse **5.779** (improved −0.035 vs 5.814, still over the standing parent-infeasible
5.45 bar), test **19.437** (parent test 19.066) — **0/3 bars**. Why: the magnitude fork of the
two-card closure OR landed on the SimPrior residual — the path the leverage probe measures as
weak and net-suppressive (remove residual → count **+29.9%**; fine sens 15.2 vs cond 1.6, ~10×)
— so a per-image amplitude scalar on a 64-ch suppressive residual could not place integrated
mass the way `fine` does; construction C1–C7 confirmed (as-booked dispatch, no N0023-style
exclusive-dispatch bug; cellcal/peakcal retained on all four call sites); alpha engagement reads
(R1/R2) **not in local dump** (no `val_attr.json`; F1 vs F2/F3 indeterminate — not claimed).
Noise: ≈+1.20 of +2.88 is parent draw-luck (draw-mean 20.544); the rest is structure (dense
outside retrain band, mid at family's ~42 plateau, ep5 crossover vs N0023, HALT above worst
draw 21.59). Consensus: **mechanism-wrong (low-leverage actuator)**, as-booked, not quiet-null
and not gate-too-strict. Feedback agreements: all four say REFUTED via futility-HALT on the
measurable val/triple-bar line; residual disagreements (F1 vs F2/F3 sub-class; sparse as
verdict-driver vs footnote) resolve toward diagnostic/qual: verdict rests on R4 + futility +
noise-invariant structure; F1–F3 stay open until an alpha dump is pulled.

## 2. Distinct contributions of the 4 feedbacks

| Feedback | Unique load-bearing claims |
|---|---|
| quant | Artifact inventory + §6 addendum (val_result/test_result now local); full triple-bar table with parent dump cross-checks; tb trajectory vs N0023/N0024 (ahead only eps 1–4, behind from ep5, +0.057 / −0.84 at ep24); exact futility arithmetic (ep16 WARN need 0.3489; ep24 ceiling 20.0031, need 0.3972); R1–R3/AE-split rows all **not in dump** with source citations |
| qual | Two-joint BECAUSE reading (diagnosis fork OR + CountingDINO z-anchor remedy); line-by-line booked-vs-code **MATCH** (clamp, detach, post-`out`, four call sites — N0023 bug did NOT recur); leverage-probe interpretation (α multiplies a net-suppressive low-gain actuator; clamp-max 4× is a sub-0.4 fine-equivalent before co-adaptation); sibling three-card arc table; rules-out residual/`ev` scale as primary mass actuator; operational lesson: ship attr dump next time |
| causal | Construction ledger C1–C7 (file-anchored) vs measured R1–R4 table; residual path structural contrast (`cond_map = cond + SimPrior`; α only moves post-`out` amplitude); leverage-probe §3 reading (wrong-lever by construction; directional F2 risk if α<1); causal chain A–E with file-backed vs blocked arrows; single load-bearing statement for the paper |
| diagnostic | Failure taxonomy **mechanism-wrong / low-leverage actuator** with noise decomposition (≈+1.20 luck / ≈+1.68 child deviation); dispatch-vs-booking full test (substantively fully tested for bars — R4∧R1 card, R4 fails alone); F1–F5 matrix (only F5 trajectory half readable); root-cause hypothesis with 5-link evidence chain; follow-up alpha-dump read list; recommendation: no residual-scale re-encode; proceed H0025 CapFilm, pull dump for paper-grade F1-vs-F3 localization |

## 3. Confidence check (ledger-reported, not recomputed)

`python3 scripts/discovery.py prove H0024` and `calibration` standings both report:
**H0024 conf = 0.4150 (uncertain)**, tests=1 — after the single evidence event
`contradicts w=0.85` (N0025_h0024, in-tree note: FUTILITY-HALTED ep24, val +2.88, bars missed,
mechanism reads unmeasurable). Below the refuted threshold (<0.25) is NOT met; REFUTED-class
status here follows the verdict semantics (futility refutation + all three bars missed), while
the Eq.1 posterior stays 0.415 until the machinery books further evidence. Eq.1 math lives only
in `src/cac/expt/` — no hand recomputation performed. H0025 stands at 0.415 (tested elsewhere);
H0026 at 0.500, tests=0 — neither touched by this node.

## 4. Quality gate

### (a) Format gate — no new candidate texts this node

No new hypothesis text was drafted for booking. The card-as-tested is H0024's booked line
(`idea.md:166`), markers already PASS from booking time.

### (b) K_SYNTH=2 cap — candidate bookings (novelty-checked)

| # | Candidate | novelty_check | Registered duplicate? | Action |
|---|---|---|---|---|
| A | Residual/`ev`-path scale or content re-encode (any unit-mass/z-anchor/α-scale variant on SimPrior residual) | would fail structural-twin (this card) | **YES** — H0024 itself; §5.11 requires `contradicts` evidence on H0024 with a genuinely new falsifier | **NOT booked** — spent direction; any re-encode needs the existing id + new falsifier, not a fresh id |
| B | Decoder/consumption-side probe (CapFilm-class) or exemplar-distinctness | already registered | **YES** — H0025 CapFilm in flight (N0026); exemplar-distinctness registered fallback (HANDOFF §48/62, N0023 synthesis §4b) | **NOT booked** — both pre-registered elsewhere; booking now would duplicate the ledger and preempt the HANDOFF card budget |

**Bookings created: 0 of 2 allowed** (zero bookings is the justified outcome — the response×post-out
magnitude fork is closed at the booked bars; decoder-side and encoding-layer directions already
have their own ids/slots).

### (c) Opposite-of-existing rule

Applied: the natural opposite of H0024 ("scale the residual" vs "do not scale / move mass
elsewhere") is either (i) a `contradicts` event on H0024 — already appended (`w=0.85`) — or
(ii) the already-booked H0025 decoder-side card. No opposite-of-existing text minted a new id.

### (d) Misattributed reasoning — remap or discard

- **R1/R2 "mechanism did/didn't engage" claims → DISCARDED as unmeasurable.** No alpha/attr
  dump and no local child `val_perimage` reads for wipe/KEEP medians in the feedback set that
  produced them; F1 (quiet-null) vs F2/F3 (low-leverage) remains **indeterminate** and is not
  carried forward as a settled sub-class. Only the probe-based design explanation (qual/causal/
  diagnostic §4) is carried, labeled PLAUSIBLE.
- **Sparse FAIL row as verdict-driver → DISCARDED** (footnote): child improved 5.779 vs parent
  5.814; 5.45 is the standing parent-infeasible bar — same pattern as N0023; dense + val decided.
- **`info.json` status `"timeout"` as budget exhaustion → DISCARDED**: `budget_hit: false`,
  elapsed 1333.6 < 1800; runner maps futility→timeout (N0023 diagnostic note).
- **"N0023 exclusive-dispatch bug" applied to this card → DISCARDED** (belongs to H0023): code
  threads `use_canchor` through all four call sites; cellcal/peakcal retained — remapped nowhere.
- Remapped intact: leverage-probe premises → stay as recorded file facts from
  `N0026_h0025/idea.md` + `journal/events.jsonl:65`, not re-ablated on N0025; alpha-dump read
  list → diagnostic §5 (post-hoc, non-blocking).

## 5. Calibration (verbatim `python3 scripts/discovery.py calibration`)

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

## 6. Decision-relevant summary for the Lead

1. This node **closes fork (b)** (absolute count-bearing magnitude on the SimPrior residual /
   post-`out` scale) of the two-card closure OR, at the level of the booked bars — and with the
   three-card arc (H0022 sign, H0023 phase/substrate, H0024 magnitude) it brackets **sign,
   phase, and magnitude on the residual/`ev` path**, all +2.8…+3.7 with dense +123…+161. No new
   bookable direction opens here; both successors are pre-registered.
2. **Book nothing new now** (0/K_SYNTH): decoder/consumption-side is H0025 CapFilm (in flight,
   N0026); encoding-layer fallback is exemplar-distinctness (HANDOFF slot). Any residual-scale
   re-encode must be `contradicts` evidence on H0024 with a genuinely new falsifier (§5.11) —
   thin re-encode, negative expectation.
3. Verdict currency: **R4 triple bars 0/3 + futility HALT outside the draw band**, with
   noise-invariant structure (dense +93 past retrain-band edge, mid ~42 family plateau, ep5
   crossover). F1 vs F2/F3 localization needs the alpha dump (diagnostic §5) — pull when
   convenient for the paper; does not block the next card.
4. Operational: always ship `val_attr.json`-style mechanism dumps with runs so R1/R2 are
   locally decidable; do not misread `info.json` `"timeout"` on a futility halt as budget stop.
5. Sequence unchanged: proceed with the already-booked decoder-side probe, then the registered
   exemplar-distinctness card if that fails, then the HANDOFF “3 cards no >1.5 → paper”
   decision. Live parent remains N0015_h0014 (19.3431).

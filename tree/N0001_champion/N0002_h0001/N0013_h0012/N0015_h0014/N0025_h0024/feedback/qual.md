# Qualitative feedback — N0025_h0024 (`use_canchor` CountAnchor, child of N0015_h0014)

Text-log analysis per AGENTS §9/§3. Sources read directly this session (all local, no ssh, no
commit): node `idea.md` / `model.py` / `config.toml` / `result.json` / `info.json` /
`val_result.json` / `test_result.json` / `feedback/quant.md` / `feedback/causal.md`;
parent backup `/home/qkun/cac_backup/N0015_val_result.json`; sibling feedback sets
`../N0023_h0022/feedback/{qual,diagnostic}.md` and `../N0024_h0023/feedback/{qual,diagnostic}.md`;
leverage-probe premises in `../N0026_h0025/idea.md:8,47` and `journal/events.jsonl:65`.

**Artifact caveat (honesty §5.5/§5.7):** no `val_attr.json` / alpha dump and no
`val_perimage.json` are local under the node (quant §0 inventory), so every R1/R2 alpha read
below is reported **not in dump** — neither claimed nor reinterpreted. Slice numbers in §1 come
from the now-present `val_result.json` (addendum to `feedback/quant.md` §6).

Verified outcome:

| quantity | booked bar | N0025 | vs parent (19.3431 / 330.81 / 5.814) |
|---|---|---|---|
| EMA val MAE | ≤19.0431 | **22.2127** (`val_result.json`; best 22.2205 @ep24 in `result.json`) | **+2.87** |
| dense gt>500 (n=17) | <330.81 | **454.05** | **+123.2** |
| mid 50–500 (n=374) | (guard) | **41.91** | +4.41 |
| sparse gt<50 (n=895) | ≤5.45 | **5.779** | −0.035 (improved, still over bar) |
| test MAE (n=1190) | — | **19.437** | parent test 19.066 (STATE) |
| futility | ep24 HALT iff best >20.00 | **HALT**, best 22.2205, need 0.3972/ep ≫ 0.12 | outside (20.0, 21.6) draw band |
| R1a/b/c alpha reads | CV≥0.05, wipe/KEEP ≥1.3×, resid +20% | **not in dump** | — |
| R2 wipe movement | mean ≥0.40, ≥4/11 ≥0.50 | **not in dump** (no child perimage) | — |

---

## 1. What CountAnchor was supposed to do

The booked BECAUSE (`idea.md:12,166`) has two joints:

1. **Diagnosis (the closure's open fork).** After H0022 (sign-mask, 0/11 wipe rescue with the
   mask fully engaged) and H0023 (substrate swap, 0/11 even with construction-true post-swap
   evg_sum +18.8k), the dense-wipe locus localizes **below the evidence/gain layer**, leaving an
   OR: (a) the consumption interface `out→cond→decoder` cannot convert evidence into mass, or
   (b) the evidence simply **lacks count-bearing absolute magnitude** — cosine `top1 ∈ [-1,1]`,
   `mhat` mean-1 by construction (spatial sum pinned ≈9216), cellcal's gain a relative ~1.1–1.3×
   multiplier of a unit-sum field, and `out` linear. Both substrates are **scale-free**: no
   channel encodes "this image needs ~500 more counts here" (`idea.md:46`). H0024 tests fork (b)
   only.
2. **Remedy (CountingDINO's unit-mass anchor, in-head).** Inside `SimPrior.forward`, after the
   unchanged cellcal/peakcal blocks: minmax-normalize detached positive `top1`, ROI-align mean
   over the K=3 annotation boxes (each box structurally holds one object → fixes absolute mass
   K), `alpha = clamp(1/z, 0.25, 4.0)`, return `self.out(ev) * alpha` (`model.py:255–267`) —
   one stop-gradient per-image scalar on the post-`out` residual, zero params/modules/RNG, step-0
   identity (zero-init `out` ⇒ `0·alpha = 0`). Then the THEN-bars: val ≤19.0431 AND dense
   <330.81 AND sparse ≤5.45.

**Implementation as booked (diff-verified construction, CONFIRMED this session):** flag threaded
through all four `simprior(...)` call sites (`model.py:177–187`) so cellcal/peakcal stay runtime-
active on every image (the N0023 exclusive-dispatch bug did NOT recur — unlike N0023, no parent
gain was bypassed); anchor block sits after both gain blocks (`:255`), before `return`; config
delta is exactly `use_canchor = true` (`config.toml:30`), `use_simprior/cellcal/peakcal` all
true; temp-pin path untouched (`use_peakcal=true`). The card executed what it booked.

**Outcome vs the remedy's predictions:** every bar missed — val +2.87, dense +123.2, sparse
5.779 (improved vs parent but over the 5.45 bar), futility HALT at ep24 with best 22.2205
(+2.22 above the 20.0031 ceiling, outside the card's own draw-contaminated band (20.0, 21.6),
`idea.md:132,139`). R1/R2 alpha engagement reads: **not in dump** — the mechanism neither
confirmed nor cleared locally; the card needed BOTH R4 AND R1, and R4 alone already fails as
booked.

## 2. Honest design assessment: why response-based alpha on the residual path is low-leverage

The mask-swap-scale trilogy (H0022 sign → H0023 substrate → H0024 magnitude) walked the closure
OR correctly, and H0024 put the magnitude fork on the **one path the leverage probe measures as
weak and net-suppressive**:

- **Leverage probe premises (CONFIRMED as recorded file facts, from `N0026_h0025/idea.md:8,47`
  and `journal/events.jsonl:65`, not re-run on N0025):** ablations on the trained parent —
  `fine=0` → count **−91.6%**; `cond_map=0` → count **+23.8%**; **remove SimPrior residual →
  count +29.9%** (the residual is *net-suppressive*); `cond×2` → −17.6%; channel sensitivity
  fine mean **15.2** vs cond mean **1.6** (~10×).
- **Reading for H0024 (mechanism-consistent, PLAUSIBLE):** count mass lives on `fine`. The
  SimPrior residual — the exact tensor `alpha` multiplies — is a weak, net-suppressive cond-path
  term: removing it *raises* count by 29.9%. Scaling a suppressive actuator by α ∈ [0.25, 4]
  multiplies the wrong quantity: even the clamp-max 4× cannot inject mass onto the generator path
  the decoder integrates, and α<1 on healthy images (the priced F2 starvation) would suppress
  *further* — fewer counts, higher AE. A response-based scalar derived from in-box similarity
  response (`z`) is a **per-image sample-weight on a low-gain suppressor**, not a count actuator.
- **Under-powered even if α engaged:** R1 only demanded CV≥0.05 and 1.3× wipe/KEEP median
  separation — engagement ≠ count movement. With cond sensitivity ~0.106 of fine's, a 1.3–4×
  residual scale is a sub-0.4 fine-equivalent perturbation before decoder co-adaptation; F3
  ("scale without shape" / consumption-interface fork) is the card's own pre-registered name for
  this outcome (`idea.md:130`).
- **What α structurally cannot do (CONSTRUCTION from `model.py:255–267`):** change spatial phase
  of the residual, add a tensor, touch `fine`, or reach the decoder's generator input. It moves
  only the amplitude of the already-trained `out(ev)` term (64-ch, added into `cond_map`). Six
  prior residual/evidence cards (H0016/H0017 quiet-null ×2; H0018/H0022/H0023 wipe 0/11) already
  died editing this path; H0024 tested **absolute magnitude on the same low-leverage path** — the
  right fork of the OR, plausibly the wrong actuator within that fork.

**Design-quality positives worth keeping:** zero params/modules/RNG (strongest append-only
form); step-0 identity via zero-init `out`; stop-gradient everywhere (no second grad path —
H0006/H0007 family clean); cellcal/peakcal retained at runtime (N0023-bug class avoided);
one-line config delta; pluggable single switch; CountingDINO (WACV 2026, reproduced in-lab at
val 39.70) is a real published source for the z-anchor direction, so the idea was surveyed and
novel (gate exit 0, top sim 0.483 vs H0023). The falsifier bars were specific and pre-registered
to the live parent. The card was a well-formed bet that simply landed on an actuator the leverage
probe had already priced as weak.

## 3. Comparison to sibling verdicts (N0023 / N0024)

| | N0023 (H0022 margin) | N0024 (H0023 antimatch) | **N0025 (H0024 canchor)** |
|---|---|---|---|
| operator | sign-mask on `ev` (evidence layer) | substrate swap on `ev` (evidence layer) | post-`out` residual ×α (consumption-adjacent) |
| parent gain stack | cellcal BYPASSED (w_c=0, booked replacement) | cellcal retained, learned weak (w_c 0.0282) | cellcal/peakcal retained, as-booked |
| val MAE | 22.163 (+2.82) | 23.062 (+3.72) | **22.213 (+2.87)** |
| dense gt>500 | 458.69 (+127.9) | 491.81 (+161.0) | **454.05 (+123.2)** |
| mid | 41.71 (+4.21) | 42.64 (+5.14) | **41.91 (+4.41)** |
| sparse | 5.703 (improved, over bar) | 5.972 (+0.16 vs parent) | **5.779 (improved, over bar)** |
| wipe rescue R2 | 0/11 | 0/11 (7/11 correctly routed, +18.8k) | **not in dump** |
| futility HALT ep24 | yes (need 0.403/ep) | yes (need 0.525/ep) | **yes (need 0.3972/ep)** |
| verdict | REFUTED, contradicts w=0.85 | REFUTED, contradicts w=0.85 | **REFUTED via futility-HALT** |

Reading across the three: N0025 lands essentially at **parity with N0023** (+0.057 at ep24 on
the tb curve; endpoint val 22.21 vs 22.16) and **better than N0024** (−0.85), but all three sit
+2.8 to +3.7 above the live parent, all three HALT at ep24, all three blow the dense bar by
+123 to +161. The three-card arc is coherent: sign repair (H0022) failed, phase repair (H0023)
failed, and now **absolute-magnitude repair on the residual (H0024) also failed** — closing fork
(b) of the two-card closure OR at the level of the booked bars. What remains genuinely open
(except where alpha reads would have distinguished it) is fork (a), the consumption interface —
which is exactly the locus the leverage probe points at (fine=generator, cond/residual=weak
suppressor) and the direction `journal/events.jsonl:65` pivoted the next card class to
(decoder-side, H0025 CapFilm on N0026). Sparse behavior echoes N0023 (slight improvement, still
over the standing parent-infeasible 5.45 bar) rather than N0024's regression — consistent with
α applying continuously to all images rather than hard-routing a majority.

**Honest residue:** without the alpha dump we cannot say whether N0025 died as F1 quiet-null
(α≈1 everywhere — in-box self-match stays near-ceiling because boxes are the exemplars' own
crop, the card's dominant priced risk, `idea.md:136`) or as F2/F3 (α moved, counts didn't —
the low-leverage reading above). The verdict as booked rests on R4 alone (all three bars
missed + futility HALT outside the draw band); any claim of a *specific* failure mode without
the attr dump would be invention. The level result (22.21, worst-band-adjacent) is consistent
with the low-leverage hypothesis but does not by itself distinguish the failure modes.

## 4. Booking-quality notes

- **Booked vs code: MATCH** (line-checked `model.py` vs `idea.md` §2.2 snippet, incl. clamp
  [0.25, 4.0], `top1.detach()`, post-`out` placement, four call-site kwargs). No dispatch bug.
- **Falsifier well-posedness:** the three numeric bars are specific and parent-bound; the two
  engagement disjuncts (alpha CV <0.05, wipe/KEEP median <1.3×) are report-style R1 reads placed
  inside DISPROVED — unlike H0022/H0023's vacuous construction-true 4th disjuncts, these are
  *empirically falsifiable* (a flat α would fire), so the structure is sound; they just cannot
  fire locally with no dump. Sparse ≤5.45 is the standing parent-infeasible bar again (parent
  5.814) — a footnote, not the story (child actually improved to 5.779).
- **Process:** `info.json` records `status: "timeout"` while `result.json` has
  `futility_hit: true` — by design (run_node maps futility to timeout status, N0023 diagnostic
  note), easy to misread. `config.toml` still says `augment = false` while the launch carried
  `--set augment=true --set futility_bar=19.0431` (recorded in `result.json.cfg_overrides`) —
  same config-vs-runtime semantics gap siblings flagged. The runner-parseable
  `1. **H0024** — <text>` line IS present (`idea.md:166`; STATE gotcha 2 satisfied,
  `tested_hypotheses: ["H0024"]`).
- **Noise framing honored:** same-seed draws {19.34, 20.70, 21.59}, sd≈1.13; 22.21 sits above
  the worst historical draw and outside the card's own (20.0, 21.6) contaminated band — the
  refutation is the pre-registered bar + futility line, not a mechanism read we do not have.

## 5. What this rules in / rules out for the next card

**Rules out.** (a) Response-based absolute-scale (1/z unit-mass) applied to the SimPrior
residual as a wipe/mass cure — this card, as booked, REFUTED. (b) Any further content or scale
edit of the residual/`ev` path as the *primary* mass actuator: the three-card family (mask,
swap, scale) now brackets sign, phase, and magnitude on that path, all +2.8 to +3.7 with dense
+123 to +161 — the path is measured weak-suppressive by the probe and empirically inert-to-harmful
under all three operators. (c) Quiet readings: the run HALTed outside the draw band and missed
every bar; no successor may argue H0024 "ran out of epochs."

**Rules in.** (1) Fork (b) is closed at the level of the booked bars → the closure OR collapses
toward fork (a), **the consumption interface** — precisely the `journal/events.jsonl:65` pivot
(decoder-side per leverage probe; H0025 CapFilm booked on N0026 as the registered probe).
(2) The registered exemplar-distinctness fallback remains available as the encoding-layer
direction if decoder-side also fails. (3) Any future card that re-encodes unit-mass/z-anchor or
residual scaling must book as `contradicts`-evidence on H0024 with a genuinely new falsifier
(§5.11); the CountingDINO transfer-into-trained-head form is spent. (4) Operational lesson for
successors: ship the attr dump (`val_attr.json`-style, alpha or equivalent mechanism scalar)
with the run so R1/R2 are locally decidable — this card's biggest analysis gap is the missing
alpha dump, not the missing bars.

---

**Files written:** this document only — `feedback/qual.md`. No commit, no ssh, no ledger /
`info.json` / `quant.md` edits by this role (the quant §6 addendum is the Quantitative agent's).

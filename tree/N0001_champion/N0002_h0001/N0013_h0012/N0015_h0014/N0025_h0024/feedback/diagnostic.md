# Diagnostic — N0025_h0024 (H0024 `use_canchor` CountAnchor) — FAILURE (futility HALT)

- **id/parent**: N0025_h0024 ← N0015_h0014 (live, val 19.3431 @ep32 v2).
- **mechanism**: zero-param stop-gradient exemplar-box unit-mass anchor — after unchanged
  cellcal/peakcal, `alpha = clamp(1/z, 0.25, 4)` (z = ROI-align mean of minmax-normalized
  detached top1 over K=3 boxes) multiplies the post-`out` SimPrior residual on every image;
  one boolean `use_canchor`, zero params/modules/RNG (`idea.md:4`; `model.py:255–267`).
- **outcome**: **futility-HALTED ep24** — `result.json`: `best_mae` **22.22052001953125**,
  `best_epoch` 24, `n_epochs_done` 24/32, `futility_hit` **true**, `budget_hit` false,
  elapsed_s 1333.6, `cfg_overrides` {augment: true, futility_bar: 19.0431}; shas config
  `f30d9a561f34090d` / model `8599b71f9a5644dd`. Full eval pulled locally:
  `val_result.json` mae **22.2127** (+2.88 vs live 19.3431), dense **454.05** (+123.2 vs
  330.81, n=17), mid **41.91** (n=374), sparse **5.779** (n=895); `test_result.json` mae
  **19.437** (n=1190). **All three booked bars failed** (val ≤19.0431, dense <330.81,
  sparse ≤5.45). `info.json` `status: "timeout"` (runner maps futility→timeout), best_metric
  22.2205, epochs 24, `tested_hypotheses: ["H0024"]`.
- **constraints**: frozen backbone, head-only, seed 20260830 fixed, v2 protocol, AGENTS §5.16
  futility (ref slope 0.06/ep, margin 2.0 → threshold 0.12; ep16 WARN / ep24 HALT), AGENTS §11
  no-repeat register. No commits, no ssh performed for this document.

Verified locally from node artifacts read this session: `result.json`, `info.json`,
`val_result.json`, `test_result.json`, `config.toml` (delta = exactly `use_canchor = true`,
`use_simprior/cellcal/peakcal` all true), `model.py`, tb event file
`run/latest/tb/t/events.out.tfevents.1790054936…10646.0`, `feedback/quant.md`,
`feedback/causal.md`, parent `/home/qkun/cac_backup/N0015_val_result.json`
(mae 19.3259, dense 330.8131), sibling diagnostics `../N0023_h0022/feedback/diagnostic.md` and
`../N0024_h0023/feedback/diagnostic.md`, leverage-probe premises `../N0026_h0025/idea.md:8,47`,
`journal/events.jsonl:65`.

---

## 1. Failure taxonomy

**Primary class: mechanism-wrong / low-leverage actuator — the magnitude fork landed on the
measured weak-suppressive path and every bar missed by mechanism-level margins.** Secondary:
analysis gap (missing alpha dump) prevents sub-classifying F1 vs F2/F3. Not operator-wrong
(dispatch as-booked, §2), not gate-too-strict (HALT beyond every observed draw).

**(a) What part of +2.88 can be noise.** Same-seed draws of the identical N0015 config:
{19.3431 live, 20.697 N0021 retrain, 21.5928 exact repro}, mean 20.544, sd ≈ 1.13
(sibling diagnostics §1a, `perimage_diagnostic.md:231`). Decomposition of the child's +2.88
(22.2127 − 19.3431 ≈ 22.2205 − 19.3431 = +2.8774):
- **≈ +1.20 is parent draw-luck** (20.544 − 19.3431) — live parent is the minimum draw.
- **≈ +1.68 is the child's deviation above the draw-mean** (~1.5σ of a single draw; under
  exchangeable sd(child−parent) ≈ 1.60, +2.88 ≈ 1.8σ) — noise-consistent in isolation but
  strained; per the HANDOFF ≥+2 "credible" heuristic most should be treated as real.
- Decisive noise-invariant reads below carry the verdict regardless.

**(b) What is mechanism-caused and cannot be noise.**
- **Dense +123.2** (454.05 vs booked 330.81): the same-seed retrain band for N0015's dense
  tail is [300, 361] (H0020 booking, cited in sibling diagnostics). 454.05 exceeds the band
  upper edge by **+93** — at least +93 of the +123.2 is structural.
- **Mid +4.41** (41.91 vs parent 37.50, n=374): the third card in this lineage to push mid to
  ~42 (N0023 41.71, N0024 42.64) — a stable slice regression tracking residual/`ev`-path
  interventions, not level noise.
- **Trajectory shape (tb `val/mae`, 24 pts):** ahead of sibling N0023 only at **eps 1–4**
  (ep1 lead −4.40), **behind from ep5** (+0.102) and every epoch thereafter, ending **+0.057**
  at ep24 (parity with N0023) and **−0.84** vs N0024 — a systematic mid-train divergence, not a
  level offset. Monotone decrease through ep24; best = last = 22.2205. (Full table: `quant.md` §4.)
- **Futility arithmetic (exact, `quant.md` §4):** ep16 best 24.6253 → need 0.3489/ep > 0.12 →
  WARN; ep24 ceiling 19.0431 + 0.12×8 = 20.0031, best 22.2205 → need 0.3972/ep > 0.12 →
  **HALT**, +2.2174 above ceiling. Booked gate (`idea.md:132,139`): HALT iff best > 20.00;
  the draw-contaminated band is (20.0, 21.6) — **22.2205 sits above 21.6**, i.e. outside the
  band by the card's own arbitration rule → not draw-contaminated; R1/R2 were the registered
  tiebreakers and are unavailable (§2), so the verdict is futility + triple-bar as booked.
- **Sparse 5.779:** improved vs parent 5.814 (−0.035) but over the 5.45 bar — same pattern as
  N0023 (5.703); sparse never decided this verdict (standing parent-infeasible bar; dense + val
  did).
- **Test 19.437** (n=1190): full eval per AGENTS §5.16 (futility-halted run keeps best.pth →
  verdict = futility refutation + full eval). Above parent test 19.066 (STATE) — corroborating,
  not the booked currency (val is).

**(c) Classes explicitly ruled out.**
- *Gate-too-strict / early stop*: need 0.3972/ep ≫ 0.12 at ep24; even extrapolating the
  **parent's own** late slope −0.28/ep from 22.22 ends ≈20.0 @ep32 — still **+0.95 above**
  the 19.0431 bar. The HALT saved ~8 epochs without changing the verdict (AGENTS §5.16).
- *Operator-wrong / dispatch bug*: code matches booking line-for-line (§2); unlike N0023 there
  is no exclusive-dispatch — cellcal/peakcal flags true and threaded through all four call
  sites (`model.py:177–187`). Smoke was recorded OK pre-launch (journal, Lead-side).
- *Draw-noise as sole cause*: contradicted by (b) — dense +93 outside band, mid +4.41, the
  ep5-crossover trajectory, and best 22.22 > worst draw 21.59.
- *Budget timeout*: `budget_hit: false`, elapsed 1333.6 < 1800 — this is a futility halt, and
  `info.json`'s `"timeout"` label is the runner's mapping, not a wall-clock blow (N0023
  diagnostic note).

**Taxonomy verdict: mechanism-wrong (low-leverage actuator).** The anchor's construction
executed as booked; the absolute-scale scalar multiplied a residual the leverage probe measures
as net-suppressive and ~10× weaker than `fine` (remove residual → +29.9% count;
`N0026_h0025/idea.md:8,47`, `journal/events.jsonl:65`) — so the magnitude fork, correctly
identified by the two-card closure, was delivered through the wrong channel. Noise accounts for
roughly the first +1.20 of +2.88; the rest is structure. The refutation rests on reads noise
cannot produce: triple-bar miss with dense outside the retrain band, mid at the family's ~42
plateau, and a futility HALT beyond every observed draw.

---

## 2. What was actually tested — dispatch vs booking, partial-test residue

**Facts (code).**
- `config.toml:27–30`: `use_simprior/cellcal/peakcal/canchor` all `true` — matches runtime (no
  N0023-style config-language caveat; filtered delta vs parent is exactly the one new line).
- Dispatch: `model.py:177–187` threads `bboxes=…, img_size=…, use_canchor=…` through **all four**
  parent arms — cellcal and peakcal reachable on every image. Anchor is a kwarg block inside
  `SimPrior.forward` **after** cellcal (`:237–242`) and peakcal (`:243–254`), not a branch that
  returns early — booked order (`idea.md:52`).
- Anchor math as booked: `top1.detach().clamp_min(0)` → spatial minmax → ROI-align mean over
  K boxes → `alpha=clamp(1/(z+1e-6), 0.25, 4.0)` → `return self.out(ev) * alpha`
  (`model.py:255–267`); flag-off path `return self.out(ev)` byte-identical (`:268`).
- Zero new parameters (no module constructed); temp-pin condition untouched
  (`use_peakcal=true` keeps `temp.requires_grad_(False)`, `model.py:331–332`).

**Assessment — is H0024 only partially tested?**
- **On the booked design: no — substantially (fully) tested for the bars.** Every code clause
  executed; all three numeric bars were measured on the pulled `val_result.json`/`test_result.json`
  and all three failed; futility fired as specified. The card needed R4 ∧ R1; R4 fails decisively,
  so the card is refuted regardless of R1.
- **One honest residue: the mechanism reads are locally unmeasurable.** R1a (alpha CV ≥0.05),
  R1b (wipe-11 vs KEEP-27 median alpha ≥1.3×), R1c (residual L2 +20%), R2 (wipe pred/gt
  movement) all require an alpha/attr dump and/or `val_perimage.json` — **none are in the local
  node dir** (no `val_attr.json`, no alpha field anywhere; `val_perimage.json` absent — only
  `val_result.json`/`test_result.json` slices were pulled). Stated honestly: **engagement
  R1/R2 is unmeasurable this session; nothing is invented.** Consequence: we cannot distinguish
  the card's pre-registered failure modes —
  - **F1 quiet-null** (α≈1 everywhere; in-box self-match near-ceiling because boxes are the
    exemplars' own crop — the card's dominant priced risk, `idea.md:128,136`), vs
  - **F2/F3 low-leverage** (α moved with real spread, but scaling a net-suppressive weak residual
    could not place mass / starved healthy images).
  Both are consistent with the level outcome; the probe-based reading (§1b) favors F2/F3 as the
  *design* explanation, but without the dump that remains a hypothesis, not a measurement.
- **Untested counterfactuals (successor territory, not incomplete test):** α on `fine` or at the
  decoder (the CapFilm/H0025 direction, `journal/events.jsonl:65–66`); a count-derived rather
  than response-derived scalar (would re-encode H0010 lessons). Neither is an untested limb of
  H0024-as-booked.

**Caveat for the record:** `info.json` status `"timeout"` on a futility HALT is by runner design
— do not misread as budget exhaustion (`result.json.budget_hit: false`).

---

## 3. Failure-mode matrix (pre-registered modes, fired?)

| mode | booked signature (`idea.md:126–132`) | evidence available | fired? |
|---|---|---|---|
| **F1** quiet null | α flat, CV<0.05, numbers level with parent | **alpha dump not in dump**; level NOT level (+2.88) | **indeterminate** (cannot claim) |
| **F2** wrong-direction / starvation | α separates backwards or α<1 drags majority; mid/sparse regress | alpha not in dump; mid +4.41 (family-consistent), sparse *improved* slightly | **indeterminate / partial circumstantial** (mid up, sparse not) |
| **F3** scale-without-shape | R1 holds but wipes stay wiped → consumption fork (a) wins | R1 not in dump; dense +123.2 with wipes unknown | **indeterminate** (the scientifically central read is missing) |
| **F4** clamp saturation | α piles at 0.25/4.0 | histogram not in dump | **indeterminate** |
| **F5** late-phase divergence | crossover with anchor live, cellcal retained | tb: crossover at **ep5 vs N0023** (ahead only eps 1–4), then steady behind; HALT ep24 | **YES (trajectory half)** — but vs sibling, parent tb not local |

Only F5's trajectory half is directly readable; F1–F4 all gate on the missing alpha dump. The
bars + futility carry the verdict alone, as §1–2 establish.

---

## 4. Root-cause hypothesis (with file evidence)

**Hypothesis (PLAUSIBLE — mechanism-consistent, not ablated on N0025):** CountAnchor put the
closure's absolute-magnitude fork on the residual path, which the leverage probe measures as
**weak and net-suppressive** — so even a correctly-engaged response-based α was structurally
under-powered to move integrated count mass, while any α<1 pressure on healthy images worked
against the undercount regime the lineage sits in.

Evidence chain (each link file-backed):
1. Residual is the scaled object: `return self.out(ev) * alpha` (`model.py:267`) — α cannot
   touch `fine`, the generator (probe: `fine=0` → −91.6% count; sens 15.2 vs cond 1.6).
2. Residual is net-suppressive: probe **remove residual → +29.9% count**
   (`N0026_h0025/idea.md:8,47`; `journal/events.jsonl:65` — recorded, not re-run here).
3. Family precedent: six residual/evidence-content cards died on this path (H0016/H0017
   quiet-null; H0018/H0022/H0023 wipe 0/11; sibling diagnostics §3) — H0024 varied *scale*, the
   last untested axis of the same path, and landed in the same +2.8…+3.7 / dense +123…+161
   corridor.
4. Outcome: all three bars missed (`val_result.json`), futility HALT beyond draw band
   (`result.json` + arithmetic), trajectory behind sibling from ep5 (tb).
5. Design-level fit: CountingDINO's z-anchor works **training-free on the full density
   pathway** (`idea.md:18`); transplanting it as a scalar on one suppressive residual term
   preserves neither property — the survey's own "Difference" paragraph (`idea.md:18`) states
   this; the run is the empirical confirmation.

**What would falsify/refine this hypothesis:** the alpha dump. If CV≥0.05 with wipe-loading
≥1.3× (R1 true) and counts still flat → confirms low-leverage/F3 and closes fork (b) hard,
pointing to fork (a) (consumption interface — already the `journal:65` pivot to CapFilm). If
CV<0.05 → F1 quiet-null: the in-box self-match structural risk (`idea.md:136`) fired and the
magnitude fork was never actually *tested* in-head (a weaker closure claim). **Until the dump is
pulled, both remain open; this document claims neither.**

---

## 5. Follow-up diagnostic reads (when the dump is pulled)

All inference-only, using the existing attribution tooling behind `local/dump_antimatch.py` /
`perimage_diagnostic.md` (alpha instrumentation must be added for `use_canchor` — it was never
dumped by the run):
1. **Alpha histogram + CV** on val at best.pth (R1a): decisive for F1 vs F2/F3.
2. **Median α on the 11 booked wipes {840, 851, 865, 935, 3428, 3481, 3482, 3484, 3487, 3488,
   7656} vs 27 KEEP (gt≥300)** (R1b): direction of the response-based anchor.
3. **Clamp saturation fraction** (F4): share of images at α=0.25 or 4.0.
4. **Child `val_perimage` paired vs `/home/qkun/cac_backup/N0015_val_perimage.json`**: wipe
   ratios (R2), KEEP-guard (R3), dense-block 17-image table — completes what `val_result.json`
   slices only summarize.
5. **Residual L2 on wipes vs parent** (R1c): did amplitude actually move ≥20%?
6. **w_c / w_p / temp scalars in the same pass**: confirm parent gain stack stayed at its
   operating point (w_c ∈ [0.14, 0.17] band, temp == 0.07) — code says it should; a read closes
   the N0024 "alive-but-weak actuator" third-state lesson.

---

## 6. Recommendation

**Verdict stands as booked: REFUTED via futility-HALT (triple bars 0/3, HALT outside draw
band).** Do not retry unit-mass/z-anchor or any residual-scale variant on this path (§5.11 —
would need `contradicts` on H0024 with a genuinely new falsifier; thin re-encode, negative
expectation). Sequence unchanged: the closure OR now leans on fork (a) — proceed with the
already-booked decoder/consumption-side probe **H0025 CapFilm (N0026_h0025)** per the
`journal/events.jsonl:65–66` pivot (leverage-probe directed: edit the generator's consumption,
not the suppressive residual); exemplar-distinctness remains the registered encoding-layer
fallback. Pull the alpha dump when convenient for the paper-grade collapse corpus (F1 vs F3
localization of the magnitude fork) — it does not block the verdict or the next card.

**Noise framing carried:** single-seed bet, draws {19.34, 20.70, 21.59} sd≈1.13; 22.22 is above
the worst historical draw; the refutation rests on the pre-registered bars + futility rule +
dense/mid structure, not on any mechanism read this document does not have.

---

**Files written:** this document only —
`tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014/N0025_h0024/feedback/diagnostic.md`.
No commits, no ssh, no ledger / `info.json` / `quant.md` edits by this role.

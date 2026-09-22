# Diagnostic — N0027_h0026 (H0026 `use_geme` GemeCal) — FAILURE (futility HALT)

- **id/parent**: N0027_h0026 ← N0015_h0014 (live, val 19.3431 @ep32 v2).
- **mechanism**: +1-param zero-init geometry magnitude embedding scale — per-image
  `f = clamp((me/32)^w_g, 0.25, 4)` from annotation exemplar-box areas
  (`me = mean_K(S²/(w_k·h_k))`, CACViT ME, stop-grad geometry), multiplying SimPrior
  evidence `ev` **after** unchanged cellcal/peakcal gains and **before** the unchanged
  zero-init `out` projection; threaded through all four `simprior(...)` call sites;
  one boolean `use_geme`; step-0 identity (w_g=0 ⇒ f=1, and `out` zero-init)
  (`idea.md:4`; `model.py:179–194, 261–263, 295–310, 346–347`).
- **outcome**: **futility-HALTED ep24** — `result.json`: `best_mae` **22.087779998779297**,
  `best_epoch` 24, `n_epochs_done` 24/32, `futility_hit` **true**, `budget_hit` false,
  elapsed_s 1316.5, `cfg_overrides` {augment: true, futility_bar: 19.0431}; shas config
  `c48cbec8d36450f4` / model `f8b0470a4542fd92`. Full eval pulled locally:
  `val_result.json` mae **22.0730** (+2.73 vs live 19.3431), dense **445.05** (+114.2 vs
  330.81, n=17), mid **41.61** (n=374), sparse **5.876** (n=895); `test_result.json` mae
  **19.804** (n=1190). **All three booked bars failed** (val ≤19.0431, dense <330.81,
  sparse ≤5.45). ep16 WARN best **24.0491** (need 0.313/ep); ep24 need **0.3975/ep**
  ≫ 0.12 → HALT, ceiling 20.0031, gap **+2.085**. Local `info.json` still
  `status: "proposed"` / `tested_hypotheses: []` (server sync gap — rule 3, no hand-edit).
- **constraints**: frozen backbone, head-only, seed 20260830 fixed, v2 protocol, AGENTS §5
  futility (ref slope 0.06/ep, margin 2.0 → threshold 0.12; ep16 WARN / ep24 HALT), AGENTS
  §11 no-repeat register. No commits, no ssh performed for this document.

Verified locally from node artifacts read this session: `result.json`, `val_result.json`,
`test_result.json`, `val_perimage.json`, `info.json`, `config.toml` (delta = exactly
`use_geme = true`; `use_simprior/cellcal/peakcal` all true), `model.py`, `idea.md`,
`local/research/h0026_idea_DRAFT.md`, parent `/home/qkun/cac_backup/N0015_val_result.json`
(mae 19.3259, dense 330.8131, sparse 5.8136), sibling diagnostics
`../N0025_h0024/feedback/diagnostic.md` and `../N0026_h0025/feedback/diagnostic.md`,
leverage-probe premises `../N0026_h0025/idea.md:8,47`, `journal/events.jsonl:65`,
draw stats `local/research/perimage_diagnostic.md:229`. **No `w_g`/`f` dump and no local
`run/` tb exist under the node** — R0/R1/R2 and per-epoch trajectory are unmeasurable.

---

## 1. Failure taxonomy

**Primary class: futility-HALT triple-bar level-refutation on the trilogy's similarity-side
amplitude probe — val/dense/sparse all failed with mechanism-level margins; sub-class
(quiet-null vs dead-covariate vs wrong-direction) INDETERMINATE pending the `w_g`/`f` dump.**
Not operator-wrong (dispatch as-booked, §2), not gate-too-strict (HALT beyond every observed
draw), not budget (`futility_hit`, `budget_hit: false`).

**(a) What part of +2.73 can be noise.** Same-seed draws of the identical N0015 config:
{19.3431 live, 20.697 N0021 retrain, 21.5928 exact repro}, mean 20.544, sd ≈ 1.13
(sibling diagnostics §1a, `perimage_diagnostic.md:229`). Decomposition of val 22.0730 vs live
(22.0730 − 19.3431 = +2.7299):
- **≈ +1.20 is parent draw-luck** (20.544 − 19.3431) — live parent is the minimum draw.
- **≈ +1.53 is the child's deviation above the draw-mean** — (22.073 − 20.544)/1.13 ≈
  **1.35σ** of a single draw; under exchangeable sd(child−parent) ≈ 1.60, +2.73 ≈ 1.7σ.
  Strained but the **smallest level gap of the five-card corridor** (N0023 +2.82 … N0026 +3.89);
  still **+0.48 above the worst historical draw** (21.5928).
- Decisive noise-invariant reads below carry the verdict regardless.

**(b) What is mechanism-caused and cannot be noise.**
- **Dense +114.2** (445.05 vs booked 330.81): the same-seed retrain band for N0015's dense
  tail is [300, 361] (H0020 booking, cited in sibling diagnostics). 445.05 exceeds the band
  upper edge by **+84** — at least +84 of the +114.2 is structural. Smallest dense blowup of
  the corridor (N0023 +128, N0024 +161, N0025 +123) but still far outside band.
- **Mid +4.11** (41.61 vs parent 37.50, n=374): fourth card in this lineage to push mid to
  ~42 (N0023 41.71, N0024 42.64, N0025 41.91) — the family's stable mid plateau tracking
  evidence/residual-path interventions, not level noise.
- **Sparse 5.876:** +0.062 vs parent 5.814, over the standing parent-infeasible 5.45 bar —
  essentially flat (unlike N0023/N0025's slight improvements); sparse never decided this
  verdict (dense + val did).
- **Futility arithmetic (exact):** ep16 best 24.0491 → need (24.0491−19.0431)/16 =
  **0.3129/ep > 0.12** → WARN; ep24 ceiling 19.0431 + 0.12×8 = 20.0031, need
  **0.3975/ep > 0.12** → **HALT**, +2.085 above ceiling. Booked gate (`idea.md:176`): HALT in
  (20.0, 21.6) is draw-contaminated — **22.0878 sits above 21.6 by +0.49** → outside the band
  by the card's own arbitration rule → not draw-contaminated; R0/R1/R2 were the registered
  tiebreakers and are unavailable (§2), so the verdict is futility + triple-bar as booked.
  (Post-update convention from final best: need 0.3806/ep — HALT under either ordering.)
- **Test 19.804** (n=1190): full eval per AGENTS §5 futility rule (halted run keeps best.pth →
  verdict = futility refutation + full eval). Above parent test 19.066 (STATE) — corroborating,
  not the booked currency (val is).

**(c) Classes explicitly ruled out.**
- *Gate-too-strict / early stop*: need 0.3975/ep ≫ 0.12; extrapolating the **parent's own**
  late slope (−0.28/ep) from 22.0878 ends ≈19.85 @ep32 — still **+0.81 above** the 19.0431
  bar. The HALT saved ~8 epochs without changing the verdict.
- *Operator-wrong / dispatch bug*: code matches booking line-for-line (§2); all four call
  sites thread `geme_f` (no N0023 recurrence); cellcal/peakcal retained; RNG append-after-
  PeakCal honored; step-0 double identity; shas recorded in `result.json`.
- *Draw-noise as sole cause*: contradicted by (b) — dense +84 outside band, mid +4.11 family
  plateau, and best 22.0878 > worst draw 21.5928. (Level share of noise is larger here than
  N0026 — ≈+1.20 of +2.73 — but the structural slice reads carry the rest.)
- *Budget timeout*: `budget_hit: false`, elapsed 1316.5 < 1800 — this is a futility halt;
  local `info.json` "proposed" is a sync gap, not a run status.

**Taxonomy verdict: futility-HALT level-refutation, sub-class indeterminate.** The trilogy's
similarity-side amplitude probe missed every booked bar by mechanism-level margin; noise
explains ≈+1.20 of +2.73, the rest is structural (dense outside band, mid plateau) or
draw-tail that only the `w_g`/`f` dump can split. Best of the corridor, still 0/3 bars.

---

## 2. What was actually tested — dispatch vs booking, partial-test residue

**Facts (code, line-checked this session).**
- `config.toml:27–30`: `use_simprior/cellcal/peakcal` all true plus exactly one new line
  `use_geme = true` — filtered delta is the single switch.
- Dispatch: `model.py:179–194` computes `geme_f` only if flag + module exist and threads
  `geme_f=geme_f` through **all four** parent arms — cellcal and peakcal reachable on every
  image. Multiply is a kwarg block inside `SimPrior.forward` **after** cellcal (`:243–248`)
  and peakcal (`:249–260`), **before** `return self.out(ev)` (`:261–263`) — booked order.
- Math as booked: `GemeCal` (`:295–310`) `f = exp(w_g·(log me − log 32)).clamp(0.25, 4)`,
  `me = mean_K(S²/area).clamp(1, 1e4)`; `w_g = nn.Parameter(torch.zeros(()))`.
- Attach at Counter end after PeakCal (`:346–347`); temp-pin (`:349–351`) untouched; flag
  non-module (`:167`); +1 param only.

**Assessment — is H0026 only partially tested?**
- **On the booked design: substantially (fully) tested for the bars.** Every code clause
  executed across 24 epochs under the booked protocol; all three numeric bars were measured on
  the pulled eval files and all three failed; futility fired as specified. The card needed
  R3 ∧ R0–R2; R3 fails 0/3 decisively, so the card is refuted regardless of R0–R2.
- **Honest residue: mechanism reads are locally unmeasurable.** R0 (corr(log me, log(1+gt))
  ≥0.20), R1 (CV(f) ≥0.05 + `w_g` value + clamp histogram), R2 (mean f|gt≥300 ≤0.90× mean
  f|gt<50) all require a `w_g`/`f` dump or me/f fields in `val_perimage.json` — **none are in
  the local node dir**. No local `run/` tb → no per-epoch trajectory, no sibling crossover
  table. Stated honestly: **R0–R2 engagement is unmeasurable this session; nothing is
  invented.** Consequence: we cannot distinguish the card's pre-registered failure modes —
  - **F1 quiet-null** (`w_g ≈ 0`, MSE never rewards geometry scaling — the dominant priced
    risk, `idea.md:163`), vs
  - **F2 dead covariate** (corr(log me, gt) < 0.20 — boxes don't track capacity on FSC147), vs
  - **F3 wrong-direction load** (`f` amplified suppression on dense instead of attenuating).
  All three are consistent with the level outcome; without the dump none can be claimed.
- **Untested counterfactuals (successor territory, not incomplete test):** `f` (or any
  capacity actuator) on `fine`/decoder generator path; a train-time count-conditional
  actuator per the offline-probe oracle band (STATE:195). Neither is an untested limb of
  H0026-as-booked.

**Caveat for the record:** local `info.json` `status: "proposed"` / `tested_hypotheses: []`
reflects an unsynced server-side run — `result.json` on disk records the completed futility-
HALTed run; do not read "proposed" as "never ran", and do not hand-edit it (rule 3).

---

## 3. Failure-mode matrix (pre-registered modes, fired?)

| mode | booked signature (`idea.md:163–168`) | evidence available | fired? |
|---|---|---|---|
| **F1** quiet null | `w_g≈0`, CV(f)<0.05, numbers level with parent | **w_g/f dump not present**; level NOT level (+2.73, above all draws) | **indeterminate** (level argues against a pure null, cannot claim) |
| **F2** uninformative covariate | corr(log me, gt) < 0.20 | corr not in dump | **indeterminate** |
| **F3** wrong-direction load / sparse blowup | f loads UP on dense; sparse >5.45 | f not in dump; dense +114.2 outside band; sparse 5.876 over bar (flat vs parent) | **indeterminate / partial circumstantial** (dense up, sparse flat not blown) |
| **F4** clamp saturation | f piles at 0.25/4.0 | histogram not in dump | **indeterminate** |
| **F5** engage-but-wrong-fork | R0–R2 hold, R3 bars fail | R0–R2 not in dump; R3 fails 0/3 | **indeterminate** (bars half visible) |
| **F6** late crossover → ep24 futility HALT | HALT path; verdict = futility refutation + full eval | **YES** — WARN ep16 24.0491, HALT ep24 need 0.3975/ep | **FIRED (fully)** |

Only F6 is directly readable; F1–F5 all gate on the missing `w_g`/`f`/corr dump. The bars +
futility line carry the verdict alone, as §1–2 establish.

---

## 4. Root-cause hypotheses (ranked, each with file evidence)

**#1 — Amplitude-on-suppressive-residual is structurally under-powered (PLAUSIBLE — most
consistent with the corridor; NOT measured on N0027).** The trilogy's third actuator multiplies
the SimPrior evidence — a path the leverage probe measures as weak and net-suppressive
(remove → +29.9% count; sens 1.6 vs fine 15.2). A pre-`out` scalar, even with a published
covariate and learnable sign, reweights a term the decoder barely integrates for mass.
Evidence chain: (i) construction binds `f` to `ev` only (`model.py:261–263`); (ii) probe
premises (`N0026_h0025/idea.md:8,47`, `journal:65`); (iii) corridor: five loci (sign,
substrate, residual-scale, decoder-FiLM, similarity-amplitude) all land +2.7…+3.9 with dense
+114…+161 — a consistent envelope across genuinely different operators points at shared
low-leverage/ wrong-path structure, not five independent bad draws; (iv) N0027 is the *best*
of the corridor (+2.73, dense +114 smallest), consistent with the geometry covariate being
least damaging — and still 0/3 bars.
**What would distinguish:** the `w_g`/`f` dump. CV≥0.05 with correct-direction loading
(R1+R2 true) and counts still failing → confirms under-powered/F5 and hard-closes the
amplitude trilogy. CV<0.05 → F1 quiet-null: MSE never wanted geometry scaling (weaker closure
claim: the covariate was never actually tested in-head). corr<0.20 → F2 dataset-level death
(reusable fact for the paper). **Until the dump, none is claimed.**

**#2 — Quiet-null plus draw-tail (POSSIBLE — second-ranked).** `w_g` stayed ≈0; child ≈
parent + inert scalar; +2.73 a bad draw of the {19.34, 20.70, 21.59} distribution.
Evidence: zero-init makes escape loss-optional (H0009 lesson inverted — gradient is nonzero
but MSE may not reward it, `idea.md:105,163`); draws unbounded above; no `w_g` statistic
exists locally to contradict it.
**Against:** 22.073 exceeds the worst draw by +0.48 and the card's own contamination ceiling
21.6 by +0.49 — the arbitration rule was written to exclude this reading when best sits above
the band. Strained, not excludable without the dump.

**#3 — Dead covariate (WEAKEST but uniquely valuable if true).** If corr(log me, log(1+gt))
< 0.20 on val, the geometry ME never carried count information on FSC147 (annotation boxes may
be too uniform / too few per image: K=3 exemplars). This is the card's F2 and a **dataset-level
fact independent of the verdict**. Evidence: construction only (`model.py:304–308`); R0 was the
registered test — not in dump. Cannot rank above #1/#2 without the corr read.

**Explicitly not claimed:** any `w_g`, CV(f), clamp-saturation, corr, or loading-direction
number; wipe/bucket movement beyond the `val_result.json` slices. The verdict does not need
them: futility HALT outside the draw band + triple-bar miss = refutation as booked.

---

## 5. Follow-up diagnostic reads (when the dump is pulled)

Inference-only, reusing the attribution tooling behind `local/dump_antimatch.py` /
`perimage_diagnostic.md` with `w_g`/`f` instrumentation added for `use_geme` (never dumped):
1. **`w_g` final value + per-image f histogram + CV on val @ best.pth** (R1): decisive for
   F1 vs #1 — the single most valuable missing read.
2. **R0 corr(log me, log(1+gt)) on val**: F2 detector; reusable dataset fact either way.
3. **R2 capacity loading**: mean f on gt≥300 vs gt<50 (bar ≤0.90×) + clamp saturation share
   at 0.25/4.0 (F4): direction and health of the actuator.
4. **Child `val_perimage` paired vs `/home/qkun/cac_backup/N0015_val_perimage.json`**: bucket
   ratios (S-curve movement), dense-block 17-image table — completes what slices only summarize.
5. **w_c / w_p / temp scalars in the same pass**: confirm parent gain stack stayed at its
   operating point (w_c ∈ [0.14, 0.17], temp == 0.07) — closes the N0024 third-state lesson
   for this card too.

---

## 6. Recommendation

**Verdict stands as booked: REFUTED via futility-HALT** (triple bars 0/3: val +2.73, dense
+114.2 outside retrain band, sparse over 5.45; HALT ep24 need 0.3975/ep ≫ 0.12, outside the
card's own draw band). Sub-class **quiet-null vs dead-covariate vs wrong-direction:
indeterminate without the `w_g`/`f`/corr dump — claim none until R0–R2 are read.** Do not
silently retry geometry-ME-on-similarity or any amplitude edit of the SimPrior residual
(§11/§5.11: `contradicts` on H0026 with a genuinely new falsifier if ever revisited).

**Sequence: the three-site amplitude trilogy is complete and 0/3** (similarity / decoder /
post-out — +2.73 / +3.89 / +2.88). Corridor N0023 +2.82, N0024 +3.72, N0025 +2.88,
N0026 +3.89, **N0027 +2.73**. Next-card direction per STATE: legal count-conditional
train-time actuator on the generator path (offline oracle band −2.3…−7.5, STATE:195), or
ExDiv only if forced; pull the `w_g`/f/corr reads when convenient for the paper-grade
collapse corpus — they do not block the verdict or the next card.

**Noise framing carried:** single-seed bet, draws {19.34, 20.70, 21.59} sd≈1.13; 22.07 is
above the worst draw; the refutation rests on the pre-registered bars + futility rule +
dense/mid structure, not on any mechanism read this document does not have.

---

**Files written:** this document only —
`tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014/N0027_h0026/feedback/diagnostic.md`.
No commits, no ssh, no ledger / `info.json` / `quant.md` edits by this role.

# Qualitative feedback — N0027_h0026 (`use_geme` GemeCal, child of N0015_h0014)

Text-log analysis per AGENTS §3/§5. Sources read this session (all local, no ssh, no commit):
node `idea.md` / `model.py` / `config.toml` / `result.json` / `info.json` / `val_result.json` /
`test_result.json` / `val_perimage.json`; parent backup `/home/qkun/cac_backup/N0015_val_result.json`;
style gold `../N0025_h0024/feedback/{qual,diagnostic}.md` and `../N0026_h0025/feedback/{qual,diagnostic}.md`;
draft `local/research/h0026_idea_DRAFT.md`; leverage/S-curve premises `local/research/reexamination_20260922.md`.

**Artifact caveat (honesty §5.5/§5.7):** no `w_g` value, no per-image `f` histogram, no
corr(log me, gt) field, no local `run/` tb under the node. Therefore every R0/R1/R2 mechanism
read below is reported **not in dump** — neither claimed nor reinterpreted. Slices in §1 come
from the pulled `val_result.json`/`test_result.json`.

Verified outcome:

| quantity | booked bar | N0027 | vs parent (19.3431 / 330.81 / 5.814) |
|---|---|---|---|
| EMA val MAE | ≤19.0431 | **22.0730** (`val_result.json`; best 22.0878 @ep24 `result.json`) | **+2.73** |
| dense gt>500 (n=17) | <330.81 | **445.05** | **+114.2** |
| mid 50–500 (n=374) | (guard) | **41.61** | +4.11 |
| sparse gt<50 (n=895) | ≤5.45 | **5.876** | +0.062 (over bar) |
| test MAE (n=1190) | — | **19.804** | parent test 19.066 (STATE) |
| futility | ep16 WARN / ep24 HALT | **WARN 24.0491**, **HALT** need 0.3975/ep, best 22.0878 | outside (20.0, 21.6) band by +0.49 |
| R0 corr(log me, gt) | ≥0.20 | **not in dump** | — |
| R1 CV(f), w_g, clamp hist | ≥0.05 | **not in dump** | — |
| R2 f-load gt≥300 ≤0.90× gt<50 | ≤0.90× | **not in dump** | — |

---

## 1. What GemeCal was supposed to do

The booked BECAUSE (`idea.md:13,30`) has two joints:

1. **Diagnosis (the scale-free similarity pathway).** The SimPrior readout is
   `ev = cat[top1∈[-1,1], cons]` → cellcal mean-1 relative gain → peakcal ≈1 gain → zero-init
   linear `out` — no term expresses a per-image order-of-magnitude ("this image needs ~500
   counts"). The leverage probe measures the residual **net-suppressive** (remove → +29.9%
   count); the measured S-curve (gt7–10 ratio 1.40 → gt138+ 0.685, mid50–300 0.86 carrying 44%
   AE) is monotone in gt — an order-of-magnitude capacity miss. CACViT (AAAI'24) independently
   names softmax normalization destroying order-of-magnitude on similarity scores and multiplies
   exemplar-box magnitude embedding back in — the published ancestor of this actuator.
2. **Remedy (GemeCal).** One zero-init scalar `w_g`; `f = clamp((me/32)^w_g, 0.25, 4)` from
   annotation exemplar-box areas (`me = mean_K(S²/(w_k·h_k))`, stop-grad geometry); threaded into
   all four `simprior(...)` call sites; inside `SimPrior.forward` after the unchanged
   cellcal/peakcal gains and before the unchanged zero-init `out`: `ev ← ev · f`
   (`model.py:177–194, 261–262, 295–310, 346–347`). +1 param, +0 RNG draws, step-0 identity
   (`w_g=0 ⇒ f=1`). Then the triple bars: val ≤19.0431 AND dense <330.81 AND sparse ≤5.45.

**Implementation as booked (diff-verified this session):** config delta is exactly
`use_geme = true` (`config.toml:30`); `use_simprior/cellcal/peakcal` all true; `geme_f` threaded
through **all four** call sites (N0023 exclusive-dispatch bug did NOT recur); multiply sits
after both gain blocks, before `return self.out(ev)`; GemeCal constructed after PeakCal
(append-only, rule 13); temp-pin path untouched. **The card executed what it booked.**

**Outcome vs predictions:** every measurable bar missed — val +2.73, dense +114.2, sparse 5.876
(over 5.45), futility HALT at ep24 outside the card's own draw band. R0/R1/R2: **not in dump** —
the mechanism neither confirmed nor cleared locally; the card needed R3 ∧ R0–R2, and R3 alone
already fails 0/3 as booked.

## 2. Honest design assessment: published covariate, still the weak-leverage path

The three-site localization trilogy booked H0024 (post-out residual), H0025 (decoder FiLM), and
H0026 (similarity-side pre-out amplitude). H0026 is the **cleanest of the three designs** —
published covariate (CACViT ME), +1 param, geometry-only (model-independent, no response `z`),
learnable sign, no mask/substrate/decoder edit — and it still lands in the same failure envelope:

- **The actuator still multiplies the measured weak-suppressive residual.** Leverage probe
  premises (file-recorded, `N0026_h0025/idea.md:8,47`, `journal:65`): `fine=0` → −91.6% count,
  sens 15.2; remove SimPrior residual → **+29.9%** count, cond sens 1.6 (~10× below fine). `f`
  scales `ev` pre-`out`; it cannot change spatial phase, cannot touch `fine`, cannot inject mass
  onto the generator. Even a perfectly-learned capacity loading moves a term the decoder barely
  integrates for mass — the same structural handicap N0025's α carried, now with a better
  covariate on the same path.
- **But the covariate question is genuinely new and stays open.** Unlike H0024's response-z
  (flat in-box self-match) the geometry `me` varies by construction across the S-curve — so F1
  quiet-null is less structurally forced here. Without the `w_g`/`f` dump we cannot say whether
  MSE simply never rewarded capacity scaling (F1), whether `corr(log me, gt) < 0.20` killed the
  covariate (F2), or whether `f` loaded in the wrong direction (F3). The level (22.07, worst-
  draw-adjacent) is consistent with all three; claiming a specific mode would be invention.
- **Sparse moved the wrong way for the R2 story** (5.876 vs parent 5.814, +0.062 — tiny but
  positive, unlike N0023/N0025's slight improvements). If `f` engaged at all, there is no
  observable sparse-side benefit; if it didn't, sparse is draw noise. Undecidable without R1.

**Design-quality positives worth keeping:** single pluggable switch; one-line config delta;
strongest append-only form (`torch.zeros(())` draws no RNG); step-0 double identity (f=1 and
zero-init `out`); no second grad path, no shared projection, no temp/gain-form/mask/substrate
edit; novelty gate green (top sim 0.601 vs H0025); falsifier bars specific and parent-bound;
R0 (corr information gate) is a genuinely falsifiable dataset-level disjunct, unlike the
vacuous construction-true 4th disjuncts of H0022/H0023. The card was well-formed; it bet on the
last untested amplitude cell of a path six prior cards had already bracketed.

## 3. Comparison to sibling verdicts (N0023–N0026)

| | N0023 | N0024 | N0025 | N0026 | **N0027** |
|---|---|---|---|---|---|
| locus | sign-mask on ev | substrate swap on ev | post-out ×α (response) | decoder FiLM (γ,β) | **pre-out ×f on ev (geometry)** |
| val MAE | 22.163 (+2.82) | 23.062 (+3.72) | 22.213 (+2.88) | 23.232 (+3.89) | **22.073 (+2.73)** |
| dense gt>500 | 458.69 (+127.9) | 491.81 (+161.0) | 454.05 (+123.2) | (pending) | **445.05 (+114.2)** |
| sparse | 5.703 | 5.972 | 5.779 | (pending) | **5.876** |
| mechanism dump | wipes known 0/11 | wipes 0/11 | alpha not in dump | γ not in dump | **w_g/f not in dump** |
| futility HALT ep24 | yes | yes | yes | yes | **yes (need 0.3975/ep)** |
| verdict | REFUTED | REFUTED | REFUTED | REFUTED | **REFUTED** |

Reading across: N0027 is the **best member of the corridor** (+2.73, dense blowup smallest at
+114) — the geometry covariate is the least damaging of the five interventions — yet it still
misses all three bars and sits above every historical draw. The trilogy is now **complete and
0/3**: similarity-side amplitude (this card), decoder consumption (N0026), post-out residual
(N0025) all futility-HALT at ep24 in the +2.7…+3.9 envelope. Sparse once again never decided
the verdict (standing parent-infeasible 5.45 bar); dense + val did.

**Honest residue:** without the `w_g`/`f` dump we cannot distinguish F1 quiet-null (MSE never
rewarded geometry scaling — the card's dominant priced risk, `idea.md:163`) from F2 dead
covariate or F3 wrong-direction load. The verdict as booked rests on R3 (0/3) + futility
outside the draw band; any claim of a *specific* failure mode without the dump would be
invention. Pulling `w_g` + f-histogram + corr(log me, gt) from `best.pth` remains the single
most valuable missing read — it does not block the verdict.

## 4. Booking-quality notes

- **Booked vs code: MATCH** (line-checked `model.py` vs `idea.md` §2.2/§2.4: clamp [0.25,4.0],
  post-gain pre-out placement, four call-site kwargs, GemeCal after PeakCal). No dispatch bug.
- **Falsifier well-posedness:** three numeric bars specific and parent-bound; R0/R1/R2 are
  empirically falsifiable engagement disjuncts (corr, CV, loading direction) — sound structure,
  simply undecidable without the dump. Sparse ≤5.45 is the standing parent-infeasible bar again
  (parent 5.814) — a footnote, not the story (child 5.876, essentially flat).
- **Process:** local `info.json` still `status: "proposed"` / `tested_hypotheses: []` while
  `result.json` records the completed futility-Halted run — server sync gap (same as N0026;
  rule 3: do not hand-edit). `config.toml` says `augment = false` while launch carried
  `--set augment=true --set futility_bar=19.0431` (in `result.json.cfg_overrides`).
- **Noise framing honored:** same-seed draws {19.34, 20.70, 21.59} sd≈1.13; 22.07 sits above
  the worst draw and outside the card's own (20.0, 21.6) contaminated band — the refutation is
  the pre-registered bars + futility line, not a mechanism read we do not have.

## 5. What this rules in / rules out for the next card

**Rules out.** (a) Geometry ME power-law amplitude on SimPrior `ev` as booked — REFUTED.
(b) The entire (any-source × any-operator × similarity-or-decoder-or-post-out) amplitude trilogy
as the *primary* mass actuator: five loci, one corridor, 0/3 bar clears — amplitude edits of
the suppressive residual / consumption path are empirically inert-to-harmful under joint MSE.
(c) Quiet readings: HALT outside the draw band, every bar missed — no successor may argue
N0027 "ran out of epochs."

**Rules in.** (1) The trilogy's shared failure shifts weight to the offline-probe conclusion
(STATE:195): the path to val <18 needs a **train-time count-conditional actuator** on the
generator side (gt-binned scale oracle 17.06), not another scalar on evidence/decoder output.
(2) R0 (`corr(log me, gt)`) is still worth measuring once for the paper-grade dataset fact —
it is independent of this card's verdict. (3) Any successor re-encoding ME-on-similarity or
geometry-capacity scaling must book as `contradicts`-evidence on H0026 with a genuinely new
falsifier (§5.11). (4) Operational lesson repeated from N0025/N0026: ship the mechanism dump
(`w_g`, f-histogram, corr) with the run so R0–R2 are locally decidable — this card's biggest
analysis gap is the missing dump, not the missing bars.

---

**Files written:** this document only — `feedback/qual.md`. No commit, no ssh, no ledger /
`info.json` / `quant.md` edits by this role.

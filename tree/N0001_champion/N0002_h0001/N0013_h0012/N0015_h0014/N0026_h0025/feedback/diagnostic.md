# Diagnostic — N0026_h0025 (H0025 `use_capfilm` CapFilm) — FAILURE (futility HALT)

- **id/parent**: N0026_h0025 ← N0015_h0014 (live, val 19.3431 @ep32 v2).
- **mechanism**: ~25k-param zero-init FiLM conditioning of the DensityDecoder — `CapFilm`
  MLP maps GAP(fine) ‖ log(ME) (ME = mean_K(S²/(w_k·h_k)), exemplar-box geometry) to
  per-channel (γ, β), applied `h ← h·(1+γ)+β` on the block output **pre-final-1×1-head,
  pre-softplus**; one boolean `use_capfilm`, CapFilm appended after every parent module,
  step-0 identity via zero-init last Linear (`idea.md:4`; `model.py:139–144, 293–315,
  351–356`).
- **outcome**: **futility-HALTED ep24** — `journal/events.jsonl:70`: best **23.232**,
  need ~0.52/ep vs threshold 0.12 → HALT; val level **+3.89 vs live 19.3431** (bar 19.0431
  missed by ~4.19). Launch record `journal:68` (config_sha `4b56e8e656961e0c`, model_sha
  `917596dd5b2df04e`, `--set augment=true --set futility_bar=19.0431`, green smoke).
  **Slice eval UNMEASURABLE this session**: no `result.json`, no `val_result.json`, no
  `test_result.json`, no γ dump under the local node dir (inventory: `idea.md`, `model.py`,
  `config.toml`, `info.json`, empty `feedback/`); local `info.json` still `status: "proposed"`
  (server-side run not synced). Full val/test eval was launched server-side (`journal:70`) —
  dense/mid/sparse may become available later; **nothing is estimated here**.
- **constraints**: frozen backbone, head-only, seed 20260830 fixed, v2 protocol, AGENTS §5.16
  futility (ref slope 0.06/ep, margin 2.0 → threshold 0.12; ep16 WARN / ep24 HALT), AGENTS §11
  no-repeat register, rule 11 (no silent retry of refuted mechanisms). No commits, no ssh
  performed for this document.

Verified locally from node artifacts read this session: `idea.md`, `model.py` (line-checked vs
parent `../model.py` decoder path and vs `idea.md` §2.2/§2.4), `config.toml` (delta = exactly
`use_capfilm = true`; `use_simprior/cellcal/peakcal` all true), `info.json`; parent backup
`/home/qkun/cac_backup/N0015_val_result.json` (mae 19.3259, dense 330.8131, bias −12.41);
`journal/events.jsonl:66–71`; `local/research/reexamination_20260922.md` (leverage probe,
S-curve); sibling diagnostics `../N0023_h0022/feedback/diagnostic.md`,
`../N0024_h0023/feedback/diagnostic.md`, `../N0025_h0024/feedback/diagnostic.md`.

---

## 1. Failure taxonomy

**Primary class: futility-HALT level-refutation on the first decoder-consumption probe —
val level +3.89 above live, outside every observed draw; sub-class (quiet-null vs
engaged-harmful) INDETERMINATE pending the γ dump.** Not operator-wrong (dispatch as-booked,
§2), not gate-too-strict (HALT beyond every observed draw), not budget (`futility`, not
wall-clock — journal verdict line, launch budget 1800s standard).

**(a) What part of +3.89 can be noise.** Same-seed draws of the identical N0015 config:
{19.3431 live, 20.697 N0021 retrain, 21.5928 exact repro}, mean 20.544, sd ≈ 1.13 (sibling
diagnostics §1a, `perimage_diagnostic.md:231`). Decomposition of best 23.232 vs live
(23.232 − 19.3431 = +3.889):
- **≈ +1.20 is parent draw-luck** (20.544 − 19.3431) — live parent is the minimum draw.
- **≈ +2.69 is the child's deviation above the draw-mean** — (23.232 − 20.544)/1.13 ≈ **2.4σ**
  of a single draw; under exchangeable sd(child−parent) ≈ 1.60, +3.89 ≈ 2.4σ. Under the
  HANDOFF ≥+2 "credible" heuristic this is **real, not noise**; it also exceeds the worst
  historical draw (21.59) by **+1.64**.
- Decisive noise-invariant reads below carry the verdict regardless; slice reads are unavailable
  (§2), so unlike N0023/24/25 the dense/mid structural arguments **cannot** be made yet.

**(b) Futility arithmetic (exact).** Booked: `--set futility_bar=19.0431` (`idea.md:7,184`;
`journal:68`). Threshold at ep24 = bar + 0.12×(32−24) = 19.0431 + 0.96 = **20.0031**.
best 23.232 → required slope (23.232 − 19.0431)/8 = **0.5236/ep > 0.12** → **HALT**,
**+3.229 above the ep24 ceiling**. The card's own arbitration: HALT in (20.0, 21.6) is
draw-contaminated — **23.232 sits above 21.6 by +1.63**, i.e. outside the band by the card's
own rule (`idea.md:184`) → not draw-contaminated; R1/R2 were the registered tiebreakers and
are unavailable (§4), so the verdict is futility + the val-level bar as booked. Even
extrapolating the **parent's own** late slope (−0.28/ep) from 23.232 ends ≈20.9 @ep32 — still
**+1.86 above** the 19.0431 bar; the HALT saved ~8 epochs without changing the verdict.

**(c) Classes explicitly ruled out.**
- *Gate-too-strict / early stop*: need 0.5236/ep ≫ 0.12; extrapolation above still misses.
- *Operator-wrong / dispatch bug*: code matches booking line-for-line (§2); SimPrior cascade
  intact, cellcal/peakcal reachable on every image (no N0023 recurrence); smoke green
  (`journal:68`); RNG append-after-PeakCal honored (rule 13); step-0 identity by zero-init.
- *Budget timeout*: this is a futility halt at ep24, not τ_max; and local `info.json` "proposed"
  is a sync gap, not a run status.
- *Draw-noise as sole cause*: contradicted by +3.89 ≈ 2.4σ above draw-mean, +1.64 above the
  worst draw, and a futility HALT beyond the card's own contaminated band — though the final
  quantitative share of noise cannot be slice-decomposed until `val_result.json` lands.
- *Re-encode of a closed family*: CapFilm touches no residual content, no gain form, no mask,
  no substrate, no post-decoder term, no scalar output calibration — different locus from
  H0016–H0024; the failure, whatever its sub-class, is **not** a rule-11 re-hash.

**Taxonomy verdict: futility-HALT level-refutation, sub-class indeterminate.** The
consumption-interface fork's first registered probe missed every observable bar by
mechanism-level margin; noise explains ≈+1.20 of +3.89, the rest is structural-or-draw-tail
that only the γ dump and the slice eval can split. Comparable in class to N0023/24/25 (all
ep24 HALT, all +2.8…+3.9) but at a **new locus** and the **largest gap of the corridor**.

---

## 2. What was actually tested — dispatch vs booking, partial-test residue

**Facts (code, line-checked this session).**
- `config.toml:27–30`: `use_simprior/cellcal/peakcal/canchor`→ (parent flags) all true plus
  exactly one new line `use_capfilm = true` — filtered delta is the single switch.
- FiLM site: parent `return F.softplus(self.head(self.block(x)))` (`../model.py:139–140`) →
  child applies `h·(1+γ[:, :, None, None]) + β[:, :, None, None]` on `h = self.block(x)`
  **before** `self.head`/softplus (`model.py:139–144`). `DensityDecoder.__init__` untouched.
- `CapFilm` (`model.py:293–315`) exactly as booked: GAP, box-area ME, `log(me.clamp(1, 1e4))`,
  129→64→256 MLP, last Linear zero-init → (γ, β)=(0,0) at init.
- Call site (`:192–195`): γ,β computed only when flag on and module exists; SimPrior cascade
  (`:181–191`) runs unchanged before it. Attach at Counter end after PeakCal (`:351–356`);
  temp-pin (`:349–350`) untouched. Flag non-module (`:169`).

**Assessment — is H0025 only partially tested?**
- **On the booked design: substantially tested for the level verdict.** Every code clause
  executed across 24 epochs under the booked protocol; futility fired as specified; the val
  level (best 23.232) misses the bar by +4.19 and sits outside the draw band. The card needed
  R4 ∧ R1; the level half of R4 fails decisively, so the card is refuted regardless of R1.
- **Honest residue: mechanism reads and slices are locally unmeasurable.** R1a (CV of
  mean_c(γ) ≥0.05), R1b (capacity loading mean γ|gt≥300 ≥1.5× mean γ|gt<50), R1c (channel
  split), R2 (S-curve bucket ratios, wipe movement), R3 (cond×2 leverage re-probe) all require
  a γ dump and/or child `val_perimage.json` — **none are in the local node dir**. Dense/mid/
  sparse slices await the server eval (`journal:70` launched it). Stated honestly: **engagement
  R1–R3 and all slices are unmeasurable this session; nothing is invented.** Consequence: we
  cannot distinguish the card's pre-registered failure modes —
  - **F1 quiet-null** (γ≈0 everywhere; MSE never rewards capacity scaling — the card's
    dominant priced risk, `idea.md:172,181`), vs
  - **F2/F3 engaged-harmful** (γ moved with spread, but wrong-direction capacity loading
    over-gains sparse/dense, or the decoder co-adapted and the affine distorted the trained
    operating point).
  The *level* argues which is more consistent (§4); without the dump both remain open.

**Caveat for the record:** local `info.json` `status: "proposed"` reflects an unsynced
server-side run — the launch and verdict exist in `journal/events.jsonl:68,70`; do not read
"proposed" as "never ran", and do not hand-edit it (rule 3).

---

## 3. Failure-mode matrix (pre-registered modes, fired?)

| mode | booked signature (`idea.md:170–177`) | evidence available | fired? |
|---|---|---|---|
| **F1** quiet null | γ≈0, CV<0.05, numbers level with parent | **γ dump not present**; level NOT level (+3.89, above all draws) | **indeterminate** (level argues against, cannot claim) |
| **F2** uniform overgain / sparse blowup | γ>0 everywhere; sparse >5.45 | γ not in dump; sparse slice UNMEASURABLE locally | **indeterminate** |
| **F3** dense blowup | dense ≥330.81 | dense slice UNMEASURABLE locally | **indeterminate** |
| **F4** engage-but-wrong-fork | R1+R3 hold, R4 bars fail on level/mid | R1/R3 not in dump; level fails | **indeterminate** (bars half visible) |
| **F5** late crossover → ep24 futility HALT | HALT path, verdict = futility refutation + full eval | **YES** — HALT ep24, best 23.232, need 0.5236/ep | **FIRED (fully)** |
| **F6** H0024 collateral | N0025 separate child of N0015 | N0025 finished/refuted before N0026 launch (`journal:67,68`) | not applicable — clean |

Only F5 is directly readable; F1–F4 all gate on the missing γ dump and/or the pending slice
eval. The futility line + val level carry the verdict alone, as §1–2 establish.

---

## 4. Root-cause hypotheses (ranked, each with file evidence)

**#1 — Engaged-but-harmful / wrong-direction actuator (PLAUSIBLE — most consistent with the
level; NOT measured).** CapFilm's affine escaped zero and *moved* the trained decoder's
operating point in a direction MSE did not reward (or rewarded on sparse at dense's expense),
producing structural degradation beyond the draw band.
Evidence chain: (i) FiLM site is on the generator path proper — `h` feeds `self.head`
directly (`model.py:139–144`), probe: `fine=0` → −91.6% count, sens 15.2
(`N0026_h0025/idea.md:8,47`, `journal:65` premises) — a *nonzero* γ here has ~10× the
leverage of anything on the cond/residual path, so a wrong-signed or uniformly-positive γ is
*capable* of producing +2…+4 MAE damage, unlike the quiet residual cards; (ii) gradients at
init are nonzero by construction (`∂L/∂γ = ∂L/∂h ⊙ h`, `idea.md:112`) — nothing structurally
prevents escape from zero; (iii) best 23.232 exceeds every observed draw by +1.64 and sits
2.4σ above draw-mean (§1a) — the signature of an *added harmful DOF*, matching the historical
"active but harmful" precedent (H0003 padapt: +3.00, mechanism corruption, STATE:7); (iv) the
corridor: three prior multi-site interventions all produced +2.8…+3.7 with structural slice
damage — a fourth at a *higher-leverage* site landing at +3.89 is consistent with the same
"engaged and wrong" family rather than with pure luck.
**What would distinguish:** the γ dump. If **CV(mean_c(γ)) ≥0.05 with capacity loading**
(R1 true) → engaged-harmful confirmed: the actuator fired, direction or magnitude wrong;
successor must change sign-constraint or conditioning source, not re-tune γ. If loading runs
**backwards** (mean γ|gt≥300 < mean γ|gt<50) → F2 specifically (ME/GAP not discriminative
for "need more counts"). **Until the dump, this is the more-consistent hypothesis, not a
measurement.**

**#2 — Quiet-null plus draw-tail (POSSIBLE — consistent, second-ranked).** γ stayed ≈0;
the child behaved as parent + ~25k inert params, and the +3.89 is a bad draw of the same
distribution that produced {19.34, 20.70, 21.59}.
Evidence: (i) zero-init last Linear means escape-from-zero is *loss-optional* — MSE on 69%
gt<50 images may never reward capacity scaling (card's own dominant risk, `idea.md:181`;
same family as H0016/H0017 quiet-nulls on residual); (ii) draws are unbounded above — 2.4σ
is strained but not impossible for a single draw (sd≈1.13, sibling diagnostics §1a);
(iii) no γ statistic exists locally to contradict it.
**Against (why ranked #2):** a pure quiet-null should land *inside* the parent draw cloud;
23.232 is above its maximum by +1.64 and above the card's own contamination ceiling 21.6 —
the card's arbitration rule (`idea.md:184`) was written exactly to exclude this reading when
best sits above the band. Still possible; not excludable without the dump.
**What would distinguish:** same R1 read — **CV <0.05 or flat γ** → F1 quiet-null confirmed;
then the closure claim weakens honestly ("consumption FiLM never actually tested in-head"),
exactly the residue N0025's diagnostic recorded for α.

**#3 — Decoder co-adaptation / renormalization absorbed the affine (PLAUSIBLE mechanism
subset of #1/#2; design-level).** Even with γ moving, 24ep of joint training can retune
`self.head`/`block` to cancel a slowly-varying per-channel affine — the consumption-side
analog of the "MSE normalizes away perturbations" dynamics that killed H0016/H0017
(`reexamination_20260922.md:8–10`). Placement is post-GN (`model.py:140–143`), so GN does
*not* provably erase γ at the site — but co-adaptation over epochs can. Evidence: GroupNorm
inside `self.block` (`model.py:131–133`) normalizes before FiLM, leaving training-time
absorption as the only renormalizer; family precedent of quiet-nulls under joint MSE. This
hypothesis predicts mid-train γ drift toward 0 or toward a constant — **distinguishable only
by a γ trajectory/histogram dump** (epoch-wise CV), which the run did not produce.

**#4 — ME/GAP covariate inadequacy (WEAKEST; sub-case of F2).** log-ME is 1-D and clamped
(`model.py:311–312`), possibly collinear with GAP(fine) so the 129→64→256 MLP learns a global
gain in per-channel clothing (H0002/H0010/H0015 scalar-family shadow). Evidence: construction
(`model.py:307–315`); R1b capacity-loading read was the registered test — not in dump. Cannot
be ranked above #1/#2 without γ; would show as CV≥0.05 with **no** gt-loading separation.

**Explicitly not claimed:** dense/mid/slice regressions (UNMEASURABLE locally — unlike
N0023/24/25 where dense +123…+161 outside band [300,361] carried structural proof); wipe
movement; any γ/α number. The verdict does not need them: futility HALT outside the draw band
+ val level miss = refutation as booked.

---

## 5. Follow-up diagnostic reads (when dump/eval land)

Inference-only, reusing the attribution tooling behind `local/dump_antimatch.py` /
`perimage_diagnostic.md` with γ instrumentation added for `use_capfilm` (never dumped by the
run):
1. **γ histogram + CV of mean_c(γ) on val @ best.pth** (R1a): decisive for F1 vs #1 — the
   single most valuable missing read.
2. **Capacity loading**: mean mean_c(γ) on gt≥300 vs gt<50 (R1b, bar ≥1.5×) + per-channel
   positive-γ share (R1c): direction of the actuator (F2 vs healthy-loading-but-still-failed).
3. **Epoch-wise γ CV trajectory** (tb-side if any tensor logged; else two-point init vs final):
   tests hypothesis #3 (co-adaptation decay toward constant/zero).
4. **Child `val_perimage` paired vs `/home/qkun/cac_backup/N0015_val_perimage.json`**: bucket
   ratios (R2 S-curve: does gt≥300 ratio move toward 0.75?), dense-block 17-image table,
   sparse slice — completes the pending server eval once `val_result.json` is pulled.
5. **cond×2 leverage re-probe on trained best.pth** (R3): parent ratio 0.824; did consumption
   dynamics move at all? Distinguishes "engaged but ineffective" from "engaged and harmful".
6. **w_c / w_p / temp scalars in the same pass**: confirm parent gain stack stayed at its
   operating point (w_c ∈ [0.14, 0.17], temp == 0.07) — code says it should; a read closes
   the N0024 third-state lesson for this card too.

---

## 6. Recommendation

**Verdict stands as booked: REFUTED via futility-HALT** (best 23.232 @ep24, need 0.5236/ep
≫ 0.12, +3.89 vs live outside the card's draw-contaminated band; level bar missed by ~4.19;
slice bars pending server eval but unnecessary for the verdict). Sub-class **quiet-null vs
engaged-harmful: indeterminate without the γ dump**; the level makes **engaged-harmful the
more-consistent reading** (§4 #1), quiet-null second (§4 #2) — **claim neither until R1 is
read**. Do not silently retry CapFilm (rule 11: `contradicts` on H0025 with a genuinely new
falsifier if the decoder-amplitude family is ever revisited).

**Sequence unchanged:** the decoder/consumption fork has spent its first probe and missed;
the similarity-side direction remains open per `reexamination_20260922.md:43–47` and is
**already booked** — proceed with **H0026 GemeCal (`N0027_h0026`)**, geometry ME power-law
scale on SimPrior `ev` (`journal:71`, `local/research/h0026_idea_DRAFT.md`): orthogonal site,
+1 param, completes the three-site localization trilogy (similarity / decoder / post-out
residual — the latter two now both measured-negative at +3.89 and +2.88 respectively).
Exemplar-distinctness remains the registered encoding-layer fallback. When the server eval
lands, pull `val_result.json`/`test_result.json` and append slices as a quant addendum;
pull/build the γ dump for the F1-vs-F1 split — it does not block the verdict or N0027.

**Corridor context for synthesis:** N0023 +2.82, N0024 +3.72, N0025 +2.88, **N0026 +3.89** —
four futility-HALTs at ep24, four loci (sign / substrate / residual-scale / decoder-FiLM),
one corridor. Noise framing carried: single-seed bet, draws {19.34, 20.70, 21.59} sd≈1.13;
23.23 is above the worst draw; the refutation rests on the pre-registered futility rule +
val level, not on any mechanism read this document does not have.

---

**Files written:** this document only —
`tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014/N0026_h0025/feedback/diagnostic.md`.
No commits, no ssh, no ledger / `info.json` / `quant.md` edits by this role.

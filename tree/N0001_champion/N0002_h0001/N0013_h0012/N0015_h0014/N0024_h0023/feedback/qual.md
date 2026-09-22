# Qualitative feedback — N0024_h0023 (`use_antimatch`, child of N0015_h0014)

Text-log analysis per AGENTS §9/§3. Sources read directly (all local, no ssh, no commit):
node `idea.md` / `model.py` / `config.toml` / `info.json`; parent `N0015_h0014/model.py` +
`config.toml` (diffed this session); booked text `memory/hypotheses.jsonl:42,44`;
`journal/events.jsonl:59,62`; `local/research/h0023a_idea_DRAFT.md`;
`local/research/perimage_diagnostic.md` appendix (N0015 wipe table);
`local/dump_antimatch.py` (read definitions); sibling feedback set
`../N0023_h0022/feedback/{qual,causal,diagnostic,quant}.md`; `HANDOFF.md`, `STATE.md:145–161`,
`/home/qkun/cac_backup/N0015_val_result.json`.

**Artifact caveat (honesty gate §5.5/§5.7):** the handoff lists `result.json`,
`val_result.json`, `val_attr.json` as local inputs, but the node dir contains **none of them**
(`find` shows only `info.json`; `dump_antimatch.py:188` writes `val_attr.json` beside the node —
it ran server-side). Every outcome number below therefore cites the Lead-recorded evidence notes
`memory/hypotheses.jsonl:44` / `journal/events.jsonl:62` (both read this session), which quote the
dump. Per-image independent recompute (the 11 ratios, the 19-vs-11 wipe count) was **not**
possible locally; those are taken as Lead-verified, and the missing sync is logged as a process
nit (§5).

Verified outcome (all from `hypotheses.jsonl:44`):

| quantity | booked bar | N0024 | vs parent | vs sibling N0023 |
|---|---|---|---|---|
| EMA val MAE | ≤19.0431 (−0.30 vs 19.3431) | **23.0618 @ep24, futility HALT, need 0.525/ep** | **+3.72** | +0.90 worse (22.163) |
| dense gt>500 | <330.81 | **491.81** | **+161.0 (+49%)** | +33.1 worse (458.69) |
| mid 50–500 | (guard) | **42.64** | **+5.14** (parent 37.50) | +0.93 worse (41.71) |
| sparse gt<50 | ≤5.45 | **5.972** | **+0.16 — worse than parent 5.814 too** | +0.27 worse (5.703) |
| wipes gt≥300 ratio<0.5 | no new (R3) | **19 vs 11** | +8 new | — |
| R2 rescue | ≥7/11 at ≥0.65 | **0/11** (7/11 correctly routed, evg_sum +18.8k; 4/11 not routed) | — | 0/11 (same) |
| route fire | minority ~2% (draft:4,133) | **531/1286 = 41%; KEEP 0.63; MID 0.55** | — | n/a (no route) |
| mechanism reads | dispatch fix + parent gain | **w_c=0.0282 (alive), gain_on_neg 1.017, temp 0.07, w_p 0.00643** | parent band w_c 0.148–0.166, gain_on_neg 1.076–1.098 | w_c=0 (bypassed) |

---

## 1. Did the BECAUSE hold? — three claims, judged separately

The booked BECAUSE (`idea.md:9` = `hypotheses.jsonl:42`) decomposes into three claims:

### Claim A — "the dense wipe is a measured anti-match artifact" (anti-phase IS the substrate)

**Measurement half: held (as pre-registered). Causal half: refuted by this run.**

- The measurements are real and were already on the books pre-launch: parent dump WIPE group
  top1_mean −0.258, neg_ev_frac 0.667, ev_sum −4924 → evg_sum −5301, vs KEEP +0.141/+2455/+3022
  (`perimage_diagnostic.md` appendix; restated `idea.md:13–14`, `draft:34–41`). Nothing in N0024
  contradicts the numbers.
- What N0024 refutes is the word **"artifact ... substrate" as causal**: on the 7 booked wipes the
  gate correctly routed, the anti-phase substrate was *replaced* by construction-positive evidence
  (post-swap evg_sum ≈ **+18.8k**) and the wipes stayed wiped — ratios 0.05–0.48, **0/11**
  (`hypotheses.jsonl:44`). H0022 had already shown the gain's deepening is only a ~377-AE amplifier,
  not the cause (`N0023/feedback/qual.md:76–82`); H0023 now shows the anti-phase evidence's *sign*
  is not the cause either. Anti-phase is a **correlate of the wipe, not its operative substrate**.
- Selector stability also cracked: 4/11 booked wipes (840, 935, 3482, 3487) sat on the *positive*
  side at the deployed checkpoint (top1 +0.035, +0.103, +0.060, +0.097 — near-zero margins), so
  even the target set's membership in "anti-match" is checkpoint-marginal (F5-family; per-image
  parent top1 not locally verifiable to say whether these flipped).

**Judgment: premise TRUE as measurement, causal claim FALSE.**

### Claim B — "the energy field is the only in-phase mass signal the gain/decoder interface accepts, converting the identical confirmed gain from negative-deepener to positive mass source"

**Arithmetic half: true by construction. Outcome half: refuted. "Identical gain": source-true only.**

- Construction worked exactly as designed: on routed images `ev → cat(mhat, mhat) ≥ 0`
  (`model.py:237–243`) then the parent cellcal gain multiplies it (`model.py:244–249`), giving the
  recorded routed evg_sum **+18.8k**. That number is not a learning result: 2·96² · ⟨gain⟩ =
  18432 · ≈1.02 ≈ **18.8k** — the read certifies the route fired and the arithmetic held, nothing
  more (see §5 on R1's vacuity).
- The conversion to **counts** did not happen: 0/11 rescued on the routed 7, and the 4 un-routed
  also stayed — R2 0/11 vs ≥7 booked (`idea.md:9`, read design `draft:118–119`).
- "The identical confirmed gain" held only at source level. Runtime: **w_c = 0.0282** vs the
  parent band 0.148–0.166 (`STATE.md:145–161`, `idea.md:14`), gain_on_neg **1.017** vs parent
  1.076–1.098 (`dump_antimatch.py:12,150` defines it as mean gain over negative pre-swap cells) —
  the actuator ran **~5× weak** (`hypotheses.jsonl:44`). Byte-identical code lines
  (diff-verified, §5) did not pin the learned operating point: routing 41% of val through a
  detached substrate changed the gradient/input statistics of the shared `out`/`w_c`, and `w_c`
  collapsed (causal link plausible, not ablated; the measurement itself is confirmed).
- And even a parent-strength gain multiplies a **mean-1, fixed-total** field — see §4: there is no
  count magnitude for it to amplify.

**Judgment: REFUTED as written (sign conversion delivered, mass conversion did not; gain not
identical in effect).**

### Claim C — "without touching the healthy majority"

**Construction half: true. Empirical half: decisively false — the majority is what got touched.**

- Route-off path is byte-parent: `(1 − anti) * ev` identity (`model.py:243`), stop-grad decision
  (`model.py:241`), non-routed gradient untouched — as booked (`draft:62`).
- But the gate fired on **531/1286 = 41% of ALL val**, **63% of healthy dense KEEP**, **55% of
  MID** (`hypotheses.jsonl:44`). By arithmetic on the dump's group sizes (KEEP = 38−19 = 19 at
  gt≥300, MID = 353 at 50≤gt<300, `dump_antimatch.py:167–169`), the unaccounted residual implies
  **≈1/3 of the 895 sparse images routed too** (exact split needs the non-local `val_attr.json`).
  The "~2% minority" (`draft:4,24,133`) missed by ~20×.
- Cost, in the booked guard's own terms (R3, `idea.md:35`, `draft:120`): **19 vs 11** wipes at
  gt≥300, dense +161 (+49%), mid +5.14, sparse +0.16 — **all three slices worse than parent**;
  R3 refuted. Each false positive pays double: forward (both evidence channels of a healthy image
  replaced by detached `mhat`) *and* backward (zero gradient to qproj/kproj/fine from it,
  `model.py:238,241`), plus the shared-parameter drift of Claim B hitting even non-routed images
  (global gain_on_neg 1.017 < parent 1.08).

**Judgment: REFUTED.**

### Verdict on the BECAUSE

**Does not hold.** 2 of 3 claims refuted outright (B outcome, C scope); the surviving claim (A)
survives only as the pre-run *measurement* it always was, having lost its causal reading to this
run and to the sibling's. The card executed what it booked and the world still moved the wrong
way on every bar plus R2/R3 — a clean mechanistic refutation, not a quiet null.

---

## 2. Which booked failure mode fired

Pre-registered modes: F1 route-ineffective, F2 route-too-coarse, F3 MSE-veto
(`idea.md:35`, `draft:125–129`; F5 `draft:129`).

| mode | booked signature | evidence | fired? |
|---|---|---|---|
| **F1** route-ineffective | routed evg_sum stays <0 | routed evg_sum **+18.8k** (≈ construction identity 18432·gain) — swap arithmetic engaged | **NO** — its non-firing is what makes F3 possible |
| **F2** route-too-coarse | gate misroutes healthy majority | 41% of all val routed; KEEP 0.63, MID 0.55, ≈1/3 sparse; 19 vs 11 new wipes; dense +161, mid +5.14, sparse +0.16; R1's fire-half also failed (7/11 = 64% < booked ≥90%, `draft:118`) | **YES** |
| **F3** MSE-veto | evg_sum ≥0 but wipes stay wiped → locus at out/decoder | 7/11 correctly routed with construction-positive evidence, **0/11 rescued**, ratios 0.05–0.48 unchanged (`hypotheses.jsonl:44`) | **YES** |
| F5 route flap | sign flips across epochs | single checkpoint only — booked as needing an ep16 dump (`draft:129`); circumstantial: 4/11 at ±0.03–0.10 margins | untestable locally; suggestive |

**PRIMARY: F3.** Ranking by the counterfactual test: a *perfect* gate (route only true
anti-match wipes) would have removed F2's damage but left the measured outcome — correctly-routed
images with massively positive evidence gained **zero counts** — so the BECAUSE's conversion claim
(§1.B) still dies. F3 is the gate-independent, hypothesis-killing failure; it is exactly the
failure the two-card sequence was built to isolate (§4). **F2 is the co-fired amplifier**: by
arithmetic on the slice deltas, dense+mid contribute (161·17 + 5.14·374) / (3.7187·1286) ≈
**97% of the +3.72** — i.e. F2 is what made N0024 worse than its sibling and killed R3/sparse,
but a better gate alone would never have cleared R2 or the BECAUSE. F1 did not fire; booked F4
("route off on sparse", `draft:128`) is also falsified as a premise — sparse routed ≈1/3 and
regressed against parent (+0.16), sibling (5.703), and bar (5.45) simultaneously.

---

## 3. Design assessment: the hard `top1_mean < 0` whole-image gate

**The 41% fire rate says the "anti-match minority" premise was never true at deployment scope —
and that a band-conditional precision was silently promoted into a whole-split scope selector
without the one read that would have caught it.**

1. **Precision ≠ prevalence.** The booked premise (`idea.md:13`, `draft:40`) — "10/12 wipes at
   gt[250,500) carry top1_mean<0; **71% precision** selector" — is a *within-band, parent-checkpoint,
   gt-conditioned* conditional. It bounds neither (a) the marginal fire rate over the other ~1,250
   val images nor (b) the fire rate at the child's own learned checkpoint. The deployment question
   was always P(top1_mean<0) over all 1286; that read was **free** (the parent dump already carried
   per-image top1_mean) and was never taken. It would have shown the gate is a majority gate
   before launch.
2. **Both selectivity directions failed at deployment.** Recall on the target set: 7/11 = 64% vs
   booked ≥90% (`draft:118`) — 4 misses at margins +0.035…+0.103. Precision on the full split:
   of 531 fires, at most ≈31 lie in the entire gt≥300 population at all (≤19 routed child wipes
   — only 7 of the booked 11 confirmed routed locally — plus ≈12 routed KEEP, arithmetic on the
   ledger's group fractions; exact split needs the non-local `val_attr.json`); true wipe fires are
   a subset. **Single-digit-% whole-val precision.** A selector that misses a third of its targets
   and fires on 41% of everything is not a minority detector; the 71% figure and the 41% reality
   are answers to different questions.
3. **The statistic is endogenous.** `top1` is built from *trained* qproj/kproj (`model.py:230–236`),
   so its whole-image sign distribution moves during training; a fixed hard threshold on a drifting,
   learned statistic cannot be certified from one checkpoint. This is precisely F5's territory
   (`draft:129`) — booked as a read, unresolvable without a second dump. The near-zero margins on
   4/11 are the flap population.
4. **Blast radius per false positive is maximal by construction.** The route is per-image and
   whole-tensor: one negative aggregate replaces *both* evidence channels of a healthy image with
   detached `mhat` (`model.py:241–243`) — no partial credit, no per-cell fallback. At a 41% fire
   rate that is a forward-distribution hijack of most batches *and* a backward starvation of
   qproj/kproj/fine on those images — which is also how the shared `w_c` drifted to 0.0282
   (§1.B), extending the damage to images the gate never fired on. All three slices regressed;
   the healthy majority paid both for the premise being wrong (F2) and for the swap being
    informationally weak (F3/§4). Whole-split end-to-end precision for the wipe phenotype is
    **≤~4% at absolute best** (at most the 19 child gt≥300 wipes as true positives, if all had
    routed — of 531; counting the booking's own gt[250,500) partial-wipe band too, at most ~35
    fires, ≈6%); the rest are KEEP/mid/sideways fires the premise never licensed.
5. **Design lesson (analysis, not a booking):** any aggregate-sign route must ship with
   pre-launch reads of *whole-split fire rate* and *target-set recall* at the deployed
   checkpoint's statistic — precision-within-band cannot license whole-split scope. A gate that
   could work would need prevalence calibration, or a conjunction with a scene-density cue the
   wipe actually depends on (the wipe is scene-conditioned, gt-agnostic to the model,
   `idea.md:15`), or no gate at all. Given this family's outcome (§1–2), the last option is the
   only one not already re-falsified; anything else re-encodes a refuted mechanism and dies on
   §5.11 (new falsifier + `contradicts` booking required).

---

## 4. Two-card synthesis (H0022 + H0023)

H0022 could not change the sign of anti-phase evidence — its mask structurally routes `ev≤0` at
identity, so with the mechanism fully engaged (w_m +0.2666, cellcal bypassed) all 11 wipes stayed
wiped and the sign-domain claim died (`N0023/feedback/qual.md:96–102`, `causal.md:234–240`).
H0023 *did* change the sign on routed images — post-swap evg_sum flipped from the parent's −5301
to ≈ +18.8k, construction-true — and mass still did not move: 0/11 rescued, ratios 0.05–0.48
(`hypotheses.jsonl:44`). Two cards, two operators (gain-domain mask vs substrate swap), one
invariant outcome: **the wipe's count deficit is not carried by the sign of the evidence or by
the gain's sign domain — it localizes below the evidence/gain layer**, in one or both of:
(i) the *consumption interface* `out → cond → decoder` (`model.py:244–262,189`), which cannot
turn a coarse energy field into sharp instance mass under pixel-MSE — the F3 text's own reading
and the H0016/H0017 quiet-null precedent (`draft:101,127`); note the routed SimPrior output is a
*detached, channel-duplicated, normalized* function of `fine`, which the decoder already receives
directly via `cat([fine, cond_map])` (`model.py:189,238`) — informationally redundant, gradient-
frozen, so the head is structurally predisposed to ignore it; and/or (ii) the evidence simply
**lacks count-bearing absolute magnitude** — mhat is mean-1 by construction, so its per-image sum
is pinned at ≈18432 = 2·96² (clamp aside) whether the scene holds 40 or 2000 objects: phase-
correct but **absolute-scale-free**, carrying zero "how many" in its total and only a relative
spatial budget; the observed evg_sum +18.8k matching the construction identity to ~0.5% is the
direct empirical statement that no count information was added. What a *successful* substrate
would therefore need — inferred, not booked: (a) in-phase/non-negative (now established
necessary, twice bracketed); (b) an **absolute scale that covaries with scene count** (unnormalized
energy, or similarity evidence with magnitude the interface can read — bounded cosine means are
no better); (c) spatial structure sharp enough that MSE rewards rather than vetoes it; and (d)
delivery that does not depend on a prevalence-unvalidated sign route (or, if gated, a gate with
measured whole-split selectivity, §3). Supporting contrast: mid slices order
parent 37.50 < N0023 41.71 < N0024 42.64 — mid did **not** recover when cellcal was kept
structurally live, so N0023's mid damage is not purely its dispatch choice and the
touch-ev intervention family itself is implicated (`N0023/feedback/diagnostic.md:199–204`), with
the honest confound that N0024's actuator was behaviorally ~5× weaker anyway (w_c 0.0282), so the
clean dispatch-vs-family split that read hoped for remains unseparated. Net: the locus handed to
the next card sits at the exemplar→similarity encoding (why anti-match appears at all) or at the
out/decoder consumption of magnitude-bearing fields — and any successor re-encoding the
aggregate-sign route family must book as `contradicts` with a genuinely new falsifier (§5.11);
`AGENTS.md:264–266` currently bans only the gain-domain form, so synthesis should extend the
register to this substrate-swap-on-aggregate-sign family too.

---

## 5. Booking-quality notes

### 5.1 Booked vs code — MATCH (diff-verified this session), dispatch fix confirmed good

- **Delta vs parent is exactly the booking:** `diff N0015/model.py N0024/model.py` shows only
  (a) the non-module flag (`model.py:164–165`), (b) `am = dict(use_antimatch=True) …` threaded as
  `**am` through all four parent dispatch arms (`model.py:178–188`), (c) the `use_antimatch`
  kwarg on `SimPrior.forward` (`model.py:228`), (d) the 7-line swap block (`model.py:237–243`)
  inserted after `ev=cat` (`:236`) and **before** cellcal (`:244`) — matching the booked snippet
  (`idea.md:21–29` ≡ `draft:50–58`) line-for-line, incl. `fine.detach()` (`:238`), hard
  stop-grad route (`:241`), `mhat` clamp [0,10] (`:240`), identity-on-route-0 (`:243`).
  Everything else — cellcal block, peakcal block, temp-pin (`:325–326`), construction order
  (`:311–322`), zero new params — is byte-identical. Config delta is exactly one line
  (`config.toml:30` diff). Pluggable single switch — passes the standing-regime ablation rule.
- **Pre-launch dispatch fix: correct and well-handled.** The draft's original attach spec
  (`draft:73–80`: prepend an exclusive `if self.use_antimatch:` branch, peakcal "structurally
  bypassed in this branch") would have violated the booked "cellcal gain … byte-identical" phrase
  that distinguishes this card from H0022; the audit caught it pre-launch (`events.jsonl:59`:
  "exclusive if use_antimatch drops cellcal+peakcal for ALL images … dispatch fix + smoke before
  launch"; account in `N0023/feedback/diagnostic.md:104–110`). Final code threads `**am` instead,
  both parent gains execute, and **w_c = 0.0282 ≠ 0 proves runtime liveness** (the N0023-bypass
  bug did not recur), temp pinned 0.07, w_p 0.00643 dead as parent (0.0017). Process worked as
  intended: bug → fix → smoke → launch, journal-recorded.
- **Residual precision gap:** the booked phrase was source-fidelity, and source-fidelity is what
  was delivered — but the BECAUSE leans on "the *identical confirmed gain*", and the learned gain
  collapsed 5× (w_c 0.0282, gain_on_neg 1.017). The sibling diagnostic's engagement check was
  binary — "w_c ≈ +0.15 else card invalid" (`N0023/feedback/diagnostic.md:145–149`) — and this run
  landed in a third state that framing cannot classify: **alive-but-weak**. Booking lesson: cards
  whose BECAUSE depends on a parent actuator's operating point need an explicit R-read pinning it
  (w_c ∈ [0.14, 0.17]) — R1–R4 as booked (`draft:116–121`) never did.

### 5.2 Falsifier well-posedness — the 4th disjunct is vacuous in exactly the world that occurred

- **The three bars are fine** and decisive at mechanism-level margins: val +3.72 decomposes as
  ≈+1.2 parent-luck + ≈+2.5 child-deviation (~2.2σ over the measured draw-mean, `HANDOFF.md`
  noise band; `N0023/feedback/diagnostic.md:26–39` method), dense +161 sits +131 above the
  same-seed dense band [300,361], mid +5.14 and sparse +0.16 are new-vs-parent regressions, R2
  0/11 and R3 19-vs-11 are within-checkpoint reads noise cannot produce. The ep24 HALT did not
  distort the verdict: best 23.06 ≫ the 20.00 draw-contamination line (`diagnostic.md:152–158`),
  and even extrapolating at the parent's own −0.28/ep gives ≈20.8 at ep32 — still above 19.0431.
- **4th disjunct (structural flaw, H0022 lesson recurring):** booked as "fewer than 7 of the 11
  … reach ≥0.65 **with** the routed post-gain evidence sum not non-negative" (`idea.md:9`).
  Read conjunctively, the limb fires only when rescue fails **AND** sum<0 — i.e. only in the F1
  world. In the measured world (route fired, sum ≥0, rescue 0/11 = **F3**) the conjunction is
  false and the limb is **silent** — it cannot register the very failure that refuted the card.
  Worse, its sum-component is *construction-true for any routed image at any weights*
  (mhat ≥0 ∧ gain = exp(·) >0 ⇒ post-gain sum ≥0; the +18.8k matches the construction identity),
  so — exactly like H0022's 4th disjunct (`N0023/feedback/qual.md:168–181`) — a
  mechanism-by-construction read sits inside a DISPROVED limb where it can never do work. The
  verdict was carried entirely by the three bars (thankfully decisive); as a *registered mechanism
  falsifier* the limb is mis-aimed. Lesson, extending H0022's: construction-true engagement reads
  belong in R-reads (report-only), each independently-refuting condition gets its own disjunct,
  never a conjunction of "failure-of-rescue" with "construction-true arithmetic".
- **R1 as restated in `idea.md:35`** ("routed-evidence-sum non-negative on ≥90% of routed images
  & ≥7/11 ratio≥0.65") is worse: first conjunct 100% construction-true, second duplicates R2 —
  vacuous as written. The draft's R1 (`draft:118`) at least had an informative half — route fires
  on ≥90% of the 11 — which **failed honestly** at 7/11 = 64%. R3 (no-new-wipes guard,
  `idea.md:35`) was well-posed and fired (19 vs 11, dense +49%). R4 bars as booked.

### 5.3 Process nits (not verdict-affecting)

- **Missing local artifacts:** `result.json` / `val_result.json` / `val_attr.json` /
  `val_perimage.json` absent from the node dir (listed as inputs in the handoff); outcome numbers
  trace only to `hypotheses.jsonl:44` / `events.jsonl:62`. Sync from the server copy so feedback
  can re-verify per-image (sibling N0023's feedbacks had local dumps; this card's cannot).
- **`info.json` stale:** `status: "proposed"`, `best_metric: null`, `tested_hypotheses: []`
  despite ledger create (`hypotheses.jsonl:42`, 12:07) and evidence (12:48) — sibling N0023's
  info *was* updated (`status: timeout`, best 22.163). Also `idea.md` §0 lacks the
  runner-parseable `1. **H0023** — <text>` line (same STATE gotcha 5 as N0023). Repair via the
  `discovery` API only, never by hand (§5.3).
- **Config-vs-runtime semantics:** `config.toml:34` still says `augment = false` while the v2
  launch carries `--set augment=true --set futility_bar=19.0431` (`idea.md:42`); neither override
  is recorded in the local config. Same semantics-gap family the sibling diagnostic already
  flagged for displaced flags (`diagnostic.md:130–133`).
- **Draft superseded but unmarked:** `draft:73–80` attach spec ≠ shipped dispatch (§5.1); the
  ledger one-liner is authoritative and matches — mark the draft section superseded to stop
  future readers citing it.
- **Missing pre-launch read (the costly one):** whole-split prevalence of `top1_mean<0` was
  computable from the existing parent dump and would have exposed F2 before burning a GPU slot
  (§3.1). The sparse ≤5.45 bar was booked again despite being parent-infeasible (5.814) — same
  nit as N0023 (`N0023/feedback/qual.md:211–215`); §6 semantics allow it, but it still adds no
  discrimination here (missed, as did parent and sibling).
- **§11/§5.11 hygiene:** nothing banned was re-encoded at booking time (aggregate-sign substrate
  swap ≠ gain-domain mask; novelty gate exit 0 vs H0022 at 0.579, `idea.md:39`), but with
  H0018/H0022 (form/domain) and now H0023 (substrate-on-aggregate-sign) refuted, any further
  card in this neighborhood must book as `contradicts` evidence on the existing ids with a
  genuinely new falsifier — synthesis's call to extend `AGENTS.md:264–266`.

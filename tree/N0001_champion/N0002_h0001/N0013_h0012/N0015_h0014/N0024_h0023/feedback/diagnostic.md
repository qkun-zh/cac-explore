# Diagnostic — N0024_h0023 (H0023 `use_antimatch` AntiMatch substrate-swap) — FAILURE

- **id/parent**: N0024_h0023 ← N0015_h0014 (live, val 19.3431 @ep32 v2).
- **mechanism**: fixed stop-gradient per-image substrate swap — on images whose whole-image
  top1 cosine mean < 0, the two evidence channels fed to the confirmed zero-init `out`
  projection are replaced by the mean-1 fine-energy field `mhat` (both channels), BEFORE the
  parent cellcal gain; zero new parameters, one boolean (`idea.md:3,19-35`;
  `model.py:237-243` swap → `:244-249` cellcal).
- **outcome**: futility-HALTED ep24 (best 23.0618 @ep24, need 0.5251/ep vs ref 0.12 → HALT;
  ledger says result.json clean); full eval: val 23.062 (+3.72 vs 19.3431), dense 491.81
  (+161.0), mid 42.64 (+5.14), sparse 5.972 (+0.16) — all three slices worse than parent AND
  worse than sibling N0023; R2 wipe-rescue 0/11 (bar ≥7/11); R3 no-new-wipes refuted
  (19 new wipes at gt≥300 ratio<0.5 vs parent 11). Ledger verdict: `contradicts` w=0.85,
  H0023 conf 0.415 (`memory/hypotheses.jsonl:44`, 2026-09-22T12:48:03;
  `journal/events.jsonl:62`).
- **constraints**: frozen backbone, head-only, seed 20260830 fixed, v2 protocol, AGENTS §5.16
  futility, AGENTS §11 no-repeat register, rule 11 (no silent retry of refuted mechanisms).
  No commits, no ssh performed for this document.

**Provenance caveat (honest):** this node's local `result.json` / `val_result.json` /
`val_perimage.json` / `val_attr.json` are **absent** (`info.json` still `status: proposed` —
run artifacts live server-side). All outcome numbers below are quoted from the append-only
ledger evidence note (`hypotheses.jsonl:44`) and `journal/events.jsonl:62`, which the Lead
recorded after full eval + `dump_antimatch.py` attribution (`local/dump_antimatch.py`).
Sibling cross-checks (N0023 slices, parent slices) are re-verified from local files.

---

## 1. Failure taxonomy

**Primary class: mechanism-wrong — specifically F3 (consumption interface cannot convert
evidence sign into mass) with a dominant F2 secondary (route gate far too coarse).** Not
quiet-null (the swap fired), not operator-wrong (dispatch as-booked post-fix, §2), not
gate-too-strict (HALT is beyond every observed draw).

**(a) What part of +3.72 can be noise.** Same-seed draws of the identical N0015 config:
{19.3431 live, 20.697 cleaned-retrain N0021, 21.5928 exact repro}, mean 20.544, sd ≈ 1.13
(`perimage_diagnostic.md:231`; `journal/events.jsonl:55,57`). `HANDOFF.md:34-35` voids the
±0.02 precision claim; heuristic ≤±1.5 untrustworthy, ≥+2 credible. Decomposition of +3.72:
- **≈ +1.20 is parent draw-luck** (20.544 − 19.3431): live parent is the minimum of the three
  draws ("favorable coin flip").
- **≈ +2.52 is the child's deviation above the draw-mean** (23.062 − 20.544 ≈ 2.2σ of a single
  draw; under exchangeable sd(child−parent) ≈ 1.60, +3.72 ≈ 2.3σ, one-sided p ≈ 0.01) — well
  past the HANDOFF ≥+2 "credible" line. **Beyond the noise band.**
- The ep24 HALT at best 23.06 sits **above even the worst same-seed draw (21.59) by +1.47** —
  this is not the draw-contaminated band (20.0, 21.6) that N0023's diagnostic pre-registered
  (`N0023/feedback/diagnostic.md:153-158`); the halt is mechanism-decisive.

**(b) What part is mechanism-caused and cannot be noise.**
- **R2 = 0/11 with the route correctly fired on 7/11** (bar ≥7/11, `idea.md:9`): of the booked
  11 wipes, 7 had top1_mean<0 and were routed; their routed **post-swap evg_sum = +18.8k**
  (construction-true non-negative — R1 literally true by arithmetic: mean-1 `mhat` sums to
  ~96·96 ≈ 9216 per channel ×2 after gain), yet per-image ratios stayed at 0.05–0.48. The 4
  not routed (840, 935, 3482, 3487 — top1 slightly positive) also stayed wiped. **A
  phase-correct, massively positive evidence tensor produced zero counts.** This is the
  noise-invariant kill: within-checkpoint mechanism read, identical to N0023's 0/11 but now
  under the *opposite* evidence-sign condition N0023 could not produce.
- **R3 refuted: 19 new wipes vs parent 11** at gt≥300 ratio<0.5 (dense +49% of wipe count);
  dense slice 491.81 exceeds the same-seed retrain band [300, 361] (H0020 booking) by
  **+131** — at least +131 of the +161 dense regression is structural.
- **F2 route-too-coarse (measured, not inferred):** 531/1286 = **41% of ALL val routed**;
  KEEP routed_frac = **0.63**, MID = 0.55 (`dump_antimatch` val_attr, `hypotheses.jsonl:44`).
  The hard `top1_mean < 0` gate — booked as firing on "exactly that anti-match minority"
  (draft priced ~2–3% of val, `h0023a_idea_DRAFT.md:134(a)`) — actually fired on the
  **healthy majority**, replacing their similarity substrate with the mean-1 field. The
  booking's minority-scope premise is empirically false at gate level.
- **Mid +5.14 (42.64 vs 37.50):** with cellcal *retained* (unlike N0023), mid still sits at
  ~42 — the regression tracks touching `ev` on the majority (via the mis-calibrated gate), not
  cellcal removal. This is the pre-registered family-vs-dispatch discriminator (N0023
  diagnostic §4.3) and it reads **family-caused**.
- **Sparse +0.16 (5.972):** first card in this lineage to make sparse *worse than parent*
  (5.814) — consistent with 41% over-routing bleeding into sparse images (F4-adjacent).
- **Mechanism engaged, gain stack weak:** dispatch fix confirmed alive (w_c = 0.0282 ≠ 0,
  temp pinned 0.07, w_p = 0.0064) but the learned gain ran **~5× weaker than parent**
  (parent w_c 0.148–0.166; gain_on_neg 1.017 vs parent ~1.08–1.10). Not a bypass — a
  training-dynamics consequence of 41% of inputs having their substrate replaced (§2).

**(c) Classes explicitly ruled out.**
- *Gate-too-strict / early stop*: need 0.5251/ep ≫ 0.12 at ep24; extrapolating the parent's own
  late slope −0.28/ep from 23.06 still ends ≈20.7 @ep32, +1.66 above the 19.0431 bar. The HALT
  saved ~8 epochs without changing the verdict (AGENTS §5.16).
- *Operator-wrong / dispatch bug*: the pre-launch exclusive-branch bug was caught and fixed
  (journal 12:21:57); runtime w_c ≠ 0 proves the fix held. §2.
- *Quiet-null / F1 route-ineffective*: falsified by construction — routed evg_sum +18.8k,
  swap block at `model.py:237-243` executes before cellcal. The substrate DID change.
- *Draw-noise as sole cause*: contradicted by (b).

**Taxonomy verdict: mechanism-wrong.** The substrate swap executed as booked, phase-corrected
the evidence (R1 true), and the counts did not move (R2 0/11) — while a mis-calibrated gate
simultaneously corrupted the healthy majority (41% over-route, R3 19 vs 11). Noise accounts
for ≈ the first +1.20 of +3.72; the rest is structure. The refutation rests on reads noise
cannot produce: 0/11 despite correct routing + positive evg_sum, R3, dense +131 outside band,
mid ≈42 with cellcal retained.

---

## 2. What was actually tested — dispatch vs booking, and is the hypothesis only partially tested?

**Facts (code).**
- `config.toml:28-30`: `use_cellcal = true`, `use_peakcal = true`, `use_antimatch = true` —
  matches runtime (no N0023-style config-language caveat).
- Dispatch (post-fix): `model.py:178` builds `am = dict(use_antimatch=True)` and threads
  `**am` through **all four** parent arms (`:181` cellcal+peakcal, `:183` cellcal, `:186`
  peakcal, `:188` plain). The swap is a **kwarg composition inside SimPrior**, not an
  exclusive branch — parent cellcal/peakcal lines remain reachable on every image.
- Inside `SimPrior.forward`: swap block `:237-243` runs **before** the parent cellcal block
  `:244-249` — exactly the booked order (`idea.md:19`: "AFTER `ev = cat([top1,cons])`, BEFORE
  cellcal block"; `h0023a_idea_DRAFT.md:48`).
- Runtime confirmation: w_c = 0.0282 ≠ 0 (cellcal received gradient — the N0023 bypass bug
  class did NOT recur), temp = 0.07 pinned, w_p = 0.0064 (`hypotheses.jsonl:44`,
  dump_antimatch). Gain_on_neg 1.017 > 1 (gain live on negative cells).

**History of the dispatch bug (why this card is as-booked).**
- The original draft attach proposed an **exclusive** `if self.use_antimatch:` branch that
  dropped cellcal+peakcal kwargs (`h0023a_idea_DRAFT.md:73-80`) — runtime-diverging from the
  booking phrase "leaving the cellcal gain … byte-identical" (H0023 create text,
  `hypotheses.jsonl:42`; `idea.md:9`).
- Caught pre-launch by audit: "exclusive if use_antimatch drops cellcal+peakcal for ALL
  images, contradicting booked 'cellcal byte-identical' — dispatch fix + smoke before launch"
  (`journal/events.jsonl:59`, 12:21:57). Fixed code = current `model.py:178-188`
  (am-composition-through-parent-chain) + swap-before-gain order. Smoke re-run before launch.
- Contrast with N0023 (H0022): there the exclusive dispatch **was the booked design**
  ("replacement", ledger create 11:31:15; N0023 diagnostic §2) and no fix applied. H0023's
  booking explicitly required runtime persistence; the fix made the code honor it.

**What the booking actually says** (authoritative: ledger create `hypotheses.jsonl:42`,
mirrored `idea.md:9`): swap on anti-match-minority images; cellcal gain and every other parent
line byte-identical; triple bars (val ≤19.0431, dense <330.81, sparse ≤5.45) + R2 ≥7/11 with
non-negative routed post-gain sum.

**Assessment — is H0023 only partially tested?**
- **On the booked design: no — substantially (fully) tested.** Every clause executed: hard
  stop-gradient gate on aggregate top1 sign (`model.py:241`), substrate swap to mean-1 mhat
  both channels (`:240,242-243`), cellcal retained and applied after swap (`:244-249`),
  temp-pin intact, zero new params (no module → strongest RNG compliance), flag-off
  byte-identical parent path, triple bars + R1/R2/R3 evaluated. w_c ≠ 0 proves the dispatch
  fix ran. There is no H0023-style unresolved implementation-vs-booking contradiction; the
  bug was found and fixed pre-launch.
- **Two honest residues, neither of which invalidates the test:**
  1. **Gain-stack weakness (w_c 0.0282 vs 0.148–0.166, ~5×).** Code-level cellcal is
     byte-identical; the *learned* scalar collapsed because 41% of training inputs now carry
     a mean-1 detached substrate (zero grad to qproj/kproj/fine from them), changing the
     gradient landscape. The booking's "cellcal byte-identical" is a *source/runtime-persistence*
     claim (satisfied), not a claim that the optimizer would relearn the same w_c. This is an
     outcome of the intervention, correctly attributable to the mechanism, not an operator
     error — but it does mean the card tested "swap + weakened-by-dynamics gain," not
     "swap + parent-strength gain." A frozen-w_c replay is a counterfactual no booked card
     paid for; record it, don't treat it as partial-test invalidation (the gain was never
     the deciding read — R2 failed even with construction-true positive routed evidence).
  2. **Scope premise false: gate fired on 41% of val / 63% of healthy KEEP.** The booking's
     BECAUSE says "on exactly that anti-match minority" — the *code* is exactly as booked
     (hard top1<0), but the *empirically measured minority* (~2–3% priced in the draft) is
     actually a near-half of val. This falsifies the booking's scope conjunct (part of what
     was tested is "the gate is selective" — it isn't). That is a legitimate refutation path,
     not under-testing: the hypothesis as written claimed minority-scope safety, and R3
     (no new wipes) is the pre-registered falsifier for over-routing — R3 failed.
- **Net:** H0023-as-booked ran clean and was refuted. The untested counterfactuals
  (parent-strength w_c frozen; a correctly-calibrated minority gate) are *successor*
  hypotheses under rule 11, not incomplete tests of this one — and §3 rules the F2-successor
  dead on its own pre-registered mid read.

**Caveat for future cards:** `info.json` status still reads `proposed` with null best_metric
while the run completed server-side — tree-state drift to repair via the API (never by hand,
AGENTS rule 3) when run artifacts are pulled.

---

## 3. Two-card closure statement (H0022 N0023 + H0023 N0024)

The registered pair was designed as complementary probes of the dense-wipe locus: H0022
masks the gain's sign-domain (negatives at identity), H0023 swaps the substrate itself
(in-phase mhat) while keeping the gain. Both ran futility-HALTED at ep24, both as-booked,
both 0/11 on the wipe set, both REFUTED (`hypotheses.jsonl:43,44`).

**Ruled out (with evidence):**
1. **Gain-sign domain** — closed at §11 already (H0018 form + H0022 domain: "masking negatives
   cannot convert anti-phase evidence, 0/11 wipe rescue with mechanism fully engaged,"
   `AGENTS.md:264-266`). N0023: w_m=+0.2666 engaged, mask live, 0/11, mid +4.21.
2. **Evidence-sign at the `ev` interface** — H0023 made routed post-gain evg_sum go from
   parent −5301 to **+18.8k** (phase corrected, sign flipped) and counts still did not move
   (0/11, ratios 0.05–0.48). Sign/phase at `ev` is **not sufficient**.
3. **Substrate-swap-with-scale-free-field** — mean-1 `mhat` (sum ≈ 18432 by construction,
   `model.py:240`) is phase-correct but **absolute-scale-free**: it carries no count
   magnitude. Combined with (2): swapping to a *count-bearing* positive substrate remains the
   only untested evidence-layer variant — but see §11 check below before booking it.
4. **Hard top1<0 per-image gate as scope** — fires on 41% of val / 63% of KEEP (F2). The
   anti-match *minority* premise is false at gate level; any successor must re-specify the
   route statistic (new falsifier, rule 11 — this re-encodes H0023's routing mechanism).
5. **The two-regime F2-successor (blanket cellcal on majority + repair only on anti-match
   minority)** — N0023 diagnostic §4.3 pre-registered the discriminator: N0024 mid ≈37.5 ⇒
   N0023's mid regression was cellcal-removal (dispatch), ≈41 ⇒ **family**. Measured N0024
   mid = **42.64 with cellcal retained** — squarely the family read. Touching `ev`'s
   substrate on the majority (even via a "fixed" gate) damages mid/dense; **do not book the
   two-regime hybrid** (it was already conditioned "NOT booked now — conditioned on N0024 read
   3" in N0023 synthesis §4b; read 3 came back negative).
6. **Retained-gain-as-actuator** — cellcal stayed on (w_c>0) and still no rescue: the
   confirmed gain cannot mint mass from either anti-phase similarity (parent, wipes) or
   mean-1 energy (N0024, routed) once `out`/decoder decline to convert it.

**Localized conclusion (the closure sentence):** *the dense wipe lives BELOW the
evidence/gain layer* — either (A) the **ev→out→decoder consumption interface** cannot turn
evidence sign/scale into density mass, or (B) the evidence, even when phase-corrected, lacks
**count-bearing absolute magnitude** for that interface to consume. H0022 killed (mask) and
H0023 killed (swap) both close the evidence/gain layer itself.

**Still open, each checked against AGENTS §11 + prior refutations before any booking:**
- **(A) out/decoder consumption capacity.** Frontier named by H0023's own F3
  (`h0023a_idea_DRAFT.md:127`) — but: §11 bans final-layer readout and the N0009/N0010
  bans cover *any trainable post-decoder additive* (even null) on the density path;
  decoder `in_ch`=192 untouched is load-bearing; H0016/H0017 (enmass/peakmass) already
  showed new energy-additive paths into `cond` are F1 quiet-null (mhat substrate exhausted
  by cellcal's gain — `hypotheses.jsonl:31,33`). The draft itself flags H0017's peak-gated
  additive as quiet-null ⇒ "this is a weak frontier." **Expect §11/novelty death; not
  recommended as card3.**
- **(B) count-bearing positive substrate design.** Open in the narrow sense: replace mean-1
  mhat with an unnormalized / count-calibrated field routed through the *existing* `out`
  (no new projection → avoids H0016's quiet-null geometry). But this **re-encodes H0023's
  core mechanism** (substrate swap on anti-match) with a different normalizer → rule 11
  requires booking as `contradicts`-evidence on H0023 with a **genuinely NEW falsifier**;
  a single scalar/normalizer change on a decisively-refuted card (+3.72, 0/11, R3 refuted)
  is a thin re-encode with negative expectation. Also collides with the H0016 lesson that
  *every* extra path consuming raw mhat was gradient-redundant under cellcal. **Not
  recommended as card3.**
- **(C) Exemplar-encoding / exemplar-distinctness** — the registered open direction, §5.

**§11 register check for new candidates:** gain-domain masks closed (H0018+H0022);
transfer-by-fusion closed (H0021); pre-condenser exemplar *gating* closed (banned — note
distinctness ≠ gating: no gate/scalar/mask on `e`, different mechanism, measured lever
logit-var 0.013); DDCA/spatial summaries/final-layer/unfreeze closed; no second grad path /
shared projection / post-decoder additive (H0006/H0007/H0009 corpus). A "better gate" for
H0023 and an "unnormalized substrate" for H0023 both die at rule 11 without a truly new
falsifier — and neither clears novelty vs the refuted parent id.

---

## 4. HANDOFF card-budget ruling

**Ruling: remaining budget = exactly one card (the exemplar-distinctness /
exemplar-encoding-interface card), then the paper decision.**

- HANDOFF §5 sets "3 cards no >1.5 net drop → 转论文收官" (`HANDOFF.md:64`); the registered
  remaining idea is "exemplar-distinctness (期望为负)" (`HANDOFF.md:48`), with a dense-targeted
  variant as the H0022/H0023-successor form (`HANDOFF.md:62`).
- Budget composition was explicitly resolved this session (`journal/events.jsonl:60`,
  12:41:45): **3 cards = N0023 (H0022) + N0024 (H0023) + exemplar-distinctness**; paper
  trigger only after all 3 miss >1.5 net. N0022 (H0021 `use_exkern`) is **pre-HANDOFF and does
  not count** toward the budget (same journal entry).
- Card accounting now:
  | # | Node | Hyp | Net vs 19.3431 | >1.5 drop? | Status |
  |---|---|---|---|---|---|
  | 1 | N0023_h0022 | H0022 use_margin | **+2.82** (worse) | no | done, REFUTED |
  | 2 | N0024_h0023 | H0023 use_antimatch | **+3.72** (worse) | no | done, REFUTED |
  | 3 | (unbooked) | exemplar-distinctness | — | — | **budget remains: 1 card** |
- Both spent cards miss the >1.5 criterion in the wrong direction by ≫1.5; the paper trigger
  therefore fires **only after card 3 also fails to deliver a >1.5 net drop** (or dies
  honestly at the novelty/§11 gate). Stopping now would under-count the explicit 3-card
  budget and leave the registered final localization probe untested — N0023's diagnostic §5
  already prescribed the order *N0024 verdict → book third card → (if no >1.5) paper*;
  the N0024 verdict has now landed (miss), so the sequence's next step is unambiguous.

---

## 5. Recommendation

**Book card3 — the exemplar-distinctness / exemplar-encoding-interface card — as the third
and final post-HANDOFF card. Do NOT go straight to paper.**

**Why card3 and not paper (yet):**
1. The budget ruling above: 3 cards are registered; 2/3 spent; the third is the only
   remaining unexhausted readout-side frontier and its brief is already drafted
   (`N0023/feedback/synthesis.md` candidate A: `use_exdistinct`, novelty exit 0, top sim
   0.452 (H0019) < 0.82, format PASS — deliberately "NOT booked now — … registered order is
   N0024-verdict-first"; that order condition is now satisfied).
2. It targets the measured lever nobody has tried: K=3 exemplars mutually indistinct
   (logit-var 0.0134, top1 mean −0.125; `journal/events.jsonl:32`) — and the two-card
   closure (§3) points upstream of the gain at the **exemplar-encoding interface** as the
   one locus neither mask nor swap touched (H0022 F3 successor `h0022_idea_DRAFT.md:131`;
   HANDOFF:48 期望为负 / :62 dense-targeted variant). It is cheap, zero-to-few params, and
   its **expected-negative outcome is itself paper-grade**: it completes the collapse-corpus
   story (wipe localized below evidence/gain; exemplar side measured dead) that
   `paper_spine.md` already positions.
3. HANDOFF:64's paper trigger is conjunctive on all 3 cards missing >1.5 — the trigger is
   not yet armed.

**Why NOT the tempting alternatives (each dies at §11 / rule 11 / prior refutations):**
- **Two-regime F2-successor (retain cellcal on majority):** killed by its own pre-registered
  read — N0024 mid 42.64 with cellcal retained ≈ N0023's 41.71, not parent 37.50 ⇒ family-
  caused (§3.5). Never book.
- **Better/finer route gate (H0023 retry):** re-encodes a refuted mechanism → rule 11 needs
  a new falsifier on H0023; the minority-scope premise is already falsified (41% route);
  F2 was the card's own priced failure. Thin re-encode, negative expectation.
- **Count-bearing/unnormalized substrate (H0023 normalizer swap):** same rule-11 problem;
  collides with H0016/H0017 quiet-nulls (raw-mhat paths gradient-redundant under cellcal);
  single-normalizer delta on a +3.72/0/11 card is not a fresh hypothesis.
- **out/decoder consumption surgery:** N0009/N0010 bans (no trainable post-decoder additive
  even null), §11 final-layer readout ban, load-bearing condenser/decoder contract; H0017
  peak-gated frontier already quiet-null. Novelty/§11 death likely; not card3.

**Sequence (unchanged from N0023 diagnostic §5, now one step further):**
1. ✅ N0023 verdict (miss).
2. ✅ N0024 verdict (miss) — this document.
3. → **Book + run card3 (exemplar-distinctness, dense-targeted variant per HANDOFF:62)**,
   novel-falsifier fresh, futility_bar mandatory, mechanism reads (distinctness/lane-separation
   or exemplar-side surrogate read) arbitrated before bars per the single-seed noise framing
   (HANDOFF:34-35; favorable-draw counts as real only if mechanism reads hold).
4. → If card3 also delivers no >1.5 net drop (or dies at novelty/§11): **paper closure**
   per HANDOFF:64 — numbers (val 19.3431 / test 19.066), collapse corpus, two-card wipe
   localization, futility machinery (`STATE.md` paper framing, `local/research/paper_spine.md`
   skeletons) are already in hand.

---

**Files written:** this document only —
`tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014/N0024_h0023/feedback/diagnostic.md`.
No commits, no ssh, no ledger/info.json edits.

**Recommendation:** book card3 (exemplar-distinctness), run it, then decide paper — do not
skip to paper now.

# Diagnostic — N0023_h0022 (H0022 `use_margin` MarginCal) — FAILURE

- **id/parent**: N0023_h0022 ← N0015_h0014 (live, val 19.3431 @ep32 v2).
- **mechanism**: MarginCal — sign-mask-restricted `exp(w_m·log1p(mhat))` gain on positive-margin
  SimPrior evidence, negatives routed at identity, call-site exclusive of cellcal+peakcal.
- **outcome**: futility-HALTED ep24 (`result.json`: `best_mae` 22.1632, `best_epoch` 24,
  `futility_hit` true, `cfg_overrides` {augment, futility_bar 19.0431}); all triple bars missed;
  ledger verdict: `contradicts` w=0.85, H0022 conf 0.415 (`memory/hypotheses.jsonl`, 2026-09-22T12:21:33).
- **constraints**: frozen backbone, head-only, seed 20260830 fixed, v2 protocol, AGENTS §5.16 futility,
  AGENTS §11 no-repeat register. No commits, no ssh performed for this document.

Verified locally from node artifacts: slices recomputed from `val_perimage.json` reproduce
`val_result.json` exactly (sparse 5.7026 / mid 41.7143 / dense 458.6918, n=895/374/17);
the 11 wiped-image ratios at N0023 best.pth are
840→0.285, 851→0.428, 865→0.278, 935→0.148, 3428→0.091, 3481→0.047, 3482→0.072,
3484→0.063, 3487→0.224, 3488→0.092, 7656→0.298 — **0/11 ≥ 0.65** (R2 bar: ≥7/11).

---

## 1. Failure taxonomy

**Primary class: mechanism-wrong.** Secondary contributor: draw-noise inflating the headline gap.
Not operator-wrong, not gate-too-strict; implementation-vs-booking is *as-booked* (see §2, with one
config-language caveat).

**(a) What part of +2.82 can be noise.** The same-seed precision datum stands at three draws of the
identical N0015 config: {19.3431 live, 20.697 cleaned-retrain N0021, 21.5928 exact repro}, mean
20.544, sd ≈ 1.13 (`local/research/perimage_diagnostic.md:231`; `journal/events.jsonl`
2026-09-22T11:52:47 and 11:06:42). `HANDOFF.md:34` voids the AGENTS §6 "±0.02" precision claim and
sets the operative heuristic: ≤±1.5 untrustworthy, ≥+2 credible. Decomposing the child's +2.82
(22.1632 − 19.3431):
- **≈ +1.20 is certainly parent luck** (draw-mean 20.544 − 19.3431): the live parent is the minimum
  of the observed draws (`perimage_diagnostic.md:231`: "live 19.34 is a favorable coin flip").
- **≈ +1.62 is the child's deviation above the draw-mean** (≈1.4σ) — noise-consistent in isolation;
  under an exchangeable-draw null sd(child−parent) ≈ 1.13·√2 ≈ 1.60, so +2.82 ≈ 1.76σ
  (one-sided p ≈ 0.04): pure noise remains *possible* but is strained, and per the HANDOFF ≥+2
  heuristic most of it should be treated as real.
So: of the headline +2.82, roughly the first ~+1.2 is noise attributable to the parent's favorable
draw; the rest sits at the edge of the noise band and cannot alone carry the verdict.

**(b) What part is mechanism-caused and cannot be noise.**
- **Dense +127.9** (458.69 vs 330.81, `val_result.json:19` vs booked bar): the same-seed retrain
  band for N0015's dense tail is [300, 361] (H0020 booking, `memory/hypotheses.jsonl` 2026-09-21T19:26:44).
  458.69 exceeds the band's upper edge by **+98** — at least +98 of the +127.9 is mechanism-caused,
  whatever the level noise does to val MAE.
- **R2 wipe-rescue 0/11** (verified above from `val_perimage.json`; needs ≥7 per `idea.md:34`):
  a within-checkpoint mechanism read, invariant to level noise. Every wiped image stayed wiped or
  deepened (935: parent ratio 0.40 → 0.148; 3428: 0.23 → 0.091 — `hypotheses.jsonl` H0022 evidence).
  Masking negatives rescued nothing; this directly falsifies the BECAUSE clause.
- **Mid +4.21** (41.71 vs parent 37.50, n=374 — `val_result.json:28`, `idea.md:13`): a stable slice;
  the regression tracks the removal of cellcal's blanket gain from the healthy majority, not draw noise.
- **Trajectory shape**: ahead of parent through ep14–15 (23.48 vs 23.92), crossover ~ep16,
  −0.94 behind at ep23, late slope child −0.155/ep vs parent −0.28/ep (`hypotheses.jsonl` H0022
  evidence, 2026-09-22T12:21:33) — a systematic late-phase divergence, not a level offset.
- **Mechanism engaged as booked**: w_m=+0.2666, temp pinned exactly 0.07, w_c=0 (structurally
  bypassed) — same evidence event. Engaged-but-failing + 0/11 = the draft's own F3 path
  ("wipes stay wiped → collapse lives in exemplar encoding / out-projection, not the gain's sign
  domain", `local/research/h0022_idea_DRAFT.md:131`), plus an F2-flavored majority cost
  (the draft's dominant risk: "capping a load-bearing gain on healthy cells",
  `h0022_idea_DRAFT.md:136`).

**(c) Classes explicitly ruled out.**
- *Gate-too-strict*: at HALT (best 22.1632, bar 19.0431), even extrapolating at the **parent's own**
  late slope −0.28/ep gives ≈19.92 at ep32 — still +0.88 above the bar; at the child's own
  −0.155/ep, ≈20.92. The HALT saved ~8 epochs without changing the verdict (AGENTS §5.16,
  `AGENTS.md:170-177`).
- *Operator-wrong*: the dispatch matches the explicitly booked design (§2).
- *Sparse conjunct*: 5.703 vs bar 5.45 — the bar is a **standing parent miss** (N0015 own sparse
  5.814; `h0023a_idea_DRAFT.md:134(b)`), and the child actually *improved* sparse by −0.11 vs parent.
  Sparse never decided this verdict; dense + wipes did.
- *Draw-noise as sole cause*: contradicted by (b).

**Taxonomy verdict: mechanism-wrong** — sign-mask-restricted gain, engaged at full strength, failed
to convert anti-phase similarity evidence into mass (0/11) and cost the healthy majority its late-phase
consolidation (mid +4.21, late-slope loss); noise accounts for roughly the first +1.2 of the +2.82
headline and possibly a bit more, but the refutation rests on reads noise cannot produce.

---

## 2. What was actually tested — dispatch vs booking, and is the hypothesis only partially tested?

**Facts.**
- `config.toml:28-30` keeps `use_cellcal = true`, `use_peakcal = true`, `use_margin = true` —
  read literally, all three flags active.
- Runtime says otherwise: `model.py:176-188` dispatches `if self.use_margin:` as the **first**
  branch under `use_simprior`, passing only `use_margin=True, w_m=...`; `cellcal`/`peakcal` kwargs
  are not passed (they ride the `elif use_cellcal:` arm). Inside `SimPrior.forward`, the margin
  block (`model.py:237-244`) **returns at line 244**, before the cellcal block (`:245-250`) and
  peakcal block (`:251-262`). Therefore for **all images**: cellcal never executes, w_c receives no
  gradient and stays 0 (confirmed: `hypotheses.jsonl` H0022 evidence — "cellcal structurally
  bypassed by margin path (w_c stays 0)"); peakcal is dead code either way (w_p≈0.0017, H0014).
- Net: the card ran **sign-selective gain WITHOUT the parent gain stack** — cellcal (the confirmed
  H0012 actuator, w_c +0.148…+0.166) removed globally, not merely re-scoped by the mask.

**What the booking actually says.**
- The ledger-authoritative one-liner books **replacement**: "…*as a replacement for the sign-blind
  cellcal gain*…" (`hypotheses.jsonl` H0022 create, 2026-09-22T11:31:15; `idea.md:9`).
- The draft books call-site exclusivity explicitly and repeatedly: "flag-swap that REPLACES cellcal's
  sign-blind gain the moment it is on… exclusivity is enforced at the call site"
  (`h0022_idea_DRAFT.md:4`); "passes only use_margin=True; parent blocks … SKIPPED"
  (`:63-64`); attach spec "cellcal/peakcal flags NOT passed → SimPrior runs ONLY the margin block"
  (`:80`). `idea.md:21` — "parent blocks byte-identical; margin path is call-site exclusive" —
  claims **source** byte-identity of the parent blocks, not runtime persistence.
- The *runtime* phrase "leaving the cellcal gain … byte-identical" belongs to the **H0023** booking
  (`hypotheses.jsonl` H0023 create; `h0023a_idea_DRAFT.md:4,11`), and *its* exclusive-dispatch
  violation was caught as a bug and fixed pre-launch for N0024 (`journal/events.jsonl`
  2026-09-22T12:21:57: "exclusive if use_antimatch drops cellcal+peakcal for ALL images,
  contradicting booked 'cellcal byte-identical' — dispatch fix + smoke before launch";
  fixed code at `N0024_h0023/model.py:178-188`, which threads `**am` through the parent
  cellcal chain).

**Assessment — is H0022 only partially tested?**
- **On the booked one-liner: no — it is substantially (fully) tested.** The booking says
  "replacement", the draft double-books exclusivity, the code implements exactly that, and every
  clause executed: positive margins got the gain (w_m=+0.2666), negatives rode identity (mask live),
  temp pinned, triple bars + R2 evaluated → REFUTED (`hypotheses.jsonl` H0022 evidence,
  contradicts w=0.85). There is no H0023-style implementation-vs-booking contradiction here; the
  'cellcal byte-identical' runtime phrase is not in H0022's booking.
- **What the dispatch did confound (the honest partial-test residue):** global replacement merges
  two factors — (i) sign-selectivity on the anti-match minority (the BECAUSE's subject) and
  (ii) cellcal removal on the healthy majority (the card's own priced dominant risk,
  `h0022_idea_DRAFT.md:136`). The 0/11 read kills (i) as *sufficient*; the late-slope/mid regression
  prices (ii) as *costly*; the run cannot separate them. The untested counterfactual is the
  **two-regime hybrid** — blanket cellcal retained where evidence is net-positive, identity/selective
  treatment only on the anti-match minority — explicitly registered as the F2 successor requiring a
  *fresh falsifier*, "never a silent re-encode" (`h0022_idea_DRAFT.md:130`). That variant was never
  part of H0022-as-booked; booking it later is a new hypothesis, not a re-run. N0024 (H0023) is the
  registered complementary probe of the minority-scoped idea in substrate-swap form, *with* cellcal
  byte-identical (post-fix `N0024_h0023/model.py:178-188`).
- **Caveat to record (the confusion vector):** `config.toml:28-29` advertising
  `use_cellcal/use_peakcal = true` while the dispatch makes them runtime-dead is exactly the
  semantics gap that produced N0024's pre-launch audit catch. Future cards that displace a parent
  flag should set it `false` (or comment `# bypassed by use_<new>`) so config matches runtime.

---

## 3. Lessons that constrain the NEXT card (N0024_h0023 `use_antimatch`)

**Inherit (already satisfied — verify, don't re-derive):**
1. **Temp-pin hygiene** (H0014 rule, mandatory): N0024 pins via `use_peakcal`
   (`N0024_h0023/model.py:325-326`) — active only because `config.toml:29` keeps
   `use_peakcal=true`. *Latent footgun:* the pin condition does **not** mention `use_antimatch`;
   never launch antimatch with peakcal off without extending it (N0023 did extend:
   `N0023_h0022/model.py:341`, `use_peakcal or use_margin`).
2. **Cellcal must be runtime-ACTIVE** — the cardinal N0023 lesson. N0024's fixed dispatch keeps the
   parent chain and injects the swap as a kwarg: swap block `N0024_h0023/model.py:237-243` runs
   *before* cellcal `:244-249`, both reached from `:178-188`. **Feedback must verify w_c ≈ +0.15**
   (parent range 0.148–0.166) and, if an attribution dump is run, `gain_on_neg ≈ 1.08` on KEEP
   (`perimage_diagnostic.md:216`). w_c ≈ 0 ⇒ the N0023 bypass bug recurred ⇒ card invalid.
3. **Bars/futility as booked**: triple bars vs 19.3431 / 330.81 / 5.45 and
   `--set futility_bar=19.0431` (`N0024_h0023/idea.md:7,42`).
4. **Gate arithmetic — the hard operational watch:** ep24 HALT iff required slope
   (best − 19.0431)/8 > 0.12 ⇒ best > **20.00**. **N0024 must be ≤20.00 at ep24 to finish.**
   Two of the three known same-seed draws (20.70, 21.59) would halt at that line — an ep24 HALT
   with best in (20.0, 21.6) is *draw-contaminated*, not a clean mechanism kill. Per AGENTS §5.16
   (halt keeps best.pth; verdict = futility refutation + full eval) and `h0023a_idea_DRAFT.md:134`
   (single-seed bet, mechanism reads weighted above level bars), arbitrate with R1/R2 first:
   route-fire fraction, routed post-gain evidence-sum sign, per-image wipe ratios — then the bars.
5. **Trajectory watch**: N0023 was ahead through ep14–15, crossover ~ep16
   (`hypotheses.jsonl` H0022 evidence). Compare N0024's paired curve against the parent at the
   ep16 WARN: if the same crossover recurs *with cellcal retained*, the late degradation implicates
   the intervention family itself (touching `ev`'s substrate/sign for routed images) rather than
   cellcal removal — a direct constraint on any future mask/swap card.
6. **The refuting lesson N0024 exists to test**: masking negatives ≠ phase-correcting anti-phase
   evidence (N0023: 0/11 with the mask fully live). N0024's bet — swap the substrate to in-phase
   `mhat` *before* the still-active gain (`N0024_h0023/model.py:237-249`) — is the direct successor;
   its R1 (routed evg_sum ≥ 0 on ≥90% of routed images, `h0023a_idea_DRAFT.md:118`) is the exact
   read N0023 structurally could not produce (it never changed `ev`, so the sum stayed negative by
   construction).
7. **Sparse 5.45 is a standing parent miss** (5.814) — watch it (N0023 got to 5.703, an
   improvement), but dense + wipe reads arbitrate; don't let a sparse-only miss masquerade as the
   story.
8. **Noise framing**: judge with the ±1.2/1.6 decomposition of §1 in mind — a bar-clearing N0024
   still needs R1/R2 to be believed; a bar-missing N0024 with strong R1/R2 is ambiguous and must be
   recorded as such (H0023a's own §5 ruling: favorable draw + mechanism reads counts as real).

**Watch (specific N0024 failure signatures):** F1 route-ineffective (evg_sum stays <0 — substrate
swap does not feed the out interface); F3 MSE-veto (evg_sum ≥0 but wipes stay wiped → locus moves
to out/decoder, `h0023a_idea_DRAFT.md:127`); F5 route flap (ep16-vs-ep24 route-sign flips ≥5 —
needs a second dump; single-ckpt perimage cannot see it).

---

## 4. Follow-up diagnostic read on N0024's val_perimage (after it lands)

**Yes — six reads settle exactly what this run left open** (all inference-only; the wipe table comes
from the existing tool behind `perimage_diagnostic.md:207`, `local/dump_attribution.py`):

1. **Substrate vs mask, same 11 images** — per-image pred/gt ratios for
   {840, 851, 865, 935, 3428, 3481, 3482, 3484, 3487, 3488, 7656}, paired against the N0023 ratios
   verified in this document (0.285 / 0.428 / 0.278 / 0.148 / 0.091 / 0.047 / 0.072 / 0.063 /
   0.224 / 0.092 / 0.298; 0/11) and against parent N0015. This is the direct causal comparison the
   two-card sequence was designed to deliver: domain-mask (H0022) vs substrate-swap (H0023) on
   identical images.
2. **WIPE/KEEP/MID attribution table on N0024 best.pth** (columns as in
   `perimage_diagnostic.md:213-217`): (a) route-fire fraction on the 11 wipes (R1 needs ≥90%);
   (b) routed **post-gain** `evg_sum` sign — must flip from the parent's −5301 to ≥0, the phase
   correction N0023 could not perform; (c) `gain_on_neg` on KEEP/MID ≈1.08 — doubles as the
   cellcal-dispatch-fix check (§3.2).
3. **Mid-slice attribution of N0023's +4.21**: N0024 mid vs parent 37.50. If N0024 mid ≈ 37.5
   (vs N0023's 41.71), N0023's mid regression was caused by *cellcal removal* (the dispatch
   choice), not by touching `ev` per se; if N0024 mid still ≈ 41, the intervention family itself is
   implicated — this decides whether the two-regime F2-successor (`h0022_idea_DRAFT.md:130`) is
   worth ever booking.
4. **KEEP-guard / route precision (R3, F2)**: count non-routed KEEP images that moved >3% vs
   parent; partial-wipe set gt[250,500) route precision (claimed 71% precision,
   `h0023a_idea_DRAFT.md:40`); misroute counts both directions.
5. **Dense-block per-image (17 × gt>500)**: whether wipe rescue was paid for by new collapses on
   keeps (no image at ratio ≥0.8 may drop below 0.6; dense drop >3% on any routed-eligible image
   refutes) — directly comparable to N0023's dense 458.69.
6. **Checkpoint scalars in the same pass**: `w_c`, `w_p`, temp (expect w_c>0, temp==0.07);
   route-stability across epochs (F5) only if an ep16 dump exists — flag as optional, single-ckpt
   dumps cannot settle it.

Reads 1+2 settle the open causal question (does an in-phase substrate convert the confirmed gain
into mass where the mask could not?); read 3 settles the dispatch-vs-family attribution of N0023's
mid/late damage; read 2c settles that N0024 actually ran a *different* experiment than N0023.

---

## 5. Recommendation

**Proceed to the N0024 verdict when it lands — arbitrate by mechanism reads first** (R1 route fire +
routed evg_sum ≥0, R2 ≥7/11 wipe rescue, w_c≠0 proving the dispatch fix), bars second, with the
ep24 gate watch (≤20.00 to finish; a HALT in (20.0, 21.6) is draw-contaminated, not a clean kill) —
then, unless N0024 clears the triple bars with R1/R2 true (which would re-open the lineage under a
new live parent and void the stop rule), **book the registered third-card direction — the
exemplar-distinctness / exemplar-encoding interface card** (`HANDOFF.md:48` "remaining 残血 idea,
期望为负"; dense-targeted variant `HANDOFF.md:62`; H0022's F1/F3 successors
`h0022_idea_DRAFT.md:129,131`; H0023a's F1 successor `h0023a_idea_DRAFT.md:125`) as the third and
final post-HANDOFF card, and only trigger the HANDOFF *"3 cards no >1.5 → paper"* decision
(`HANDOFF.md:64`) after that third card also fails to deliver >1.5 net drop (or dies honestly at the
novelty/§11 gate as a re-encode). **Why this order:** with N0023 (+2.82, worse) and N0024 pending,
only two of the three registered cards will have run — stopping at two under-counts the explicit
HANDOFF budget and leaves the collapse-corpus story without its final localization probe (N0024's
F1-vs-F3 outcome tells the paper *which* interface — substrate or out/decoder — the wipe lives in);
the third card is the last unexhausted readout-side frontier, cheap, expected-negative, and its
negative result is itself paper-grade evidence — so the sequence is **N0024 verdict → book third
card → (if no >1.5) paper closure**, not paper immediately after N0024.

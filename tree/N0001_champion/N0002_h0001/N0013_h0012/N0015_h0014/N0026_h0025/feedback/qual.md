# Qualitative feedback — N0026_h0025 (`use_capfilm` CapFilm, child of N0015_h0014)

Text-log analysis per AGENTS §9/§3. Sources read directly this session (all local, no ssh, no
commit): node `idea.md` / `model.py` / `config.toml` / `info.json`; parent
`../model.py` decoder path; parent backup `/home/qkun/cac_backup/N0015_val_result.json`;
style gold `../N0025_h0024/feedback/{qual,diagnostic}.md`; leverage/S-curve premises in
`local/research/reexamination_20260922.md`; run/verdict record in `journal/events.jsonl:68,70`.

**Artifact caveat (honesty §5.5/§5.7):** no `result.json`, no `val_result.json`,
no `val_perimage.json` and no gamma/attr dump are local under the node (directory contains only
`idea.md` / `model.py` / `config.toml` / `info.json` / empty `feedback/`); local `info.json`
still reads `status: "proposed"` (server-side run not synced). Therefore: **slice numbers
(dense/mid/sparse) are UNMEASURABLE this session** (eval may still run on server), and every
R1/R2/R3 gamma read below is reported **not in dump** — neither claimed nor reinterpreted. No
gamma/alpha statistic is invented anywhere in this document.

Verified outcome (from `journal/events.jsonl:70` verdict line + futility arithmetic; slices pending):

| quantity | booked bar | N0026 | vs parent (19.3431 / 330.81 / 5.814) |
|---|---|---|---|
| best EMA val @HALT | ≤19.0431 | **23.232 @ep24** (journal:70; need ~0.52/ep ≫ 0.12) | **+3.89** |
| val @ full eval | ≤19.0431 | **UNMEASURABLE locally** (eval launched server-side) | — |
| dense gt>500 | <330.81 | **UNMEASURABLE locally** | — |
| sparse gt<50 | ≤5.45 | **UNMEASURABLE locally** | — |
| futility | ep24 HALT iff best >20.0031 | **HALT**, best 23.232, need 0.5236/ep > 0.12 | +3.23 above ceiling |
| R1a/b/c gamma reads | CV≥0.05, capacity load ≥1.5× | **not in dump** | — |
| R2 S-curve movement | gt≥300 ratio → ≥0.75 | **not in dump** (no child perimage) | — |
| R3 cond×2 leverage | ratio closer to 1 than 0.824 | **not in dump** | — |

---

## 1. What CapFilm was supposed to inject

The booked BECAUSE (`idea.md:8,13,29`) has two joints:

1. **Diagnosis (the consumption-interface fork of the closure OR).** After the leverage probe
   measured the trained parent as **fine=generator** (`fine=0` → count **−91.6%**;
   channel sens **15.2**) and **cond/simprior=net-suppressor** (remove residual → count
   **+29.9%**; `cond×2` → −17.6%; sens **1.6**, ~10× below fine), six evidence/residual-content
   cards (H0016/17 quiet-null; H0018/22/23 wipe 0/11) and then H0024's magnitude-on-residual
   (N0025, +2.88) all died editing a low-leverage suppressive path. What remained open was
   fork (a): the **decoder's consumption** of `cat[fine, cond_map]` — the only path with 10×
   sensitivity — plus the measured S-curve miscalibration
   (`reexamination_20260922.md:14–22`: gt7–10 ratio **1.40** over → gt138+ **0.685** under;
   mid50–300 **0.86** carrying **44%** of AE; bias −12.4; density max 0.458 / mean 0.029 =
   softplus head far from saturation). The claim: the sum is small because the **pre-head
   operating point** is trained toward the sparse-heavy conditional mean, not because the
   nonlinearity clips — a missing per-image absolute scale DOF at the consumption interface.
2. **Remedy (SNDM self-norm moved inside the decoder, CACViT magnitude made geometric).**
   A new ~25k-param `CapFilm` MLP maps **GAP(fine) ‖ log(ME)** — ME = mean_K(S²/(w_k·h_k)),
   pure exemplar-box geometry from the already-present `bboxes_in` (each annotation box
   structurally holds one object) — to a **per-image, per-channel (γ, β) pair of length 128**,
   and applies `h ← h·(1+γ) + β` on the decoder **block output, pre-final-1×1-head,
   pre-softplus** (`model.py:139–144`). Zero-init on the last Linear ⇒ step-0 (γ,β)=(0,0) ⇒
   `h·1+0 == h` bit-identical. Then the triple conjunctive bars: val ≤19.0431 AND dense
   <330.81 AND sparse ≤5.45, plus the R1 engagement disjuncts inside DISPROVED
   (CV of mean_c(γ) ≥0.05; mean γ on gt≥300 ≥1.5× mean γ on gt<50).

**Why decoder consumption was the theorized locus.** The two-card closure had localized the
dense-wipe *below* the evidence/gain layer; H0024 then tested absolute magnitude *on the
residual* and died (N0025). The reexamination (`reexamination_20260922.md:43–57`) named
decoder-side self-normalization / count-capacity conditioning as the explicitly open §11
direction, and option C ("FiLM/gate decoder by cond — decoder-side, addresses quiet-null root
cause"). First-principles: ĉ = Σ softplus(f_θ) under MSE on 69% gt<50 images estimates a
compressed E[ĉ|feat] — a monotone S-curve is the Bayes act without a per-image scale DOF;
multiplying the pre-softmax sufficient statistic by (1+γ(x)) restores one continuous absolute
DOF with **zero post-decoder additive**, and per-channel γ lets training amplify generator
channels while de-amplifying suppressor channels at high capacity — addressing the probe's
"cond=suppressor" asymmetry that a uniform output β (SNDM literal, gray vs the post-decoder
ban) cannot. Published neighbors verified in the survey: SNDM (arXiv:2203.09474), FiLM
(1709.07871), DensFiLM (2607.25465, decoder-bottleneck FiLM), CACViT (2305.04440, ME),
CountingDINO (2504.16570, unit-mass) — none fused at this site under our bans.

## 2. What the code actually does at the FiLM site (line-checked this session)

Diff of `N0026_h0025/model.py` vs parent `../model.py`, verified against `idea.md` §2.2/§2.4:

- **FiLM site** — parent decoder (`../model.py:139–140`) is a single line:
  `return F.softplus(self.head(self.block(x)))`. Child (`model.py:139–144`) threads optional
  kwargs: `h = self.block(x)`; `if gamma is not None: h = h * (1.0 + gamma[:, :, None, None])
  + beta[:, :, None, None]`; `return F.softplus(self.head(h))`. Intervention sits **after**
  both GN/GELU block stages and **before** the 1×1 `self.head` and softplus — exactly the
  booked pre-final-conv placement. `DensityDecoder.__init__` untouched (no decoder params, no
  parent RNG shift at decoder construction).
- **CapFilm module** (`model.py:293–315`): `Linear(129→64) → GELU → Linear(64→256)`, last
  layer zero weight+bias; forward computes `gap = fine.mean((2,3))`, box areas from `bboxes`,
  `me = mean_K(S²/area)`, `log(me.clamp(1, 1e4))`, chunk to (γ, β) each (B,128). As booked.
- **Call site** (`model.py:192–195`): `gamma = beta = None`; if flag and module exist →
  `gamma, beta = self.capfilm(fine, bboxes_in, self.S)`; `dens = self.decoder(cat[fine,
  cond_map], gamma=gamma, beta=beta)`. The SimPrior cascade (`:181–191`) runs unchanged
  **before** this — no N0023-style exclusive-dispatch; cellcal/peakcal stay reachable.
- **Attach order** (`model.py:351–356`): CapFilm constructed **after** PeakCal/simprior/
  cellcal/GCA — append-only RNG (rule 13); first Linear draws only inside CapFilm.
  Temp-pin path (`:349–350`, `use_peakcal=true`) untouched.
- **Flag** (`model.py:169`): `self.use_capfilm = _get(cfg, "use_capfilm", False)` — non-module,
  no RNG. Config delta is exactly one line `use_capfilm = true` (`config.toml:30`);
  `use_simprior/cellcal/peakcal` all true; `augment = false` in the file while launch carried
  `--set augment=true --set futility_bar=19.0431` (`journal/events.jsonl:68`) — same
  config-vs-runtime semantics gap siblings flagged.
- **Ban audit:** nothing added post-`decoder(...)` (GCA's pre-existing `dens+bias` at
  `:374` is parent code); no output multiply; not a global scalar (128-vector, CV gate);
  SimPrior/`ev`/sign/substrate byte-identical; no shared projections (own Linears — H0006/7
  family clean); no loss/temp/seed change. Launch shas recorded (`journal:68`: config
  `4b56e8e656961e0c`, model `917596dd5b2df04e`), green smoke pre-launch.

**The card executed what it booked.** No dispatch bug, no partial-test residue on the code side.

## 3. Outcome vs the remedy's predictions

Best 23.232 @ep24 is **+3.89 above the live parent 19.3431** and **+4.19 above the bar**
19.0431; futility HALTed at ep24 (need 0.5236/ep ≫ 0.12; above the card's own
draw-contaminated band (20.0, 21.6) by +1.63, above the worst same-seed draw 21.59 by +1.64).
Every booked bar that we can see (val via best) fails by mechanism-level margin; dense/sparse
slices are **UNMEASURABLE locally** (eval may still run on server). R1 (γ engagement +
capacity loading), R2 (S-curve movement) and R3 (cond×2 leverage re-probe) all require a gamma
dump / child perimage that do not exist locally — **the mechanism neither confirmed nor
cleared**. As with N0025, the card needed R4 ∧ R1; R4 already fails as booked on the val level,
so the verdict does not depend on the missing reads — but the *failure mode* does.

## 4. Honest weaknesses of the design (exactly the three the prompt names, plus what the level adds)

1. **Zero-init may stay quiet (the dominant priced risk, F1).** Last-Linear zero-init guarantees
   step-0 identity but also means γ escapes zero only iff joint MSE rewards capacity scaling in
   24–32ep. The gradient is nonzero at init (`∂L/∂γ = ∂L/∂h ⊙ h`, `idea.md:112`) — this is not
   a dead gate — but MSE on 69% sparse images may simply never reward it, collapsing CapFilm to
   a 25k-param bystander. The card pre-registered this as its dominant risk (`idea.md:181`)
   and built the R1 CV/capacity gates precisely to make it a *clean* quiet-null rather than a
   mystery. **Without the gamma dump we cannot say whether F1 fired.**
2. **ME is only 1-D log capacity.** `me = log(mean_K(S²/area))` clamped to [0, ~9.2] is one
   scalar per image: it carries order-of-magnitude box geometry but no spatial layout, no
   density regime beyond area, and is collinear with what GAP(fine) already summarizes
   globally. A 129→64→256 MLP on (128-D GAP + 1-D log ME) may learn essentially a
   GAP-conditioned affine — i.e. a global calibration in per-channel clothing (the
   H0002/H0010/H0015 dead family) unless ME genuinely separates capacity. The R1b capacity-
   loading read (≥1.5× gt≥300 vs gt<50) was the registered test of exactly this; not in dump.
3. **The decoder may renormalize.** The block contains GroupNorm after each conv; GN is
   per-sample per-channel normalization, so a FiLM applied **after** the block (on the GN+GELU
   output) is not immediately re-normalized at that site — but the *next* training steps adapt
   `self.head` (and, through gradients, the block) to absorb a fixed or slowly-varying affine,
   which is the same "normalized away" dynamics that killed H0016/H0017 on the residual path.
   FiLM avoids being *structurally* killed by GN (placement is post-GN), yet co-adaptation over
   24ep can still nullify it — the consumption-side analog of the quiet-null family. (Note the
   alternative placement inside the block would have been *pre*-GN and GN would provably erase
   per-channel scale; the booked post-block site is the better of the two, and still exposed to
   training-time absorption.)
4. **What the level adds (honest, dump-free).** best 23.232 sits **+1.64 above the worst
   same-seed draw (21.59)** and ~2.4σ above the draw-mean 20.54 — a pure quiet-null would put
   the child *inside* the parent's draw distribution, so the level is **more consistent with
   engaged-harmful (or at least not-quiet) degradation** than with F1; but draws are unbounded
   and no gamma statistic exists locally, so this is a consistency ranking, not a measurement
   (diagnostic §4 carries the full argument and the distinguishing read).

**Design-quality positives worth keeping:** strongest append-only form for a *parameterized*
card (CapFilm appended after every parent module, decoder `__init__` untouched, zero-init
identity); one boolean switch, one-line config delta; gradients nonzero at init (H0009 lesson
applied); no second grad path into confirmed readout (own Linears); survey-verified published
ancestry (SNDM/FiLM/DensFiLM/CACViT) with a genuinely novel in-head fusion; falsifier bars and
R1–R3 reads specific and pre-registered; expected-negative stance honestly recorded. The card
was a well-formed bet on the *right locus* (the probe's generator path) with the wrong-or-quiet
actuator left to the run to decide.

## 5. Comparison to sibling verdicts (N0023 / N0024 / N0025)

| | N0023 (H0022 margin) | N0024 (H0023 antimatch) | N0025 (H0024 canchor) | **N0026 (H0025 capfilm)** |
|---|---|---|---|---|
| site | evidence layer (`ev` sign) | evidence layer (`ev` substrate) | residual post-`out` ×α | **decoder block output FiLM (γ,β)** |
| parent gain stack | cellcal BYPASSED | cellcal retained weak | cellcal/peakcal as-booked | cellcal/peakcal as-booked |
| val / best @HALT | 22.163 (+2.82) | 23.062 (+3.72) | 22.213 (+2.88) | **23.232 best (+3.89)**; val eval UNMEASURABLE local |
| dense gt>500 | 458.69 (+127.9) | 491.81 (+161.0) | 454.05 (+123.2) | **UNMEASURABLE locally** |
| sparse | 5.703 (improved, over bar) | 5.972 (+0.16) | 5.779 (improved, over bar) | **UNMEASURABLE locally** |
| mechanism read | 0/11 wipes | 0/11 (7/11 routed) | not in dump | **not in dump (no γ)** |
| futility HALT ep24 | yes (need 0.403/ep) | yes (need 0.525/ep) | yes (need 0.397/ep) | **yes (need 0.524/ep)** |
| verdict | REFUTED | REFUTED | REFUTED | **REFUTED via futility-HALT (level)** |

Reading across the four: all four HALT at ep24, all four land in the **+2.8…+3.9 corridor**
above the live parent, all three with slices available blow the dense bar by +123…+161. The
arc is coherent and now spans **three distinct loci**: sign/phase/magnitude on the evidence
residual failed (N0023/24/25), and the **first decoder-consumption card also failed at a
larger level gap (+3.89, worst of the corridor)**. Two readings the dump must eventually
separate: either (i) the +2.8…+3.9 band is the shared structural signature of *any*
multi-thousand-param / multi-site intervention under the determinism crisis (sd≈1.13) with
mechanism-specific signal buried inside it — in which case CapFilm's specific failure (quiet vs
harmful) is still undecided — or (ii) each locus genuinely hurts and the corridor is
mechanism-caused on all four sites. What the level alone establishes: the consumption interface
did **not** clear the bar within the futility window, so decoder-side fork (a) loses its first
registered probe exactly as fork (b) lost H0024.

**Honest residue:** without the gamma dump we cannot say whether N0026 died as F1 quiet-null
(γ≈0 — MSE never rewarded capacity scaling, the card's dominant priced risk) or as
engaged-harmful/F2–F3 (γ moved, wrong-direction loading or decoder co-adaptation absorbed it).
The verdict as booked rests on the val level + futility HALT outside the draw band; any claim
of a *specific* failure mode without the γ dump would be invention. The +3.89 gap being above
every observed draw is *consistent with* engaged-harmful, not proof of it.

## 6. Booking-quality notes

- **Booked vs code: MATCH** (line-checked `model.py` vs `idea.md` §2.2/§2.4: FiLM site,
  CapFilm math incl. log-clamp ME, call-site gating, append-after-PeakCal, one-line config).
  No dispatch bug; smoke green pre-launch (`journal:68`).
- **Falsifier well-positeness:** three numeric bars parent-bound; the R1 engagement disjuncts
  (CV<0.05, capacity-loading failure) are empirically falsifiable report-style reads inside
  DISPROVED — sound structure, locally undecidable with no dump (same gap as N0025).
  Sparse ≤5.45 remains the standing parent-infeasible bar (parent 5.814) — footnote, not
  story (slices unavailable here anyway).
- **Process:** local `info.json` still `status: "proposed"` / `tested_hypotheses: []` — the
  run happened server-side and the tree copy has not been synced; do not misread as "never
  ran" (`journal:68,70` record launch + verdict). Runner-parseable
  `1. **H0025** — <text>` line IS present (`idea.md:209`, STATE gotcha 2 satisfied).
  `config.toml` says `augment = false` while launch carried `--set augment=true
  --set futility_bar=19.0431` — the usual config-vs-runtime gap.
- **Noise framing honored:** same-seed draws {19.34, 20.70, 21.59}, sd≈1.13; 23.23 sits above
  the worst draw and outside the card's own (20.0, 21.6) contaminated band — the refutation is
  the pre-registered bar + futility line, not a mechanism read we do not have.

## 7. What this rules in / rules out for the next card

**Rules out.** (a) This exact CapFilm booking — zero-init per-channel FiLM on the decoder
block output from GAP(fine)‖log(ME) — as booked, REFUTED on the val level + futility window.
(b) A silent retry of the same mechanism (rule 11): any successor decoder-amplitude card must
book as `contradicts`-evidence on H0025 with a genuinely new falsifier. (c) Any successor
arguing N0026 "ran out of epochs" — HALT is beyond every observed draw, need 0.52/ep ≫ 0.12.
(d) Quiet framing: the decoder-consumption fork has now spent its first registered probe and
missed by +4.19 vs the bar; the fork is *weakened*, not yet closed (only one card, no slices,
no γ read — unlike fork (b) which took three cards to close).

**Rules in.** (1) The similarity-side direction remains open per reexamination §43 and is
**already booked**: H0026 GemeCal (`N0027_h0026`, `journal:71`, `local/research/h0026_idea_DRAFT.md`)
— geometry ME power-law scale on SimPrior `ev`, +1 param, orthogonal site, completes the
three-site localization trilogy (similarity / decoder / post-out residual). Proceed as queued.
(2) Operational lesson (inherited from N0025, now binding twice): **ship the mechanism dump**
(γ histogram + per-image mean_c(γ) + capacity-loading split with the run) so R1 is locally
decidable — this card's biggest analysis gap is the missing γ dump, not the missing bars.
(3) When server eval lands, pull `val_result.json`/`test_result.json` and append slice numbers
as a quant addendum; dense/mid/sparse here are explicitly UNMEASURABLE, not estimated.
(4) Exemplar-distinctness remains the registered encoding-layer fallback if similarity-side
also fails.

---

**Files written:** this document only — `feedback/qual.md`. No commit, no ssh, no ledger /
`info.json` / `quant.md` edits by this role.

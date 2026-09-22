# Causal feedback — N0026_h0025 (H0025 `use_capfilm`, CapFilm decoder-block FiLM)

Causal agent, 2026-09-22. Inputs read this session (all local, no ssh, no commit): AGENTS.md §3/§5/§6/§11, STATE.md, NODE `model.py` / `idea.md` / `config.toml` / `info.json`, parent `../model.py` (decoder path) + `../config.toml`, `journal/events.jsonl:65–71`, `memory/hypotheses.jsonl:46–47`, parent dumps `/home/qkun/cac_backup/N0015_val_{result,perimage}.json`, leverage/S-curve premises `local/research/reexamination_20260922.md`, sibling gold `../N0025_h0024/feedback/{quant,causal}.md`, sibling endpoints `../N002{3,4,5}*/result.json` + `N0025 val_result.json`, draw stats `local/research/perimage_diagnostic.md:229`, futility rule `src/cac/engine/runner.py:151–191`.

Labels: **CONFIRMED** = file-backed this session · **CONSTRUCTION** = forward-logic identity from `idea.md`/`model.py` (true by arithmetic) · **PLAUSIBLE** = mechanism inference consistent with the files, not ablated.

**Artifact status (load-bearing):** under the node there is **no** `result.json`, **no** `run/` (tb server-side only), **no** `val_result.json`, **no** `val_perimage.json`, **no** γ/attr dump, **no** test dump. Local `info.json` still reads `status: "proposed"` (unsynced server run — `journal:68,70` record launch + verdict; not a "never ran" signal, not to be hand-edited). Every child-side mechanism read requiring γ or per-image fields is reported **not in dump**.

---

## 0. Verified outcome (CONFIRMED)

- **`journal:70` (verdict):** H0025 futility-HALT **ep24, best 23.232**, need ~0.52/ep vs threshold 0.12 → HALT; val **+3.89 vs live 19.3431**; full val/test eval launched, slices pending. **`journal:68` (launch):** green smoke, config_sha `4b56e8e656961e0c`, model_sha `917596dd5b2df04e`, `--set augment=true --set futility_bar=19.0431`, seed 20260830 v2.
- Full-precision endpoint (23.232019424438477@24, elapsed_s 1339.6, `futility_hit=true`, `n_epochs_done=24`) belongs to a `result.json` that is **not on local disk** (grep this session: no local hit) — carried as Lead-provided, locally unverifiable; causal claims use only 23.232 @ep24 + the journal HALT line.
- **Futility arithmetic (rule `runner.py:184–188`):** ceiling `19.0431 + 0.12×8 = 20.0031`; need `(23.232−19.0431)/8 = 0.5236`/ep ≫ 0.12 → **HALT**, best **+3.229 above ceiling**, **+3.889 vs live**, **+4.189 vs bar**; outside the card's own draw-contaminated band (20.0, 21.6) by **+1.63** (`idea.md:184`), above worst same-seed draw 21.5928 by **+1.64**, ≈**2.4σ** above draw-mean 20.544 (draws `perimage_diagnostic.md:229`).
- Parent: live 19.3431; dump recompute 19.3259, dense 330.8131, sparse 5.8136. Child dense/sparse/mid/test: **not in dump**.
- Config delta vs parent = exactly `use_capfilm = true` (diff `30d29`); `use_simprior/cellcal/peakcal` all true; file `augment=false` vs launch `--set augment=true` (config-vs-runtime gap, as siblings).

---

## 1. Construction claims vs measured reads

### CONSTRUCTION (from `model.py` + `idea.md` — true before any run)

| # | claim | anchor |
|---|---|---|
| C1 | FiLM site: parent `return F.softplus(self.head(self.block(x)))` → child `h = self.block(x)`; `h ← h·(1+γ[:,:,None,None]) + β[:,:,None,None]`; `return softplus(self.head(h))` — after both GN/GELU block stages, **before** 1×1 head + softplus; `DensityDecoder.__init__` untouched | `model.py:139–144` vs parent `../model.py:139–140` |
| C2 | `CapFilm`: `Linear(129→64) → GELU → Linear(64→256)`, **last Linear zero weight+bias** → (γ,β)=(0,0) at init; inputs `GAP(fine)` (128) ‖ `log(ME.clamp(1,1e4))`, `ME = mean_K(S²/area)` from annotation boxes; chunk to two (B,128) vectors | `model.py:293–315`; `idea.md:59–80` |
| C3 | Step-0 identity: γ=β=0 ⇒ `h·1+0 == h` ⇒ forward bit-identical to parent | `model.py:141–143`; `idea.md:144` |
| C4 | Call site computes γ,β only if flag + module exist; SimPrior cascade (cellcal/peakcal) runs unchanged **before** it — no N0023-style exclusive dispatch; temp-pin untouched | `model.py:192–195`, `:181–191`, `:349–350` |
| C5 | CapFilm constructed **after** PeakCal/simprior/cellcal/GCA — append-only RNG (rule 13); flag is non-module (no RNG); first Linear draws only inside CapFilm | `model.py:169`, `:351–356` |
| C6 | ~25k own params, own Linears (no second grad path / shared projection into confirmed readout — H0006/7 family clean); no post-decoder additive, no output multiply, no loss/temp/seed change | `model.py:293–315`; ban audit in `feedback/qual.md:100–104` |
| C7 | Gradient nonzero at init: `∂L/∂γ = ∂L/∂h ⊙ h` with `h` generically nonzero — not a structurally dead gate (H0009 lesson) | `idea.md:112` |
| C8 | Config one-line delta, pluggable by single switch `use_capfilm` | `diff` → `use_capfilm = true` only |

### MEASURED (only what the local files contain)

| read | bar (`idea.md:165–168`) | measured | verdict |
|---|---|---|---|
| **R1a** CV of per-image `mean_c(γ)` | ≥0.05 | **not in dump** | **unmeasurable** (cannot PASS; booked mechanism falsifier if CV<0.05 once dumped) |
| **R1b** capacity loading mean γ\|gt≥300 ≥1.5× mean γ\|gt<50 | ≥1.5× | **not in dump** | **unmeasurable** |
| **R1c** ≥25% channels positive mean γ on gt≥300 | ≥25% | **not in dump** | **unmeasurable** |
| **R2** S-curve movement (gt≥300 ratio →≥0.75 etc.) | see booked | **not in dump** (no child per-image) | **unmeasurable** |
| **R3** cond×2 ratio closer to 1 than parent 0.824 | see booked | **not in dump** (probe not re-run) | **unmeasurable** |
| **R4 val** | ≤19.0431 | **23.232** (`journal:70`) | **FAIL** (+4.189) |
| **R4 dense / sparse / test** | <330.81 / ≤5.45 / — | **not in dump** | **unmeasurable** |
| Futility ep24 | need ≤0.12/ep | need **0.5236** → HALT | **CONFIRMED HALT** |
| Config one-line delta | exactly `use_capfilm=true` | confirmed by diff | **PASS** |
| Launch gates (smoke, shas, overrides) | `journal:68` | recorded; log server-side | **Lead-recorded / not re-verifiable from local run artifacts** |

**Booked summary:** R4-val FAIL · R4-dense/sparse, R1/R2/R3 **not in dump** · futility HALT CONFIRMED · C1–C8 CONFIRMED from code. **Verdict: REFUTED via futility-HALT on the measurable val bar**; mechanism engagement neither confirmed nor cleared locally — the failure *sub-class* (F1 quiet-null vs F2/F3 engaged-harmful) is **INDETERMINATE** without the γ dump.

---

## 2. Structural contrast: where the actuator sat

- **Parent path (CONFIRMED):** `dens = softplus(head(block(cat[fine, cond_map])))` — `block` = Conv→GN→GELU→dil-Conv→GN→GELU (hidden 256→128), `head` = zero-bias 1×1 (`../model.py:128–140`). The decoder integrates `fine` (generator) + `cond_map` (incl. SimPrior residual).
- **Child path:** identical except the per-image per-channel affine `(γ,β)` applied on the **block output feeding `head`** (C1). Residual content, cond concat, cellcal/peakcal, Condenser, FineFuser, GCA, backbone: byte-identical (C4–C6). Unlike N0023, no call-site dispatch bug; unlike H0024's scalar α on a 64-ch residual, this is a 128-vector affine on the generator-path hidden.
- **What γ can and cannot do (CONSTRUCTION):** it rescales/shifts the pre-head sufficient statistic per channel — `softplus((1+γ)·head_in + …)` can move integrated mass without any post-decoder additive. It cannot change the input features themselves, cannot spatially restructure beyond a uniform-per-channel affine, and cannot escape co-adaptation: `head`/`block` weights keep training against it for 24 epochs.

---

## 3. Why CapFilm likely failed — three file-backed hypotheses (ranked; none measured)

**#1 — Fine-dominated decoder ⇒ engaged-harmful leverage (PLAUSIBLE; most consistent with the level; NOT measured).** The leverage probe (`journal:65`, `idea.md:8`, recorded premises not re-run): fine=0 → count **−91.6%**, fine channel sens **15.2** vs cond **1.6** (~10×), residual removal → **+29.9%** (net-suppressive), cond×2 → −17.6%. CapFilm sits **on the generator path proper** (`h` feeds `head` directly, C1) — the single highest-leverage tensor in the head. Consequences: (i) a *nonzero wrong-direction* γ here is *capable* of +2…+4 MAE damage, unlike every residual-path card whose edits were absorbed or wiped; (ii) gradients are nonzero at init (C7), so nothing structurally blocks escape from zero; (iii) the observed level — **23.232 = 2.4σ above draw-mean, +1.64 above the worst draw, +1.63 above the card's own contamination ceiling** — is the signature of an *added harmful DOF* (precedent: H0003 padapt "active but harmful" +3.00, STATE.md:7), not of an inert module inside the parent draw cloud. **What would confirm:** γ dump with **CV ≥0.05** (R1a) → engaged-harmful; loading **backwards** (mean γ|gt≥300 < mean γ|gt<50) → F2 specifically (ME/GAP not discriminative for demand). Until then: most-consistent, not measured.

**#2 — Decoder renormalization / co-adaptation absorbed the affine (PLAUSIBLE design-level; consumption-side quiet-null).** `block` contains GroupNorm (`model.py:131–133`); the booked FiLM is **post-GN** (`:140–143`), so GN does not *provably* erase γ at the site — the better of the two possible placements (pre-GN would be provably normalized away). But 24 epochs of joint MSE can retune `head` (and, through gradients, `block`) to cancel a slowly-varying per-channel affine — exactly the "MSE lets the decoder normalize away perturbations" dynamic the reexamination names as the quiet-null root cause (`reexamination_20260922.md:8–10`; H0016/H0017 precedent on the residual path). This predicts epoch-wise γ drift toward 0 or a constant — **distinguishable only by a γ trajectory/histogram dump, which the run never produced.** Under this branch the card is F1-quiet by co-adaptation: the +3.89 then needs the draw-tail explanation (ranked below #1 because 23.232 sits outside every observed draw).

**#3 — ME log-compress too weak / covariate inadequacy (WEAKEST; F2 sub-case).** `ME = log(mean_K(S²/area))` clamped to [0, ~9.2] is **one scalar per image** (`model.py:307–312`) — order-of-magnitude box geometry only, no spatial layout, plausibly collinear with `GAP(fine)`'s global summary. The 129→64→256 MLP can then learn a GAP-conditioned affine in per-channel clothing — a **global calibration in disguise**, shadow of the dead H0002/H0010/H0015 scalar family (`idea.md:155,182`), which cannot satisfy the capacity-loading gate even if CV passes. **Registered discriminator = R1b (≥1.5× gt≥300 vs gt<50)** — not in dump. Cannot be ranked above #1/#2 without γ; would show as CV≥0.05 with **no** gt-loading separation.

**Explicitly not claimed:** dense/mid/sparse regressions (UNMEASURABLE locally — unlike N0023/24/25 where dense +123…+161 outside the retrain band [300,361] carried structural proof); any γ statistic; wipe movement. The verdict does not need them: futility HALT outside the draw band + val-level miss = refutation as booked.

---

## 4. What the engagement read would have shown (R1/R3 — the missing discriminator)

| read | booked threshold | would have distinguished | status |
|---|---|---|---|
| **R1a** CV of per-image `mean_c(γ)` on val @ best.pth | ≥0.05 | **F1 quiet-null / #2 absorbed** (CV<0.05, flat γ) vs **engaged** (CV≥0.05) — the single decisive split | **not in dump** |
| **R1b** mean γ\|gt≥300 ÷ mean γ\|gt<50 | ≥1.5× | healthy **capacity loading** vs **F2 wrong-direction** (<1.0) vs scalar-shadow (≈1.0 with CV≥0.05) | **not in dump** |
| **R1c** positive-γ channel share on gt≥300 | ≥25% | per-channel selective use vs uniform dead/constant offset | **not in dump** |
| **R3** cond×2 count-ratio on trained child (parent 0.824), or fine-sens ≥16.7 | closer to 1.0 | whether consumption dynamics changed **at all** — "engaged but ineffective" vs "engaged and harmful" | **not in dump** |
| **R2** bucket pred/gt (child vs parent gt≥300 0.6228 recompute, booked ~0.685→≥0.75) | ≥0.75 | whether the S-curve actually moved at the consumption interface | **not in dump** |

All five were pre-registered inside DISPROVED (`idea.md:165–168`) precisely to make a quiet run a *clean* refutation rather than a mystery. With no γ dump and no child per-image, **neither F1 nor F2/F3 can be declared**; the level makes #1 (engaged-harmful) the more-consistent reading and #2/#1-quiet second — a consistency ranking, not a measurement (same residue N0025 recorded for α).

---

## 5. Relation to H0024's residual-scale refutation — decoder-fork status

- **H0024 (N0025) tested the residual arm of the closure OR:** post-`out` scalar α on the SimPrior residual → **REFUTED, all three bars failed** — val **22.2127** (>19.0431), dense **454.05** (>330.81, +123.2), sparse **5.779** (>5.45) (`../N0025_h0024/val_result.json`), futility HALT need 0.397/ep (`N0025 result.json`); ledger `memory/hypotheses.jsonl:47` contradicts w=0.85. Reading: absolute magnitude on the **weak, net-suppressive** residual path cannot move integrated mass.
- **H0025 tested the opposite arm by design** (`idea.md:118`: "the two cards test **opposite forks** of the closure OR … If both fail, the OR closes with two measured negatives"): the **decoder consumption interface** — the probe's generator path (fine sens 15.2, 10× cond) — via per-channel FiLM pre-head.
- **Result of the first decoder-consumption probe:** level refutation **+3.89** (need 0.5236/ep ≫ 0.12, outside every draw), slices/γ pending. **Decoder fork status: SPENT ITS FIRST REGISTERED PROBE AND MISSED — weakened, not yet closed.** Unlike the residual fork (three cards N0023/24/25, all with full slices + mechanism reads), this arm has **one card, no slices, no γ read**; a synthesis claiming the consumption interface *structurally* cannot work would overclaim the evidence. What is closed as booked: **this exact CapFilm booking** (zero-init per-channel FiLM from GAP‖log-ME at this site, one-line switch) — rule 11: any successor decoder-amplitude card must book `contradicts` on H0025 with a **genuinely new falsifier**, not re-tune γ.
- **Two-card closure OR after N0026:** residual-scale arm measured-negative (N0025, +2.88) **and** decoder-consumption arm measured-negative at level (N0026, +3.89) — the localization claim ("wipe locus sits below the evidence/gain layer; content/scale edits at either downstream site do not clear the bar") now has **both arms file-backed**, while the **similarity-side/query direction remains open** and is already the next card: **H0026 GemeCal (`N0027_h0026`, `journal:71`)** — geometry ME power-law scale on SimPrior `ev`, +1 param, third distinct locus of the trilogy (similarity / decoder / post-out residual).

---

## 6. Causal chain — file-backed vs blocked

| arrow | status | numbers / anchors |
|---|---|---|
| A. Flag wired, one-line config, C1–C8 construction | **CONFIRMED** | `model.py:139–144,169,192–195,293–315,351–356`; config diff `use_capfilm=true` only; `journal:68` shas |
| B. γ engagement / capacity loading (R1) | **not in dump** | no γ, no CV, no loading split |
| C. S-curve / slice / wipe movement (R2, R4 dense/sparse) | **not in dump** | no child per-image / val_result |
| D. Consumption-dynamics change (R3) | **not in dump** | cond×2 re-probe not run |
| E. Level outcome | **CONFIRMED** | best 23.232 @ep24, futility HALT need 0.5236/ep, +3.889 vs live, +4.189 vs bar, outside draw band |
| F. Actuator sits on fine=generator path (10× leverage) | **PLAUSIBLE** (probe premises CONFIRMED as recorded in `journal:65`/`idea.md:8`; not re-ablated on N0026) | fine=0 → −91.6%; sens 15.2 vs 1.6 |
| G. Residual arm already refuted (H0024) | **CONFIRMED** | N0025 val_result 22.2127/454.05/5.779; `hypotheses.jsonl:47` |

Elimination available **without** γ: the card needed R4 ∧ R1; R4-val fails decisively outside the draw band ⇒ refuted regardless of B. What cannot be closed locally: which failure mode (#1/#2/#3, F1 vs F2/F3) and whether dense/sparse moved like the siblings.

---

## 7. Single most load-bearing causal statement

**CapFilm put a zero-init per-channel FiLM on the one tensor the leverage probe measures as the generator (fine sens 15.2, fine=0 → −91.6% count) — the right locus by the two-card closure, but a high-leverage actuator whose escape from zero is loss-optional and whose affine the decoder can co-adapt away over 24ep of joint MSE, while the ME side-channel is a single clamped log-scalar plausibly collinear with GAP(fine) — and the run confirms no booked gain: futility HALT at ep24, best 23.232 (need 0.5236/ep > 0.12, +3.229 above ceiling 20.0031, +3.889 vs live 19.3431), outside the card's own (20.0, 21.6) draw band and above every observed draw; γ engagement (CV, capacity loading), consumption-leverage re-probe, and all slices are not in the local dump and must not be claimed either way — the decoder-consumption fork is weakened by one level-refutation, not closed, and H0025 may not be silently retried (rule 11).**

*Noise note carried: draws {19.3431, 20.697, 21.5928} mean 20.544 sd≈1.13; 23.232 ≈ 2.4σ above the mean. The refutation rests on the pre-registered val bar + futility rule, not on a mechanism read this document does not have.*

---

## Source citation per section

| § | claims | source |
|---|---|---|
| 0 | endpoint, futility, parent bars, draw stats | `journal/events.jsonl:68,70`; `runner.py:151–191`; parent `/home/qkun/cac_backup/N0015_val_{result,perimage}.json`; `local/research/perimage_diagnostic.md:229`; missing `result.json` verified by directory listing + grep |
| 1 C1–C8 | construction | NODE `model.py:139–144,169,192–195,293–315,349–356`, parent `../model.py:128–140`, `idea.md:59–80,112,144`, config diff |
| 1 measured rows | R1–R4 / futility | bars `idea.md:165–168`; child values from `journal:70` or **not in dump** |
| 2 | decoder path contrast | NODE `model.py:128–144,181–195`; parent `../model.py:128–140` |
| 3 | three failure hypotheses | leverage premises `journal:65` + `idea.md:8`; GN/co-adapt `model.py:131–143` + `reexamination_20260922.md:8–10`; ME scalar `model.py:307–315` + `idea.md:155,182`; level/draw arithmetic §0 |
| 4 | engagement reads | `idea.md:165–168`; parent bucket recompute from `N0015_val_perimage.json`; child **not in dump** |
| 5 | H0024 relation, fork status | `../N0025_h0024/{result,val_result}.json`, `memory/hypotheses.jsonl:47`, `idea.md:118`, `journal:71`, `reexamination_20260922.md:43–57` |
| 6–7 | chain + headline | all of the above |

*Files written: this document only — `feedback/causal.md` (paired with `feedback/quant.md` same session). No `qual`/`diagnostic`/`synthesis` edits, no commit, no ssh, no ledger/`info.json` edits by this role.*

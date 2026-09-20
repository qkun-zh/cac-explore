# Synthesis — N0010_h0009 (H0009 `use_densexp`) vs live parent N0002_h0001

- **Node:** N0010_h0009 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Tested:** H0009 SOLO (`use_densexp=true`, `use_simprior=true` inherited)
- **Outcome (result.json, verified):** child best EMA val MAE **24.0500 @ep32**, 32/32 epochs stepped, `budget_hit=false` (`elapsed` 1789.42 s / `elapsed_s` 1794.9 s wall, 5.1 s UNDER τ_max 1800), `code: ok`, status `done`. Margin **+1.4858 worse** than parent same-seed; bar **≤22.26** missed by **+1.7900**.
- **Verdict: REFUTE H0009 as null-and-dragging (mechanism rejected by the joint optimizer AND the dead branch still cost +1.49; harness excluded).** Global-bar conjunct of the pre-registered dual falsifier (idea.md:29) fails outright at ~50× paired-contrast precision (~0.03) in the wrong direction, on a clean late-plateau number with no truncation confound. Dense-tail conjunct (gt>500 mean |Δ| vs 511.7) not evaluable from run scalars (no per-image dump) and given no support by the n=1 probe anecdote — left unevaluated, but the global miss alone decides. Evidence event stays Lead-only via CLI; this file proposes bookings only.

## 1. Consolidated reads (quant + qual + causal + diagnostic)

- **Quant (REFUTE):** never crosses bar (min 24.0500 over 32 eps, 1.79 above 22.26); never matches parent (child best sits where the parent was between ep12 24.2563 and ep13 23.9073, then plateaus). Separation from ep01 (+117.27 @ep01, +68.08 @ep02, +46.00 @ep03, narrowing monotonically to +1.49) — interference from the start, never a tracking window. Curve shape is monotone descent all 32 epochs into a dead-flat floor: best=final @ep32 (delta +0.0000), last-5 range 0.0154, late crawl ≈ −0.0039/ep (≈460 epochs to close the 1.79 gap — not a rescuable optimum). Train `train/loss_epoch` final **5.3450** (min 5.3191 @ep22, flat last 10) vs parent **2.6183** (**+2.73**, 2.04× ratio) — train-worse AND val-worse = underfit/optimization drag, not overfit. Val RMSE is the one scalar moving toward the parent: child best=final **84.1496 @ep32** vs parent best 88.9234 @ep28 / final 89.2400 (**−4.77 best-best, −5.09 final-final**, monotone 32/32 descent, no N0008-style divergence); RMSE/MAE ratio falls (3.95→3.50) while MAE degrades — a distribution-shape observation only, since the bar is written in MAE. `train/mae` all-NaN on both runs = standing logger artifact.
- **Qual (clean run, honest miss):** smooth healthy descent to a flat plateau — zero spikes, zero NaN, zero U-turns; first ~10 epochs do the lifting (270→29.7), ep11–24 grind (28.0→24.1), ep25–32 dead-flat (24.10→24.05). Best @ep32 is plateau-edge best, not a selected minimum: trustworthy checkpoint, but slope (~−0.005/ep) says extra epochs close nothing of the +1.49. Clean by every text-log check (0 hits for Traceback/Error/NaN/BudgetStop/OOM; `code=ok`, `budget_hit=false`, canonical seed 20260830, ~31.349M params consistent with ~4.3k appended expert). Family contrast: none of the known failure signatures — no N0007 stall attractor, no N0008 NaN wall, no N0005/N0006 timeout. Shape tell handed to causal: MAE flat while RMSE still falls late (−1.1 over last 8) — the mirror of N0008's pre-collapse decoupling, on a node whose pitch was "fix the catastrophic tail".
- **Causal (null-and-dragging — primary):** one-switch test verified (4,322 params: exp1 2080 + exp2 2112 + out 65 + gate 65; `dens = base + g*r`, zero-init out, detached GAP gate, decoder in_ch 192 untouched, no S recompute, no shared projection, no temp touch by construction). Checkpoint probe: expert functionally NULL — `out.weight` norm **0.0090** (init 0), gate learned CLOSED (gate weight 0.6018 off zero-init, bias −0.2381), trunk convs decayed **−18%/−28% below init scale** (exp1 2.6888 vs ~3.27, exp2 3.2833 vs ~4.58). Functional probe (5 val images, gt 7→2092): gate ≤0.15 everywhere incl. gt=2092 (0.072 sparse → 0.145 dense: relatively 2×, absolutely closed), r RMS 3 orders below operating range, base-vs-gated sums <2%. Lens-3 cheap-falsification branch fired cleanly: the optimizer evaluated the branch and voted it off. Yet hosting the dead branch still cost +1.49 with **temp collapsed 9.0× toward the clamp** (0.03932→0.004367, only 4.4× above floor), `simprior.out` re-amplified **+80%** (0.5874→1.0573), q +33.4%, and BROAD head drift beyond the readout (decoder agg +17.4%, fuser +17.6%, GCA +24.0%; cond/exemplar intact). N0009's operating-point re-centering theory (`softplus(L+r)` rescaling) CANNOT explain this node because here r≈0 means no operating-point shift — yet the same temp-to-floor + re-amplification + stall attractor appeared. Sharpening runaway now spans four nodes (N0006 0.0214, N0008 0.0255, N0009 0.0033, N0010 0.0044 — every trainable head perturbation of N0002 collapses temp below 0.0393). Remaining plausible channel is joint-optimization drag from step 1 (extra gradient path into cond_map through the trunk + AdamW/wd competition deflecting into a worse basin where temp sharpens to compensate) — stated as inference, not measurement. Harness ruled out (shas re-verified, solo-switch diff, canonical protocol, append-only construction, 32/32 complete).
- **Diagnostic (CLEAN DONE — trustworthy, endorsed):** edge-best @ep32 is a flat floor (last-5 range 0.0154), not a dip and not rescuable; budget/harness ruled out (5.1 s under τ_max, near-ceiling elapsed = idle-plateau epochs); NaN clean (0 hits; all 32-pt series finite); TB complete (best @ep32 needs no following point at −0.004/ep against a 1.79 miss). MAE-worse/RMSE-better split cannot rescue (bar is MAE). Dense-tail conjunct unevaluated at scale; n=1 anecdote (gt=2092: base 301.6→gated 305.0, still massively undercounted) gives no support. Recommends `contradicts` w≈0.85 (docked below the 0.90 catastrophic/NaN class per the N0006 +1.02 precedent); causal concurs w=0.85.

## 2. Quality gate (AGENTS step 8; K_SYNTH=2)

**(a) Format gate — outputs pasted verbatim:**

`python3 scripts/discovery.py validate --all`:
```
validated 9 hypotheses — 0 bad (all good)
```
(exit 0)

Zero new hypothesis texts are authored in §5 (0 bookings), so there is no per-booking `validate()`/`novelty_check.py` obligation — nothing to gate. Budget and dedup below.

**(b) Budget:** 0 bookings here ≤ K_SYNTH=2. No coordination debt: N0009 already synthesized at 0 bookings; this node is the last open card of batch-4 — nothing downstream holds allowance hostage.

**(c) No duplicates / no silent retries:** no new id is minted; nothing opposite-of-existing is proposed, so no `contradicts`-on-existing booking arises from this node. Dedup walk in §3 confirms nothing is revived. In particular no "smaller r / harder detach / re-gated / re-scaled" post-decoder additive is booked — the null member already failed, so no new falsifier can separate a retry (§11).

**(d) Remap-or-discard:**

- **Superseded — N0009 operating-point re-centering as the full mechanism:** N0009 proved detach-insufficient via shared-loss coupling (`softplus(L+r)` re-centering), but N0010 has r≈0 with full detach and still pays +1.49 with the same temp-to-floor + re-amplification + stall attractor. The operating-point story is confirmed in outcome but too narrow in mechanism. Remapped, not discarded: the coupling channel is broader than one equation — any trainable head perturbation deflects joint optimization into the sharpening-runaway basin (§4.2).
- **Discarded — clean null (residual ignored ⇒ harmless):** gate ≤0.15, out 0.0090, trunk decayed, r 3 orders below range — the null is real, and it still cost +1.49 with 9× temp collapse and broad drift. Null ≠ free. Closed.
- **Discarded — overfit / late-minimum-lost:** train stalls WITH val at 2.04× parent from ep17; best=final on a flat floor. Underfit equilibrium, not overfit.
- **Discarded — truncation / edge-best fluke / near-timeout:** 32/32 epochs, `budget_hit=false`, 5.1 s under τ_max, monotone 32/32 descent with no uptick. Finished run that lost. Closed.
- **Discarded — "RMSE improved so partial credit":** bar is MAE; −5.09 RMSE with +1.49 MAE is a shape observation, not a rescue.
- **Discarded — "smaller r / harder detach" retry:** the null member (r≈0, fully detached, private, 4.3k) already failed — there is no smaller/detached-er point on this axis. Closed by the extended ban (§4.2).
- **Discarded — `train/mae` all-NaN:** standing logger artifact, parent-identical. Zero signal.

## 3. Dedup vs ledger H0001–H0009 (quality gate c)

- H0001 `use_simprior` — supported, load-bearing; dynamically perturbed again here (temp-to-floor 9.0× with `out` +80% despite null branch), not rebooked.
- H0002 `use_gca_cal` — contradicts-recorded; untouched.
- H0003 `use_padapt` — contradicts-recorded; untouched (objectness-modulated re-encode stays dead at the §11 new-falsifier gate — not revived here).
- H0004 `use_simbank` — timeout-null; this node is NOT a retry (post-decoder logits expert, no new keys) and its verdict stands alone.
- H0005 `use_verify` — contradicts-recorded (N0006, +1.02); untouched. Its temp-collapse lead (0.0214) is now the mildest of four — sharpening runaway spans N0006/N0008/N0009/N0010, recorded as pattern, not a booking.
- H0006 `use_h2pool` — contradicts-recorded w=0.95 (N0007, +25.28); untouched. Contrast: N0007 is day-zero structural corruption (shared-qproj reuse, train parked ~14 from ep1); N0010 is null-and-dragging (healthy curve, train stalls late at 5.35, val floor +1.49). Different attractor, same lesson direction.
- H0007 `use_simcal` — contradicts-recorded w=0.90 (N0008, +5.07+NaN); untouched. Contrast: N0008 is engaged-then-unstable (undetached S recompute + std Jacobians → NaN wall); N0010 is never-engaged-yet-still-dragging (detached proxy, private params, no NaN). Third flavor of readout coupling, not a duplicate.
- H0008 `use_hires` — contradicts-recorded w=0.90 (N0009, +3.89 engaged-suppressive); untouched. Contrast: N0009 engaged (−19.5% suppression, head norm 0.2250 off zero) with localized drift; N0010 null (out 0.0090, gate ≤0.15) with BROAD drift (+17–24% decoder/fuser/GCA). Sibling decoder-side failures bracket the family: engaged-harmful AND null-and-dragging both lose.
- H0009 `use_densexp` — refuted here; no retry booked. Any post-decoder additive variant re-encodes the falsified mechanism and is explicitly NOT booked per §11 (no new falsifier can separate it — the null member failed).

## 4. Portfolio consequences — the key updates

**(1) Explicit ban list (all four stand; the third is extended by this node):**

1. **Second-gradient-path ban:** ANY second gradient path into simprior projections is dead (similarity-side 4×: H0004-null, H0005 +1.02, H0006 +25.28, H0007 +5.07+NaN). Never share projections / second grad paths with the confirmed readout.
2. **Shared-projection ban:** never reuse `simprior.qproj`/`kproj` (or any confirmed-readout projection) in a new component — N0007 proved day-zero starvation from this alone.
3. **Trainable post-decoder additive ban (EXTENDED by N0010):** no trainable post-decoder additive on the density path, however detached, however private, however small. N0009 established it for the engaged case (detach-insufficient via loss coupling); N0010 extends it to the NULL case — r≈0 with full detach still paid +1.49 with 9× temp collapse. "Smaller r" or "harder detach" cannot escape a ban whose null member fails.
4. **Global-calibration ban:** global count calibration has NO headroom (best oracle affine on val worsens MAE per STATE diagnostic). No `c*D` rescale, no bias, no affine post-hoc as a hypothesis — diagnostic-only, never a node.

**(2) Trainable-head search on N0002 is 6× spent — both interfaces dead.** Similarity-side 4× dead (above) plus decoder-side 2× dead: H0008 engaged-harmful (+3.89, suppressive −19.5%, temp 11.8× to floor) and H0009 null-and-dragging (+1.49, gate ≤0.15, out 0.009, temp 9.0× to floor with broad +17–24% drift). N0002_h0001 (22.5641) stands as a local optimum under the canonical protocol: every tested trainable perturbation of either interface degrades or destabilizes, through the same joint-optimization sharpening-runaway channel (4/4 temp collapses). There is no remaining trainable-head direction that avoids all four bans without new user-authorized scope. This synthesis books 0 trainable directions and recommends consolidation plus the user decision points in §5.

**(3) Booking discipline (K_SYNTH accounting):** this node proposes 0. Portfolio trainable-direction spend from this node: 0. Honest options considered: (a) 0 bookings + recommend consolidation at N0002 + user decisions — TAKEN; (b) ≤2 bookings only if a genuinely new interface exists — examined and rejected: encoder-side is forbidden (backbone frozen, out of scope); data-side needs user authorization (augment A/B); per-image tail dump already exists as diagnostic tooling, not a hypothesis. Default-to-honesty holds: no new falsifiable mechanism survives the bans, so 0 is booked.

## 5. Bookings proposed (0 new) + user decision points

**No bookings.** Rationale: every trainable head component either opens a second grad path / shares a projection (bans 1–2), perturbs the density-path operating point or drags joint optimization into sharpening runaway even when null (ban 3, now including the null member), or is global calibration with proven zero headroom (ban 4). Booking another architectural variant would be outcome shopping against settled mechanisms. K_SYNTH spend: 0/2.

**Recommended evidence event (for the Lead to record via CLI — NOT recorded by this subagent):**

Strength justification: w=0.85 — clean full-schedule run (32/32, `budget_hit=false`, no NaN, harness excluded by sha + solo-switch diff + canonical protocol), decisive miss (+1.49 vs parent ≈ 50× paired-contrast precision, bar missed by 1.79), mechanism-measured null (gate ≤0.15, out 0.0090, trunk −18%/−28%) plus the 9× temp-collapse signature with broad drift; docked below the 0.90 catastrophic/NaN class per the N0006 +1.02 precedent (adopted from causal §6 / diagnostic §5; both concur w=0.85).

```
python3 scripts/discovery.py evidence H0009 --type contradicts --strength 0.85 --node N0010_h0009 --note "N0010_h0009 done-clean 24.0500 @ep32 (32/32ep, budget_hit false, trustworthy plateau) vs live parent 22.5641: +1.49 WORSE, bar missed by 1.79. Null-and-dragging: densexp gate <=0.15 incl gt=2092, out norm 0.0090, trunk -18%/-28%, r 3 orders below range; yet temp collapsed 0.0393->0.0044 (9.0x to floor), simprior.out +80%, broad drift decoder +17%/fuser +18%/GCA +24%, train 5.35 vs 2.62 (2.04x). Null post-decoder additive still drags. Harness excluded."
```

**Expected confidence after this one event (Eq.1, η=0.20, c=0.50; advisory only):** `c' = c − η·w·c = 0.50 − 0.20·0.85·0.50 = **0.4150**` — still `uncertain` (above the <0.25 refuted threshold), so H0009 is not yet classified refuted; a further independent contradiction event is needed to cross it. The Lead records the event; the ledger derives the stored confidence. **This synthesis does not record it and does not edit any ledger or confidence value.**

**User decision points (carried forward from N0009 synthesis §5 — no booking can proceed without these):**
1. **Augment protocol A/B** (STATE open question): authorize `augment=true` protocol variant? Data-side is the only untested lever that avoids all four bans entirely — needs explicit authorization, not bookable unilaterally.
2. **Test-set evaluation:** mission is test SOTA but the canonical protocol is val-only; authorize a test read of N0002 (and frozen-head post-hoc calibration diagnostics) as a consolidation step?
3. **Consolidation vs new scope:** accept N0002_h0001 as the local optimum and close trainable-head exploration, or explicitly scope a new interface outside the head (user-guided)? Recommendation: consolidate at N0002.

## 6. Calibration bin table (`python3 scripts/discovery.py calibration`, 2026-09-20)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)      7       1     14%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall          7       1     14%      -
reliability error (weighted |rate − pred_conf|): 0.357
WARNING: reliability error > 0.20 — confidence at test is drifting

=== Current Standings ===
hyp       confclass       tests
H0001    0.585uncertain       1
H0002    0.420uncertain       1
H0003    0.405uncertain       1
H0004    0.500uncertain       0
H0005    0.415uncertain       1
H0006    0.405uncertain       1
H0007    0.410uncertain       1
H0008    0.410uncertain       1
H0009    0.500uncertain       0
```

- Read: 7 tested hypotheses, 1 confirm (14% vs 0.50 mean pred_conf in the only populated bin); reliability error 0.357 (WARNING — confidence at test drifting, small-n bin). H0008 verdict now ledger-recorded (0.410, 1 test — shifts the table vs N0009's paste). H0009 verdict not yet ledger-recorded (Lead records via CLI) — table shifts on booking.

## 7. Live-parent reference (explicit, unchanged)

**N0002_h0001 (22.5641 @ep30) remains the live-parent reference — all future falsifier bars bind to it, not to this node.** Confirm iff child final val MAE **≤22.26** (≥0.30 below 22.5641) AND gt>500 dense mean |Δ| below 511.7, under the canonical protocol at seed 20260830.

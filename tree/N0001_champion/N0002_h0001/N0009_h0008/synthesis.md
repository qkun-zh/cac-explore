# Synthesis — N0009_h0008 (H0008 `use_hires`) vs live parent N0002_h0001

- **Node:** N0009_h0008 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Tested:** H0008 SOLO (`use_hires=true`, `use_simprior=true` inherited)
- **Outcome (result.json, verified):** child best EMA val MAE **26.4535 @ep31**, 32/32 epochs stepped, `budget_hit=true` (`elapsed` 1809.75 s vs τ_max 1800 s → **+9.7 s marginal**; `elapsed_s` 1815.2 s wall), `code: ok`. Margin **+3.8894 worse** than parent same-seed; bar **≤22.26** missed by **+4.1935**.
- **Verdict: REFUTE H0008 as used-and-harmful (engaged, not null; harness excluded).** Global-bar conjunct of the pre-registered dual falsifier (idea.md §0) fails outright at ~130× paired-contrast precision (~0.03) in the wrong direction, on a clean late-plateau number with no truncation confound. Dense-tail conjunct (gt>500 mean |Δ| vs 511.7) not evaluable from run scalars (no per-image dump) — left unevaluated, but the global miss alone decides. Evidence event stays Lead-only via CLI; this file proposes bookings only.

## 1. Consolidated reads (quant + qual + causal + diagnostic)

- **Quant (REFUTE):** never crosses bar (min 26.4535 over 32 eps, 4.19 above 22.26); never matches parent (child best sits where the parent was between ep08 27.2659 and ep09 26.2845). Separation immediate at ep1 (child leads only ep0 141.48 vs 153.29, then trails +6.10 @ep02, +6.18 @ep03, +5.29 @ep04, +4.18 @ep05, +3.33 @ep05… widening to +3.89) — no tracking window, unlike the simcal sibling's 5-epoch track. Curve shape is monotone descent into a deep floor: last-8 val/mae readings in 26.4535–26.4617, best-vs-final +0.0001, last-5-epoch range 0.0036. Train `train/loss_epoch` stalls with it: child final **6.0013** vs parent **2.6183** (**+3.38**, 2.3× ratio) — train-worse AND val-worse = underfit/optimization drag, not overfit. Val RMSE child best 96.6710 @ep12 / final 96.8673 vs parent best 88.9234 → **+7.75**; RMSE/MAE ratio falls (3.95→3.66) while both degrade — broad-based, not tail-concentrated. Timeout is a wall-clock technicality: 32/32 val points, best @ep31 with a following evaluated ep32; nothing about the flat tail suggests more epochs close a 3.89 gap at a 0.004/5ep crawl.
- **Qual (clean run, engaged-but-harmful):** zero NaN/inf in any series over 32 epochs (contrast simcal's ep10 wall); the ep3 train uptick (7.65→8.29, re-descends) is ordinary. Checkpoint kills the clean-null branch: hires `head.weight` norm **0.2250** off exact-zero init (bias 0.0461; guide/ref convs 3.70/3.40/3.10, GN ~4.5–4.6) — the residual learned a non-trivial function and dragged both train and val onto a joint floor. Isolation check fails dynamically: q/k hold parity (6.71/6.80 vs 5.73/7.36, no N0007 freeze-at-half) but **simprior temp collapses 0.0393→0.0033** (~10–12× sharper, pressed against the 1e-3 clamp) with `out` norm growing 0.587→0.933. Wiring verified structurally clean (`logits.detach()` + `h2.detach()`, own params, no S recompute, no shared projection, decoder in_ch 192, GCA untouched; flag-off op-identical) — the failure is dynamical, not a wiring bug. `train/mae` all-NaN on both child and parent is the standing logger artifact; text log is dynamics-silent (TB parsing was load-bearing).
- **Causal (isolation premise falsified — primary):** functional probe (real val batch, 4 imgs): r mean −0.502 / RMS 0.899 / max|·| 5.35 against L mean −6.44 / std 1.73 — the residual re-centers the softplus operating point. Count direction: base 42.46→34.20 (**−19.5%**, all 4 images suppressed) — a **net-suppressive bias**, not separable peak splits/fills. Readout drift: q +17%, k −7.5%, **temp collapsed 11.8× to the clamp floor**, `simprior.out` re-amplified +59%; decoder +3%, GCA −1.6% (head backbone intact — damage localized to hires + simprior temp/out). Mechanism: `detach()` blocks a direct second gradient path, but it does **not** isolate the readout, because `softplus(L+r)` rescales every gradient flowing back through L into the decoder/condenser/simprior stack — joint optimization drags temp to the floor and hires settles into suppression. Train stalls at 2.3× parent from ep8: the +3.89 is the joint equilibrium of this coupling, not seed luck or budget artifact. Harness ruled out (shas re-verified byte-identical, solo-switch diff, canonical seed/protocol, 32/32 complete).
- **Diagnostic (marginal overrun — accepted; cost-attribution rejected):** N0009's timing is byte-for-byte the N0006 signature (trainer-elapsed +60.2 s vs parent; +9.7 s over budget; +1.88 s/ep — identical to ±0.01 s/ep), and three different modules (simbank/verify/hires) show the same +1.85–1.89 s/ep to ±0.04 s — common cause outside the modules (server contention/host noise). Do NOT book "HiResDetail is expensive." Best 26.4535 @ep31 trustworthy: 32/32 epochs, 32 `val/mae` TB points, late best with a following evaluated epoch; `_BudgetStop` fired during/after the final epoch. Do NOT re-run for a `done` status (§5.1 outcome shopping). Diagnostic recommends `contradicts` w≈0.90; causal concurs w=0.90.

## 2. Quality gate (AGENTS step 8; K_SYNTH=2; coordination with N0010)

**(a) Format gate — outputs pasted verbatim:**

`python3 scripts/discovery.py validate --all`:
```
validated 9 hypotheses — 0 bad (all good)
```
(exit 0)

Zero new hypothesis texts are authored in §5 (0 bookings), so there is no per-booking `validate()`/`novelty_check.py` obligation — nothing to gate. Budget and dedup below.

**(b) Budget:** 0 bookings here ≤ K_SYNTH=2. Coordination: N0010_h0009 (densexp) is still an open card — this synthesis proposes ≤1 new direction (zero) and preempts nothing; N0010's synthesis keeps its full K_SYNTH=2 allowance.

**(c) No duplicates / no silent retries:** no new id is minted; nothing opposite-of-existing is proposed, so no `contradicts`-on-existing booking arises from this node. Dedup walk in §3 confirms nothing is revived.

**(d) Remap-or-discard:**

- **Superseded — diagnostic §4 "never readout corruption" line:** diagnostic's structural reading (ban obeyed by construction → "mechanism rejection, never readout corruption") is overtaken by the checkpoint evidence qual §3 / causal §2 surfaced: temp collapsed 11.8× to the floor with `out` re-amplified +59%. The corruption is **dynamical through the shared loss** (operating-point re-centering), not structural through a grad path. Remapped, not discarded: the ban held structurally and failed dynamically — which is exactly the §4 portfolio update.
- **Discarded — clean null (residual ignored):** head norm 0.2250 off zero-init with −19.5% functional suppression. Engaged, not ignored. Closed.
- **Discarded — overfit / late-minimum-lost:** train stalls WITH val at 2.3× parent from ep8; best-vs-final +0.0001 on a 10-epoch floor. Underfit equilibrium, not overfit.
- **Discarded — truncation / budget artifact:** 32/32 epochs, 32 val points, late best + following epoch; +9.7 s is the N0006-class marginal signature. Closed.
- **Discarded — "HiResDetail is expensive":** unattributable per diagnostic §2 (three-module ±0.04 s/ep parity). No cost booking.
- **Discarded — `train/mae` all-NaN:** standing logger artifact, parent-identical. Zero signal.

## 3. Dedup vs ledger H0001–H0009 + N0010 proposal (quality gate c)

- H0001 `use_simprior` — supported, load-bearing; dynamically perturbed by this node (temp-to-floor), not rebooked.
- H0002 `use_gca_cal` — contradicts-recorded; untouched.
- H0003 `use_padapt` — contradicts-recorded; untouched (objectness-modulated re-encode stays dead at the §11 new-falsifier gate — not revived here).
- H0004 `use_simbank` — timeout-null; this node is NOT a retry (post-decoder logits+h2, no new keys) and its verdict stands alone.
- H0005 `use_verify` — contradicts-recorded (N0006, +1.02); untouched. Its temp-collapse lead (0.0214) reappears here deeper (0.0033) — sharpening-runaway signature now spans three nodes (N0006/N0008/N0009), recorded as pattern, not a booking.
- H0006 `use_h2pool` — contradicts-recorded w=0.95 (N0007, +25.28); untouched. Contrast: N0007 is day-zero structural corruption (shared-qproj reuse, train parked ~14 from ep1); N0009 is dynamical coupling with parity q/k and a healthy-looking plateau at +3.89. Different attractor, same lesson direction.
- H0007 `use_simcal` — contradicts-recorded w=0.90 (N0008, +5.07+NaN); untouched. Contrast: N0008 is engaged-then-unstable (undetached S recompute + std Jacobians → NaN wall); N0009 is engaged-then-stalled (detached inputs, private params, no NaN — stall via loss operating point). Second flavor of readout coupling, not a duplicate.
- H0008 `use_hires` — refuted here; no retry booked. Any post-decoder additive variant (re-gated, re-scaled, re-routed residual into density logits) re-encodes the same falsified coupling mechanism and is explicitly NOT booked per §11 (no new falsifier can separate it — the coupling is through the shared loss itself).
- H0009 `use_densexp` (N0010, open card) — NOT preempted. Flag for its synthesis read (not a verdict here): densexp is also a post-decoder additive on density logits with a stop-gradient gate proxy — the same shared-loss coupling surface N0009 just proved sufficient for corruption. Its synthesis should apply the causal §5 guards (temp within 2× of parent through ep8; r RMS ≪ L std at ep1–2) and read a temp-to-floor trajectory as the predicted failure signature. No booking for or against it is made here.

## 4. Portfolio consequences — the key update

**(1) Detach / stop-gradient isolation is INSUFFICIENT — the §11 ban must expand from grad paths to the loss operating point.** N0009 obeyed every structural rule accumulated since N0007 (detached inputs, private parameters, no S recompute, no shared projection, decoder in_ch untouched) and still corrupted the confirmed readout: temp 0.03932→0.00334 (**11.8× collapse to the 1e-3 floor**), `out` re-amplified +59%, hires suppressive −19.5%. Mechanism per causal §5: any additive `r` in `softplus(L+r)` re-centers the operating point of the shared MSE(+L1) loss, rescaling every gradient that flows back through L into the decoder/condenser/simproper stack — joint optimization then finds a joint equilibrium (train 2.3×, val +3.89) in which the readout goes near-one-hot and the new branch suppresses. **New rule: no trainable post-decoder additive on the density path, however detached, however private.** Forward-read-only is not backward-safe (N0008 lesson); now: gradient-detached is not loss-coupling-safe (N0009 lesson). Remaining directions must avoid perturbing the operating point entirely.

**(2) Trainable-head search on N0002 is 5× spent — the similarity family (H0004-null, H0005 +1.02, H0006 +25.28, H0007 +5.07+NaN) plus the decoder-side family opener (H0008 +3.89, engaged-suppressive) all degrade the same confirmed readout through the same joint-optimization channel.** N0002_h0001 (22.5641) stands as a local optimum under the canonical protocol: every tested trainable perturbation of either interface degrades or destabilizes. The three remaining options are (a) frozen-head post-hoc calibration with NO training (zero-training oracle/diagnostic only — affine proven dead per STATE with ≤0.34 headroom, so this is a diagnostic, not a node); (b) data-side augment protocol A/B (needs user authorization per the STATE open question — not bookable unilaterally); (c) accept N0002 as the local optimum and consolidate. This synthesis books 0 trainable directions and recommends (c) plus the user decision points in §5.

**(3) Booking discipline (K_SYNTH accounting):** this node proposes 0. N0010_h0009 (densexp, already booked from N0008 synthesis §5) keeps its synthesis allowance intact — nothing here spends it. Portfolio trainable-direction spend from this node: 0.

## 5. Bookings proposed (0 new) + user decision points

**No bookings.** Rationale: any trainable head component that touches the density path re-centers the shared loss operating point (§4.1) — there is no remaining trainable-head direction that avoids the proven corruption channel without new user-authorized protocol scope (augment A/B) or a zero-training diagnostic frame. Booking another architectural variant would be outcome shopping against a settled mechanism. K_SYNTH spend: 0/2.

**Recommended evidence event (for the Lead to record via CLI — NOT recorded by this subagent):**

Strength justification: w=0.90 — near-maximal because the quantitative miss is decisive (+3.89 vs parent = ~130× paired-contrast precision, bar missed by 4.19, engaged-and-stalled with harness excluded and checkpoint-measured readout coupling on every diagnostic: temp collapse 11.8×, train stall 2.3×, suppressive −19.5%); docked 0.05 below the N0004/N0007 w=0.95 catastrophic level since there is no stall/NaN signature — clean plateau, marginal-timeout-only (adopted from diagnostic §4 / causal §5; both concur w=0.90).

```
python3 scripts/discovery.py evidence H0008 --type contradicts --strength 0.90 --node N0009_h0008 --note "N0009_h0008 timeout-marginal 26.4535 @ep31 (32/32ep, +9.7s over 1800s, trustworthy plateau) vs live parent 22.5641: +3.89 WORSE, bar missed by 4.19. Engaged-and-stalled: hires head norm 0.2250 off zero-init, suppressive -19.5% on probe batch; simprior temp collapsed 0.0393->0.0033 (11.8x to floor), out re-amplified +59%; train 6.00 vs 2.62 (2.3x). Detach-insufficient: shared-loss operating-point coupling with zero grad paths. Harness excluded."
```

**Expected confidence after this one event (Eq.1, η=0.20, c=0.50; advisory only):** `c' = c − η·w·c = 0.50 − 0.20·0.90·0.50 = **0.4100**` — still `uncertain` (above the <0.25 refuted threshold), so H0008 is not yet classified refuted; a further independent contradiction event is needed to cross it. The Lead records the event; the ledger derives the stored confidence. **This synthesis does not record it and does not edit any ledger or confidence value.**

**User decision points (no booking can proceed without these):**
1. **Augment protocol A/B** (STATE open question): authorize `augment=true` protocol variant? Data-side is the only untested lever that avoids the operating-point channel entirely.
2. **Test-set evaluation:** mission is test SOTA but the canonical protocol is val-only; authorize a test read of N0002 (and frozen-head post-hoc calibration diagnostics) as a consolidation step?
3. **Consolidation vs further search:** accept N0002_h0001 as the local optimum and close trainable-head exploration, or explicitly scope a new interface outside the head (user-guided)?

## 6. Calibration bin table (`python3 scripts/discovery.py calibration`, 2026-09-20)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)      6       1     17%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall          6       1     17%      -
reliability error (weighted |rate − pred_conf|): 0.333
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
H0008    0.500uncertain       0
H0009    0.500uncertain       0
```

- Read: 6 tested hypotheses, 1 confirm (17% vs 0.50 mean pred_conf in the only populated bin); reliability error 0.333 (WARNING — confidence at test drifting, small-n bin). H0008 verdict not yet ledger-recorded (Lead records via CLI) — table shifts on booking. `use_densexp` (H0009, N0010 open card) untested.

## 7. Live-parent reference (explicit, unchanged)

**N0002_h0001 (22.5641 @ep30) remains the live-parent reference — all future falsifier bars bind to it, not to this node.** Confirm iff child final val MAE **≤22.26** (≥0.30 below 22.5641) AND gt>500 dense mean |Δ| below 511.7, under the canonical protocol at seed 20260830.

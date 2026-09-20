# Synthesis — N0008_h0007 (H0007 `use_simcal`) vs live parent N0002_h0001

- **Node:** N0008_h0007 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Tested:** H0007 SOLO (`use_simcal=true`, `use_simprior=true` inherited)
- **Outcome (result.json, verified):** child best EMA val MAE **27.6371 @ep09**, 32/32 epochs stepped, `budget_hit=false`, `elapsed_s=1797.8`. Margin **+5.0729 worse** than parent same-seed; bar **≤22.26** missed by **+5.3771**. Epochs 10–32 all-NaN on every loss/metric series (23 dead epochs).
- **Verdict: REFUTE H0007.** Global-bar conjunct of the pre-registered dual falsifier (idea.md:4, §5) fails outright at ~169× paired-contrast precision (~0.03) in the wrong direction, then diverges to NaN. Dense-tail conjunct (gt>500 mean |Δ| vs 511.7) not evaluable from run scalars (no per-image dump) — left unevaluated, but the global miss alone decides. Evidence event stays Lead-only via CLI; this file proposes bookings only.

## 1. Consolidated reads (quant + qual + causal + diagnostic)

- **Quant (REFUTE):** never crosses bar (min 27.6371 over 32 eps, 5.38 above 22.26); never matches parent (child best sits below the parent's ep08 level — parent ep08 27.2659, ep09 26.2845). Collapse shape: 9 finite points of normal descent (139.95 → 27.64), then `train/loss` NaN from step 2249 (inside ep10) with every epoch-aggregated series NaN ep10–32 — `best_mae` froze at ep09 and `result.json` reports `done` / `budget_hit: false` over a diverged tail. Pre-collapse train at ep09 (5.8483) is marginally better than parent-at-same-epoch (6.1468, −0.30) while val is +1.35 worse; child RMSE rises ep06→ep09 (95.4843 → 97.4689, +1.9846) while child MAE still falls — MAE/RMSE decouple before the NaN. No budget confound (`elapsed_s` 1797.8 < 1800, 32/32 recorded).
- **Qual (dirty run, honest pre-collapse best):** ep1–5 child tracks parent almost exactly (139.9/153.3, 67.0/70.1, 48.2/48.8, 39.3/39.1, 33.9/33.5) — the zero-init contract held at step 0. Ep6–9 separates monotonically while healthy (+0.5, +0.75, +0.95, +1.35, widening every epoch, per-epoch gains decelerating −2.0, −1.7, −0.9, −0.6) — the miss was legible before anything blew up. Ep10–32 is synchronized train+val NaN, not overfit: no U-turn, no plateau, 23 uninterpretable epochs. Best @ep09 is a last-healthy-epoch best, not an optimum. Text log is silent about the collapse (only the final `val/mae: nan` line; 2 `nan` greps) — TB-only visibility.
- **Causal (second-gradient-path interference + calibration-backward instability — primary):** checkpoint probe (`best.pth` = ep9): `simcal.out` w 0.2743 / b 0.0518 off zero-init, `tau` 1.0→0.8888 — the new branch engaged off its clean null (calibrated-margin channel leads: ch1 0.2225 vs ch0 0.1604), so engaged-and-harmful, not ignored. Shared projections distorted off the working point: `qproj` 4.7036 vs 5.7314 (−18%), `kproj` 4.9405 vs 7.3547 (−33%); confirmed-readout trust starved to 37% (`simprior.out` 0.2187 vs 0.5874); `simprior.temp` over-sharpened ÷1.54 past the parent (0.0255 vs 0.0393, ÷2.75 from init) toward N0006-collapse territory (0.0214). Structural fact from code-read: recomputed `S` under `use_simcal` (`model.py:180-187`) carries no `.detach()`, so autograd flows from `simcal.out` through the μ/σ reductions, the SNR-softmax, and the `Z` normalization into the shared `simprior.qproj/kproj` — the exact risk N0007 synthesis §4.2 flagged. Backward pass carries `1/sd` and `1/sd²` Jacobians through `S.std()` plus `1/temp` gain from the sharpened softmax, double-driving the same projections; flat-`S` regions produce explosive updates → NaN at step 2249. Ordering: second path opens → q/k pulled off working point, trust starved, temp over-sharpened → ep6–9 MAE gap widens while RMSE already rises → gradient explosion → synchronized NaN ep10–32. Ruled out: harness/protocol (shas byte-identical server/local, seed 20260830, augment false, solo switch, 32/32, budget fine); decoder/cond compensation as prime mover (`decoder.head` 0.95 UNDER parent 1.29 — no compensatory doubling); `tau` runaway (`tau` 0.89 near init — blow-up did not come through the SNR scalar).
- **Diagnostic (early-best-then-diverge / runaway dynamics — accepted; overfit and stable-underfit rejected):** no U-turn exists (train collapses WITH val — a generalization gap cannot synchronize train+val to NaN); no plateau/convergence exists (child never reaches the parent's late-train regime). Prime suspect for the NaN trigger (structural read `model.py:259-269`, NaN-step activations never dumped): `Z = (S−μ)/(σ+ε)` with `ε=1e-5` amplifies flat-exemplar noise, `SNR = μ/(σ+ε)` saturates `w` toward the loudest background exemplar, both readouts go hard at once on noisy K=3 cosine (inherited temp 0.0255), tail errors spike (ep6–9 RMSE rise), MSE gradients explode at step 2249; `grad_clip=1.0` acts on finite grads, not non-numeric activations. Proven regardless: harmful-before-blow-up (widening healthy gap), forward/loss numerics (synchronized), engaged new path (out norm 0.27).

## 2. Remap / discard (quality gate d)

- **Remapped — decoder/cond growth as compensatory, not causal:** `decoder.head` under parent and `cond.out` only mildly above (4.18 vs 3.27, against N0007's 5.67) — features broke before the readout compensated. No booking.
- **Discarded — clean null (calibration ignored):** `simcal.out` 0.27, `tau` moved −0.11, both biases off zero bit-identically (shared upstream proves both branches engaged). Closed, not ignored.
- **Discarded — N0007-style frozen starvation as the whole story:** q/k 34–40% above N0007's 3.52/3.54 floor, temp sharper (0.0255) not stalled (0.0547), `simprior.out.bias` small (0.05) vs N0007's 0.33 drift. Different attractor — recorded as second flavor in §4, not a duplicate.
- **Discarded — `train/mae` all-NaN:** pre-existing logger artifact, parent-identical. Zero signal.
- **Process gaps (no booking — runner changes are §10 mechanism changes, not hypothesis bookings):** runner reports `done`/`code=ok` over a 23-NaN tail (needs a `nan_epochs` count in `result.json`); text log silent about the collapse (needs a per-epoch `val/mae` text line); no gate scalars logged (`tau`, `temp`, `out` norms recoverable only from checkpoints); no `last.pth` saved (post-collapse weights uninspectable). Repeated from the sibling read, now load-bearing.

## 3. Dedup vs ledger H0001–H0007 + N0007 proposal (quality gate c)

- H0001 `use_simprior` — supported, load-bearing; distorted by this node (q/k suppressed, trust starved), not rebooked.
- H0002 `use_gca_cal` — contradicts-recorded; untouched.
- H0003 `use_padapt` — contradicts-recorded; untouched (objectness-modulated re-encode stays dead at the §11 new-falsifier gate per N0004 synthesis §2b — not revived here).
- H0004 `use_simbank` (75-key token-max) — timeout-null; this node is NOT a retry (statistics over trusted K=3 volume, no new keys) and its verdict stands on its own; family consequence in §4.
- H0005 `use_verify` — contradicts-recorded (N0006); untouched. Its temp-collapse lead (0.0214) reappears here in `simprior.temp` (0.0255) — sharpening-runaway signature now spans two nodes, recorded as pattern, not a booking.
- H0006 `use_h2pool` — contradicts-recorded w=0.95 (N0007); untouched. Contrast: N0007 is day-zero corruption (ep1 gap +66, train parked ~14 from ep2, finite stall 47.84); N0008 tracks 5 epochs then degrades progressively to a NaN wall. Both close their branches: bank-resolution AND distribution-calibration.
- H0007 `use_simcal` — refuted here; no retry booked. A σ-floored / Z-clamped / detached-statistics calibration variant is a similarity-side-into-`cond_map` variant of a 4×-dead family (H0004-null, H0005 +1.02, H0006 +25.28, H0007 +5.07+NaN) and is explicitly NOT booked per the portfolio gate.
- N0007 Booking 1 `use_hires` (decoder-side detail residual, proposal-only, no ledger id) — NOT re-booked here. Booking below is structurally distinct: conditional tail capacity routed per image vs detail sharpness over h2 fine context (different evidence, different interface behavior, different null). No contradicts-booking applies (nothing opposite-of-existing is proposed).

## 4. Portfolio consequences

**(1) N0007 synthesis §4.2 conditional FIRES — family expands to any second gradient path into simprior projections.** N0007 predicted: "if N0008 also suppresses q/k or stalls temp, the family expands from 'second similarity bank' to 'any second gradient path into simprior projections'." q/k ARE suppressed (−18%/−33% off the 5.73/7.35 working point), so the expansion holds — with a distinct signature: temp over-sharpened (0.0255) where N0007 stalled (0.0547), NaN wall at ep10 where N0007 stalled finite at 47.84. Record two flavors: N0007 starvation-via-garbage-keys, N0008 distortion-plus-instability-via-calibration-backward. Both implicate the shared-projection pattern; neither implicates h2 specifics. The proposed gotcha line ("reuse of similarity evidence is recompute-detached or statistics-only with gradients blocked into simprior parameters") covers N0008 retroactively: SimCal was statistics-only in the forward but not in the backward — forward-read-only is not backward-safe.

**(2) Similarity-side-into-`cond_map` family now 4× dead — deprioritize the whole interface.** H0004 timeout-null, H0005 refuted +1.02, H0006 catastrophic +25.28, H0007 refuted +5.07 + NaN. The confirmed simprior readout (3-key, temp ~0.04, out norm ~0.59) is at a local optimum: every tested perturbation of the K=3 volume degrades or destabilizes. Remaining headroom lives downstream (decoder-side / density-field resolution, already 1 booking: `use_hires`) and in conditional capacity for the catastrophic tail (1 booking below).

**(3) Booking discipline (K_SYNTH accounting):** N0007 synthesis proposed 1 (`use_hires`); this synthesis proposes 1 (`use_densexp`, §5) and re-books nothing. Portfolio total 2 ≤ K_SYNTH=2. Per-node count here is 1 ≤ 2. Deliberately unbooked: detached-simcal retry (family-dead, §3), objectness-gated readout (needs NEW falsifier per §11 — deferred, not revived on identical bars), any residual-into-cond similarity variant (§4.2).

## 5. Bookings proposed (1 new; format gate verified)

**Booking 1 — NEW `use_densexp` (dense-tail expert residual; decoder-side conditional capacity, no similarity, no shared projection):**

IF dense-tail expert residual via use_densexp IN a solo single-switch child of live parent N0002_h0001 trained under the canonical protocol at seed 20260830 for 32ep/1800s with the module constructed AFTER all parent modules with append-only RNG and decoder in_ch untouched at 8k added params, THEN final EMA val MAE falls at least 0.30 below live parent 22.5641 to 22.26 or lower AND gt>500 dense-image mean |Δ| falls below the parent 511.7 baseline, BECAUSE a zero-init post-decoder expert branch gated per image by a stop-gradient dense-level proxy adds localized peak capacity on catastrophic-tail images where the single cond path saturates while staying silent on sparse images so sparse mass placement keeps the confirmed similarity readout isolated with own parameters only and no shared projection, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol at seed 20260830 AND gt>500 dense-image mean |Δ| is not reduced below the parent 511.7 baseline.

- **Why this and not a similarity fix:** attacks the dense-tail symptom (smeared peaks, severe under-count on gt>500 images per the STATE per-image diagnostic) at the post-decoder interface where N0005/N0007/N0008 never operated. Under the §4.1 shared-projection ban by construction — own parameters only, frozen-context read-only with a stop-gradient routing proxy, zero-init additive residual on density logits, decoder `in_ch` untouched, no `S` recompute, no `qproj`/`kproj` contact in forward or backward. Clean null reads as mechanism rejection (gate never opens or expert norm stays near zero); engaged-but-worse reads as capacity-misallocation rejection; neither confounds the simprior readout. This is NOT global count calibration: no affine rescale of counts (oracle-affine proven dead per STATE — no headroom), only additive localized capacity routed by a detached proxy.
- **Param envelope:** ~4.3k actual (two 1×1 expert convs 64→32→64 ≈ 4.2k + single-linear gate head ≈ 0.1k) inside the booked 8k envelope; head ≈3.529M → +0.004M; node total ≈ **31.349M ≤ 32M**.
- **Format gate self-check:** markers IF-IN-THEN-BECAUSE-DISPROVED-IF in order; no hedging; falsifier carries numbers (22.26 global + 511.7 dense secondary) bound to live parent 22.5641; canonical seed/protocol stated; solo single-switch; append-only RNG; decoder `in_ch` untouched. Verified: `validate()` → errors [], warnings [] (LEN 978); `novelty_check.py` → top sim 0.598 (H0007) < 0.82 NOVEL, no structural twin (dense-tail conditional capacity vs all similarity/verification ledger entries H0001–H0007). Not opposite-of-existing → new id correct.

## 6. Calibration bin table (`python3 scripts/discovery.py calibration`, 2026-09-20)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)      5       1     20%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall          5       1     20%      -
reliability error (weighted |rate − pred_conf|): 0.300
WARNING: reliability error > 0.20 — confidence at test is drifting

=== Current Standings ===
hyp       confclass       tests
H0001    0.585uncertain       1
H0002    0.420uncertain       1
H0003    0.405uncertain       1
H0004    0.500uncertain       0
H0005    0.415uncertain       1
H0006    0.405uncertain       1
H0007    0.500uncertain       0
```

- Read: 5 tested hypotheses, 1 confirm (20% vs 0.50 mean pred_conf in the only populated bin); reliability error 0.300 (WARNING — confidence at test drifting, small-n bin). H0007 verdict not yet ledger-recorded (Lead records via CLI) — table shifts on booking. `use_hires` (N0007 proposal) and `use_densexp` (§5 proposal) untested.

## 7. Recommended evidence event (for the Lead to record via CLI — NOT recorded by this subagent)

Strength justification: w=0.90 — near-maximal because the quantitative miss is decisive (+5.07 vs parent = ~169× paired-contrast precision, bar missed by 5.38, engaged-then-diverged with harness excluded and checkpoint-measured distortion on every diagnostic) with the N0007-predicted family expansion confirmed; docked 0.05 from the N0007 w=0.95 because the 23-NaN tail destroys late-epoch shape claims and the ep9-vs-ep30 checkpoint epoch mismatch leaves q/k norms partly age-confounded (adopted from diagnostic §9).

```
python3 scripts/discovery.py evidence H0007 --type contradicts --strength 0.90 --node N0008_h0007 --note "N0008_h0007 done 27.6371 @ep09 (32/32ep, budget fine, 23 NaN epochs ep10-32) vs live parent 22.5641: +5.07 WORSE, bar missed by 5.38. Engaged-then-diverged: simcal.out norm 0.27 off zero, simprior temp collapsed 0.0393->0.0255, simprior out suppressed 0.59->0.22, q/k suppressed -18%/-33%; RMSE/MAE decoupled ep6-9 before synchronized train+val NaN at step 2249. Harness excluded."
```

**Expected confidence after this one event (Eq.1, η=0.20, c=0.50; advisory only):** `c' = c − η·w·c = 0.50 − 0.20·0.90·0.50 = **0.4100**` — still `uncertain` (above the <0.25 refuted threshold), so H0007 is not yet classified refuted; a further independent contradiction event is needed to cross it. The Lead records the event; the ledger derives the stored confidence. **This synthesis does not record it and does not edit any ledger or confidence value.**

## 8. Live-parent reference (explicit, unchanged)

**N0002_h0001 (22.5641 @ep30) remains the live-parent reference — all future falsifier bars bind to it, not to this node.** Confirm iff child final val MAE **≤22.26** (≥0.30 below 22.5641) AND gt>500 dense mean |Δ| below 511.7, under the canonical protocol at seed 20260830.

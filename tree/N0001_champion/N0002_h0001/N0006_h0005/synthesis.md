# Synthesis — N0006_h0005 (H0005 `use_verify`) vs live parent N0002_h0001

- **Node:** N0006_h0005 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Tested:** H0005 SOLO (`use_verify=true`)
- **Outcome (result.json, verified):** child best EMA val MAE **23.5802 @ep31**, 32/32 epochs, `budget_hit=true`, `elapsed_s=1815.4`. Margin **+1.0161 worse** than parent same-seed; bar **≤22.26** missed by **+1.3202**. Train final 2.9459 vs parent 2.6183 (+0.3276 worse). Val RMSE best 92.3955 vs parent 88.9234 (+3.47 worse).
- **Verdict: REFUTE H0005.** Global-bar conjunct of the pre-registered dual falsifier (idea.md:4-6) fails outright at 44× paired-contrast precision (~0.02) in the wrong direction. Dense-tail conjunct (gt>500 mean |Δ| vs 511.7) not evaluable from run scalars (no per-image dump) — left unevaluated, but the global miss alone decides. Evidence booking stays Lead-only via CLI; this file proposes bookings only.

## 1. Consolidated reads (quant + qual + causal + diagnostic)

- **Quant (REFUTE):** never crosses bar (min 23.5802 over 32 eps, 1.32 above 22.26); never matches parent (child best sits above parent ep14 level); flat tail ep28-32 range 0.0089 — ep31 best is plateau sample, extension cannot close 1.32 at 0.009/5ep crawl. Train-worse + RMSE-worse = uniform degradation, no catastrophe-image rescue in RMSE/MAE ratio (3.95→3.97 while both degrade).
- **Qual (active-but-harmful underfit, edge-best):** child tracks parent ~10 eps then separates and flatlines high; train-worse/val-worse = optimization drag, not memorization. RMSE bottoms ~2/3 then creeps up while MAE flat = mild tail-overfit garnish on general underfit — reverse of intended dense-tail sharpening. Gate USED (checkpoint prop2/ver2 norms far off init — see causal), so "gate ignored" branch is closed. Temp 0.07→~0.02 noted as lead, not finding.
- **Causal (used-and-harmful, harness excluded):** prop2 ‖w‖ 2.2408 + bias 0.0612 (off zero-init, 0/64 dead); ver2 ‖w‖ 3.8631 + bias 0.4846 (V polarized: synthetic-input mean 0.5435, std 0.1784, only 29.9% in [0.45,0.55]); temp 0.07→0.0214 (÷3.3, 2.5× peakier than working simprior temp 0.0546). Residual out-shouts confirmed simprior readout (2.24 vs 0.96). Path: (1) self-echo proposal P from `cond_map` amplifies MSE blur once off zero-init; (2) over-sharpened K=3 verification on 32-dim q/k turns cosine noise into hard V decisions; (3) magnitude dominance steers `cond_map` mostly via corrupt gate. Checksums re-verified (config 88b8f37902f414f1, model d239d6413086b852), RNG append-only (Verify constructed last, model.py:299-300), seed 20260830, full 32-ep schedule — margin +1.016 ≫ 0.02 precision, not noise.
- **Diagnostic (marginal +9.8 s overrun, trustworthy best):** TB span 1738.6 s, +1.87 s/ep vs parent — uniform drag from ep0, flat deltas, no leak/blow-up; N0005_h0004 shows identical +1.85 s/ep with a different module → common environmental cause, NOT Verify cost. `timeout` is wall-clock bookkeeping; all 32 val points complete, ep28-32 within 0.001, best@ep31 plateau-confirmed. Do NOT re-run for `done` status; do NOT book cost hypotheses.

## 2. Remap / discard (quality gate d)

- **Discarded — "Verify is over-budget":** unattributable per diagnostic §2 (N0005 parity → environment). No booking.
- **Discarded — `train/mae` NaN:** pre-existing logger bug on parent and child alike; zero signal.
- **Discarded — missing V-histogram TB scalars:** process gap (runner logs no `verify/*` diagnostics); recovered via checkpoint instead. Not a mechanism claim.
- **Remapped — temp 0.0214 oversharpen:** lead for Direction 2 framing (calibration), not a standalone finding.
- **Closed — repetition-gated P×V sharpening as implemented (pooled-e repetition + P-from-cond self-loop):** REFUTED in strongest form (learned, engaged, wrong-way). Per AGENTS §11 any revisit must carry a NEW falsifier (P from `fine`, temp floor, objectness-supervised V) — none booked here; portfolio below attacks the two upstream causes instead.

## 3. Dedup vs ledger H0001–H0005 (quality gate c)

- H0001 `use_simprior` (pooled 3-key readout) — supported, load-bearing; not rebooked.
- H0002 `use_gca_cal` — contradicts-recorded; untouched.
- H0003 `use_padapt` — contradicts-recorded; untouched.
- H0004 `use_simbank` (r=5 75-key token-max bank) — timeout-null, uninterpretable; Booking 1 below is NOT a retry: pooled per-exemplar h2 keys (3 keys, no second full-res qproj, 48×48 + upsample, ≈12.5k) vs token-max (75 keys, full-grid einsum) — distinct mechanism with a new falsifier; books as a NEW hypothesis, never as contradicts on H0004.
- H0005 `use_verify` — refuted here; no contradicts-retry booked. Both bookings below are fresh ids.

## 4. Bookings proposed (K_SYNTH=2; fallback order keyed off causal)

Causal orders the portfolio: pooled-`e` degenerate for small exemplars (h3@1/16 ROI 1–2 cells pooled to one vector — the evidence the gate consumed was coarse before temperature ever acted) → finer bank first; anisotropy/temperature calibration second. High-res decoder residual (`use_hires`, SANet/FeatUp/DAVE) deferred, not booked — gate-right/resolution-wrong branch is disfavored because the gate corrupted train as well as val (evidence problem, not grid problem).

**Booking 1 — NEW `use_h2pool` (fallback Direction 1; refs LOCA/CountingDINO/SAFECount):**

IF h2-pooled 3-key similarity bank via use_h2pool IN a solo single-switch child of live parent N0002_h0001 trained under the canonical protocol at seed 20260830 for 32ep/1800s with the module constructed AFTER all parent modules with append-only RNG and decoder in_ch untouched at 12.5k added params, THEN final EMA val MAE falls at least 0.30 below live parent 22.5641 to 22.26 or lower AND gt>500 dense-image mean |Δ| falls below the parent 511.7 baseline, BECAUSE per-exemplar h2-at-1/8 3x3 attention-pooled keys retain small-object part geometry that h3-pooled keys average into background while the proven 3-key K-normalized readout at 48x48 upsampled to 96x96 preserves optimizer-trusted comparison dynamics within the time budget, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol at seed 20260830 AND gt>500 dense-image mean |Δ| is not reduced below the parent 511.7 baseline.

**Booking 2 — NEW `use_simcal` (fallback Direction 2; refs QK-Norm/anisotropy/CACViT):**

IF similarity-distribution calibration via use_simcal IN a solo single-switch child of live parent N0002_h0001 trained under the canonical protocol at seed 20260830 for 32ep/1800s with the module constructed AFTER all parent modules with append-only RNG and decoder in_ch untouched at 3k added params, THEN final EMA val MAE falls at least 0.30 below live parent 22.5641 to 22.26 or lower AND gt>500 dense-image mean |Δ| falls below the parent 511.7 baseline, BECAUSE per-exemplar spatial de-meaning plus signal-to-noise re-weighting of the reused simprior similarity volume removes the anisotropic cosine baseline so the learned temperature acts on true margins while the magnitude-weighted consensus preserves absolute count scale that full spatial normalization destroys, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol at seed 20260830 AND gt>500 dense-image mean |Δ| is not reduced below the parent 511.7 baseline.

- Format gate self-check (both): markers IF-IN-THEN-BECAUSE-DISPROVED-IF in order; no hedging; falsifier carries numbers (22.26 global + 511.7 dense secondary); binds to live parent 22.5641; canonical seed/protocol stated. No opposite-of-existing claim → new ids correct, no contradicts misuse.

## 5. Calibration bin table (`python3 scripts/discovery.py calibration`, 2026-09-20)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)      3       1     33%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall          3       1     33%      -
reliability error (weighted |rate − pred_conf|): 0.167

=== Current Standings ===
hyp       confclass       tests
H0001    0.585uncertain       1
H0002    0.420uncertain       1
H0003    0.405uncertain       1
H0004    0.500uncertain       0
H0005    0.500uncertain       0
```

- Read: 3 tested hypotheses, 1 confirm (33% vs 0.50 mean pred_conf in the only populated bin); reliability error 0.167. H0005 verdict not yet ledger-recorded (Lead records via CLI) — table will shift on booking.

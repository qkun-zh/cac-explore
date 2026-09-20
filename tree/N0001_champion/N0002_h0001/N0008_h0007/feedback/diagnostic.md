# Diagnostic — N0008_h0007 (H0007 `use_simcal`)

- **Node:** N0008_h0007 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Tested:** H0007 SOLO (`use_simcal=true`, `use_simprior=true` inherited).
- **Outcome:** child best EMA val MAE **27.6371 @ep09**, 32/32 epochs stepped, `budget_hit=false`, `code=ok`, `elapsed_s=1797.8`. Margin **+5.0729 worse** than parent same-seed; bar **≤22.26** missed by **+5.3771**. Epochs 10–32 all-NaN on every loss/metric series (23 dead epochs).
- **Verdict: REFUTE H0007 — optimization-instability failure (early-best-then-diverge), not overfit, not stable underfit.** The +5.07 miss is already legible while the run is still numerically healthy (ep6–9); the ep10 NaN wall destroys the tail but does not cause the verdict. Harness/budget excluded (§7).

## 1. Failure-mode classification (the asked question)

| Candidate | Test | Outcome |
|---|---|---|
| **Overfit** (train improves, val degrades — memorization U-curve) | Train-worse-or-better vs parent + finite val tail with U-turn | **Rejected.** No U-turn exists: val tail is NaN, and the train series collapses *with* val (synchronized NaN at step 2249/ep10). A generalization gap cannot Synchronize train+val to NaN. At the last healthy epoch (ep09) child train 5.8483 is marginally *better* than parent-at-same-epoch 6.1468 (−0.30) while val is +1.35 worse — a faint overfit-leaning snapshot, but it never develops into an overfit curve because optimization dies first. |
| **Stable underfit** (N0006 pattern: converged 32ep, train-worse + val-worse, plateau) | Full finite curve, train final vs parent 2.6183 | **Rejected.** Child never reaches the parent's late-train regime (diverges at ep10 instead of converging to ~2.6). No plateau, no convergence — 9 healthy epochs then a wall. |
| **Early-best-then-diverge / runaway dynamics** | Widening pre-collapse gap + tail-distress prelude + synchronized train+val NaN + engaged-but-sharpening checkpoint | **Accepted.** Gap widens monotonically while healthy (+0.5→+1.35, §2); RMSE/MAE decouple ep6–9 (§3); everything goes NaN together at ep10 (§2); ep9 checkpoint shows the new path engaged with collapsed temperatures (§4). |

## 2. Timeline: zero-init held, then lost while healthy, then blew up

- **Ep1–5: tracks parent** (child vs parent val MAE: 139.9/153.3, 67.0/70.1, 48.2/48.8, 39.3/39.1, 33.9/33.5 — quant §2 / qual §1). The zero-init contract worked: step-0 forward started at the parent and the first ~5 epochs learned the same basin. This rules out an N0007-style day-zero interference.
- **Ep6–9: separates while healthy.** Gaps +0.5, +0.75, +0.95, +1.35 — widening every epoch, with per-epoch gains decelerating (−2.0, −1.7, −0.9, −0.6). The child was already losing before anything went non-numeric. Best @ep09 (28% of schedule) vs parent best @ep30 is therefore a **last-healthy-epoch best, not an optimum**.
- **Ep10–32: synchronized NaN wall.** `train/loss` first NaN at step 2249 (inside ep10); every epoch-aggregated series (`val/mae`, `val/rmse`, `train/loss_epoch`) NaN for all 23 remaining epochs, yet the harness stepped all 32 epochs and reported `done`/`code=ok`. Train-side collapse rules out a val-metric glitch: the forward/loss itself blew up. The 23 tail epochs are uninterpretable, not signal.

## 3. Pre-collapse tell: RMSE/MAE decoupling (tail-first distress)

Child `val/rmse`: 164.95, 117.33, 106.68, 100.81, 97.12, **95.48 (ep6 bottom)**, 95.84, 96.63, **97.47 @ep09** — rising **+1.98** over ep6→9 while `val/mae` still falls 30.83→27.64. Mass was already moving the wrong way on the tail (large-count images dominate RMSE) while the median improved: the selection dynamics were straining before they broke. Same "tail-first distress" garnish as N0006's mild version, here as a prelude to blow-up rather than a plateau garnish. Left as shape; mechanism attribution in §5.

## 4. Checkpoint forensics (server single-shot reads, `best.pth` = ep9 healthy)

| Param | Child N0008 @ep09 | Parent N0002 @ep30 | Read |
|---|---|---|---|
| `head.simcal.out.weight` norm | **0.27428** (off zero-init) | — (no module) | New path **engaged**, not ignored — the clean-null branch is closed |
| `head.simcal.tau` | 0.8888 (init 1.0) | — | Near init, slightly down — **not** a tau runaway; the blow-up did not come through the SNR-sharpness scalar exploding |
| `head.simprior.temp` | **0.0255** (init 0.07) | **0.0393** | **÷1.54 sharper than the working point** — same collapse family as N0006's 0.07→0.0214 (÷3.3). Joint dynamics drove the trusted readout peakier, not the new scalar |
| `head.simprior.out.weight` norm | 0.2187 | 0.5874 | Trusted path **suppressed to 0.37×** parent while the new path engaged at 0.27 — total additive magnitude comparable (~0.49 vs 0.59) but split across two competing similarity readouts with different evidence |
| `simprior.qproj/kproj` norms | 4.70 / 4.94 | 5.73 / 7.35 | Under-developed at ep9 (expected — 9 vs 30 epochs); not interpreted |
| NaN in `best.pth` | none (all `hasnan=False`) | — | Collapse postdates ep9; no earlier checkpoint or `last.pth` exists, so post-collapse weights are uninspectable (§8) |
| Bias norms | both ≈0.052 (identical to print precision — coincidence, not interpreted) | 0.073 | Small either way; not load-bearing |

## 5. Mechanism attribution (suspect chain, not proven fact)

The failed quantity is the calibrated evidence pair, and the prime suspect is the **de-meaning division amplifying flat-exemplar noise inside a sharpening loop**:

1. `Z = (S−μ)/(σ+ε)` with `ε=1e-5`: for a flat (background-like) exemplar `σ→0` turns tiny spatial noise into large `Z` values — no floor on `σ`, no clamp on `Z`. `c_cal` then carries large-magnitude margin claims from the least informative exemplars.
2. `SNR = μ/(σ+ε)` saturates `w` toward one-hot on the same flat exemplars (large μ, tiny σ → SNR ~1e4); `τ≈0.89` stays active enough to keep the selection hard. The "peaky-over-loud" rule inverts: the loudest background wins hardest.
3. Simultaneously the inherited simprior temp collapses to 0.0255 (1.54× peakier than the working 0.0393), so **both** readouts go hard at once on noisy K=3 cosine — per-cell density errors spike on tail images (the ep6–9 RMSE rise), MSE gradients explode, and at step 2249 the forward/loss goes non-numeric everywhere at once.
4. `grad_clip=1.0` cannot save a forward that already produces Inf/NaN — clipping acts on finite grads, not on non-numeric activations.

Status honesty: (1) is structural (read from `model.py:259-269`); (2)–(4) are consistent with every number above but the NaN-step activations were never dumped, so this remains the best-supported suspect, not a proven root cause. What IS proven: the direction was harmful before the blow-up (widening healthy gap), the blow-up is forward/loss numerics (synchronized train+val), and the new path was engaged (out norm 0.27) with collapsed joint temperatures.

## 6. Contrast: third similarity-family miss (after N0005-null, N0007-catastrophe, N0006-refutation)

- **vs N0007_h0006 (H0006 bank, 47.8439 @ep32, +25.28):** opposite onset. N0007 separates at ep1 and stalls (shared-`qproj` joint interference, train 13.93 vs 2.62 — interference from step 0). N0008 tracks for 5 epochs (zero-init held; no shared-projection pollution — SimCal reuses S read-only with no new q/k) and only then separates gradually. N0007 is day-zero corruption; N0008 is progressive degradation → instability. Both close their respective branches: bank-resolution AND distribution-calibration.
- **vs N0006_h0005 (H0005 verify, 23.5802 @ep31, +1.016, converged):** opposite ending. N0006 is the stable used-and-harmful reference: full finite curve, plateau-trustworthy best, train-worse + RMSE-worse uniform degradation. N0008 never converges. N0006's temp-collapse lead (0.0214) reappears here in the simprior temp (0.0255) — the sharpening-runaway signature now spans two nodes.
- **Family read:** the confirmed simprior readout (3-key, temp ~0.04, out norm ~0.59) is at a local optimum — all three tested similarity-side perturbations degrade or destabilize: H0004 simbank (timeout-null, uninterpretable), H0005 verify (refuted +1.02), H0006 bank (catastrophic +25.28), H0007 simcal (refuted +5.07 + NaN). Per AGENTS §11 none are retried silently; any revisit needs a new falsifier (e.g. σ-floored / Z-clamped calibration is a *different* mechanism, not a retry — and needs its own booking, not taken here).

## 7. Exclusions (harness, budget, protocol)

- **Budget fine:** `elapsed_s=1797.8` < τ_max 1800 (−2.2 s), `budget_hit=false`, 32/32 epochs stepped. The NaN tail is not a `_BudgetStop` truncation (quant §6).
- **Harness excluded:** identical harness runs the parent finite for all 32 epochs; smoke passed (31.3M total); seed 20260830 + `augment=false` verified in config; paired-contrast precision ~0.02 ≪ 5.07 margin (~169× in the wrong direction before any divergence).
- **No fluke:** closest approach over all 32 epochs still +5.38 above the bar; child best sits below the parent's ep08 level (parent ep08 27.27, ep09 26.28).

## 8. Process gaps (runner/logging, found via single greps)

1. **Runner reports `done`/`code=ok` over a 23-NaN tail.** A run whose last 70% is non-numeric should not look identical in status to a converged run — at minimum a `nan_epochs` count in `result.json`.
2. **Text log is silent about the collapse:** case-insensitive `nan` grep over `/tmp/train-b3-n0008.log` returns only **2 lines** (the final `val/mae: nan` print); no loss-spike line, no exception. TB-only visibility (repeat of the sibling gap, now load-bearing).
3. **No gate scalars logged** (`tau`, `temp`, `out` norms) — the sharpening runaway (§4–5) is recoverable only from checkpoints, not from TB.
4. **No `last.pth` saved** (server `run/latest/` holds only `best.pth` + `tb`) — post-collapse weights uninspectable; the NaN-step parameter state cannot be confirmed.
5. **`train/mae` all-NaN** in parent and child alike — pre-existing logging bug, still depriving qual reads of a train-side counting signal.

## 9. Redirect + evidence proposal (Lead records via CLI — not booked here)

- **Proposed:** `contradicts` on **H0007** with high strength (margin +5.07 ≈ 169× precision, engaged-then-diverged, harness excluded). N0004-padapt (+3.00) drew w=0.95 and N0007-H0006 (+25.28) drew w=0.95; N0008's margin exceeds both in absolute terms but the 23-NaN tail slightly discounts tail-shape claims — **suggest w=0.90**, Lead decides. Note text: "N0008_h0007 done 27.6371 @ep09 (32/32ep, budget fine, 23 NaN epochs ep10–32) vs live parent 22.5641: +5.07 WORSE, bar missed by 5.38. Engaged-then-diverged: simcal.out norm 0.27 off zero, simprior temp collapsed 0.0393→0.0255, simprior out suppressed 0.59→0.22; RMSE/MAE decoupled ep6–9 before synchronized train+val NaN at step 2249. Harness excluded."
- **Redirect (portfolio consequence):** similarity-side calibration of the K=3 volume is now refuted in this form; the sibling bank branch is already catastrophic. Upstream-remaining directions are decoder-side / density-field resolution and the per-image catastrophic-tail dump (STATE per-image diagnostic: heavy-tailed, dense under-counted, count-calibration no headroom). No new hypothesis booked in this file (K_SYNTH discipline).

## Data sources (every number traced)

1. `tree/N0001_champion/N0002_h0001/N0008_h0007/result.json` — best 27.63706779 @ep9, 32/32, elapsed_s 1797.8, budget_hit false, shas.
2. `tree/N0001_champion/N0002_h0001/result.json` — parent 22.56414604 @ep30.
3. `tree/N0001_champion/N0002_h0001/N0008_h0007/info.json` — status done, tested ["H0007"].
4. `config.toml:9,27-28,33` — seed 20260830, use_simprior/simcal true, augment false.
5. `feedback/quant.md` §§1–2,4 — full TB series (9 finite + 23 NaN), train ep09 5.8483 vs parent ep09 6.1468 / final 2.6183, RMSE ep6→9 +1.98, step-2249 NaN onset.
6. `feedback/qual.md` §§1–2 — ep1–5 tracking values, ep6–9 gaps, silent-log read.
7. Server single-shot checkpoint reads (this session): child `best.pth` (epoch 9) simcal.out.w 0.27428 / b 0.0518, tau 0.888826, simprior.temp 0.0255037, simprior.out.w 0.2187, qproj 4.70359, kproj 4.94052, all hasnan=False; parent `best.pth` (epoch 30) simprior.out.w 0.587438, temp 0.0393157, qproj 5.73144, kproj 7.35467.
8. Server single grep: `grep -c -i nan /tmp/train-b3-n0008.log` → 2.
9. `model.py:259-269` (SimCal forward), `:180-187` (attachment), `:289-294` (append-only construction).
10. `idea.md:4` + §5 — pre-registered dual falsifier (22.26 + dense 511.7).
11. `memory/hypotheses.jsonl:9-12` — N0006 evidence (temp 0.0214, prop2/ver2 norms), H0006/H0007 bookings, N0007-H0006 contradicts w=0.95 (+25.28).
12. `tree/.../N0006_h0005/synthesis.md` §§1–2,4 — N0006 converged-refutation reference + portfolio order.
13. `tree/.../N0007_h0006/result.json` + `N0006_h0005/result.json` — 47.8439 @ep32 / 23.5802 @ep31 contrasts.

# Synthesis — N0007_h0006 (H0006 `use_h2pool`) vs live parent N0002_h0001

- **Node:** N0007_h0006 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Tested:** H0006 SOLO (`use_h2pool=true`)
- **Outcome (result.json, verified):** child best EMA val MAE **47.8439 @ep32**, 32/32 epochs, `budget_hit=false`, `elapsed_s=1809.5`. Margin **+25.2798 worse** than parent same-seed; bar **≤22.26** missed by **+25.5839**. Train final 13.9336 vs parent 2.6183 (5.3×). Val RMSE final 128.2024 vs parent 89.2400.
- **Verdict: REFUTE H0006.** Global-bar conjunct of the pre-registered dual falsifier (idea.md §5) fails outright at ~845× paired-contrast precision (~0.03) in the wrong direction. Dense-tail conjunct (gt>500 mean |Δ| vs 511.7) not evaluable from run scalars (no per-image dump) — left unevaluated, but the global miss alone decides. Evidence event stays Lead-only via CLI; this file proposes bookings only.

## 1. Consolidated reads (quant + qual + causal + diagnostic)

- **Quant (REFUTE):** never crosses bar (min 47.8439 over 32 eps, 25.58 above 22.26); 2.12× parent level — catastrophic blow-up, same family as N0005_h0004 timeout point (48.12). Curve: violent early collapse (219.64 → 48.27 by ep11), +0.50 mid hump peaking ep16, 0.42 grind into flat tail ~47.9 — converges to double the parent. Best=edge (ep32=final) with last-3-epoch crawl 0.0188: truncated-edge TRUE, verdict unaffected (25.58 gap not closable at that rate). Train-worse + val-worse + RMSE-worse jointly (train +11.32, RMSE best +34.90 / final +38.96, RMSE/MAE 3.95→2.68 while both degrade) = optimization failure, not tail-concentrated miss. RMSE best @ep04 vs MAE best @ep32 (28-epoch disagreement) = corrupted evidence path. Wall +54.5 s (+3.1%) with `done`/32-epochs = marginal/environmental overrun class (N0006 +9.8 s parity), not a stop event.
- **Qual (engaged-but-harmful stall, N0006-flavored ×25):** separation immediate — ep1 child 219.64 vs parent 153.29 (gap +66), ordering never flips at any of 32 points. Stall, not fall: child drifts 48.28→47.84 over last ~20 eps while parent descends 26.28→22.56. Train parks 13.9–14.3 from ep2 (parent 8.65→2.62) = underfit at a bad level, no overfit signature. Ep1 inverse (child train-better/val-worse for exactly one epoch) then train-worse forever = new branch pulls gradients elsewhere from the first updates, then holds optimization at a worse stationary point. Branch engaged (train 5× parent rules out clean null — zero-init retained would reproduce ~2.6); TB lacks `h2pool.out`/temp tags so shared-qproj vs own-out attribution is causal-scope. Clean run: 32/32, no NaN (except pre-existing `train/mae` all-NaN in both child and parent), no traceback/OOM, smoke 31.4M ≤32M.
- **Causal (shared-query corruption of the confirmed simprior readout — primary):** checkpoint probe: `h2pool.out` w 0.1966 / b 0.3128 (engaged off zero-init), `pool_attn` entropy **2.1971 ≈ log(9)** with mean max-attn 0.114 ≈ 1/9 (selector learned nothing — plain 9-token mean), `Sv` range [-0.36,-0.08] std 0.051 with across-K std 0.039 (all-negative mush, zero K-discriminability, no margin for any temperature), `W` near-uniform, live `d48` RMS 0.045 (impotent residual), dense-image count probe 105.7 vs 429.0 (severe under-count). Meanwhile confirmed readout starved: `simprior.qproj` 3.52 vs parent 5.73, `kproj` 3.54 vs 7.35 (~half norms, frozen), `simprior.temp` 0.0547 vs 0.0393 (never sharpened), `simprior.out` 0.36 vs 0.59 (immature). Ordering: h2 stage-2 geometry mismatched to h3-tuned query space (scratch 192→64 `kproj` + shared `qproj`) → flat all-negative `Sv` → pool gets no discriminative gradient, stays uniform → garbage-gradient stream into shared `qproj` grows with `out.weight` → load-bearing readout never re-matures → train AND val stuck (13.9/47.8) → decoder compensatory doubling. Zero-init gave step-0 equality but no protection once training started — the "reuse qproj" cost-saving was the load-bearing mistake (H0004 paid forward-cost for a second qproj; H0006 paid gradient-cost for sharing it).
- **Diagnostic (joint-interference optimization divergence — same attractor as N0005):** per-epoch cost +1.71 s/ep, flat from ep1, matches N0005 (+1.85) and N0006 (+1.87) to ±0.2 s/ep across three different code deltas = environmental offset, H2Pool cost claim validated, no compute blow-up, `budget_hit=false`. Train trace N0007 (19.62→flat ~14) overlays N0005 (20.53→flat ~14) despite fully different banks (3-key h2-pooled 48×48 vs 75-key token-max full-grid); val 47.84 vs 48.12, RMSE final 128.20 vs 128.56 — identical attractor. Shared element = second similarity residual into `cond_map`, not h2 specifics or 48×48 upsampling. Ruled out: loss explosion (all finite, no spikes, norms smaller than parent, `grad_clip=1.0`), clean null (branch engaged, parent branch dragged down with it), harness (seed 20260830 + augment=false confirmed, checksums byte-identical server/local, solo-switch, 32/32 complete). Contrast N0006: N0006 fits train (2.95≈2.62) and fails val +1.02 = valid used-and-harmful refutation; N0007/N0005 fail train 5× = divergence. Do not treat as the same timeout family; do not re-run (flat-tail-robust, 25 MAE from bar — re-run is outcome shopping per §5.1).

## 2. Remap / discard (quality gate d)

- **Remapped — decoder growth (`decoder.head` 2.59 vs 1.29, `cond.out` 5.67 vs 3.27) as compensatory, not causal:** optimizer pushing free gain knobs on corrupted `cond_map`; train-stuck proves features broken first. No booking.
- **Discarded — "H2Pool is expensive":** timing parity with N0005/N0006 implicates environment; booked <0.5 s/ep consistent with flat +1.7 s/ep offset. No booking.
- **Discarded — `train/mae` all-NaN:** pre-existing logger artifact, parent-identical. Zero signal.
- **Discarded — temp-collapse story:** temps 0.052/0.055 mild drift from 0.07, nothing like N0006 ÷3.3 collapse. Ruled out by causal §4.
- **Discarded — exploding-residual story:** `h2pool.out` 0.20/0.31, `d48` RMS 0.045 = engaged-but-impotent, not explosive. Ruled out by causal §4.
- **Remapped — ep1 train-better/val-worse single point as early gradient-capture lead:** supports epoch-1 capture framing for future banks (any future bank must explain why it avoids epoch-1 capture), not a standalone finding.
- **Closed — pooled keys at any single octave + shared live-readout query projection:** REFUTED in this implementation. Per §11 any revisit needs a NEW falsifier plus interference analysis (own query path or frozen-query discipline) — none booked here.

## 3. Dedup vs ledger H0001–H0007 (quality gate c)

- H0001 `use_simprior` — supported, load-bearing; corrupted by this node, not rebooked.
- H0002 `use_gca_cal` — contradicts-recorded; untouched.
- H0003 `use_padapt` — contradicts-recorded; untouched.
- H0004 `use_simbank` (75-key token-max) — timeout-null; this node is NOT a retry (3-key h2-pooled, shared-qproj, 48×48) and its verdict stands on its own; family-level consequence in §4 instead of contradicts-evidence on either instance.
- H0005 `use_verify` — contradicts-recorded (N0006); untouched.
- H0006 `use_h2pool` — refuted here; no contradicts-retry booked.
- H0007 `use_simcal` — in flight (N0008 proposed, undecided); §4 assesses honestly whether it shares the §5 risk. Booking below is NOT opposite-of-existing, so a new id is correct and no contradicts-booking applies.

## 4. Portfolio consequences

**(1) New STATE gotchas rule proposal (for Lead to append):** NEVER share projections or parameters with the confirmed simprior readout. Append-only construction order plus read-only input tensors are not sufficient protection when gradients flow back through a shared projection — `H2Pool.forward` read frozen `h2` read-only and reused `simprior.qproj` without copying, yet the h2-key garbage-gradient stream froze `qproj`/`kproj` at half parent norms and stalled `temp`. Proposed gotcha line: "Shared-projection ban: no new module calls `simprior.qproj`/`kproj` (or any confirmed readout parameter) in its forward graph; reuse of similarity evidence is recompute-detached or statistics-only with gradients blocked into simprior parameters; any exception needs explicit Lead approval plus q/k-norm + temp-trajectory diagnostics at verdict."

**(2) Second-similarity-residual-into-`cond_map` family now 2× catastrophic (N0005, N0007) — deprioritize residual-into-cond variants, prefer calibration-only.** Two different banks converged to train≈14 / val≈48: the bank-form family (second K-normalized `[top1, consensus]` residual into cond), not h2 resolution, is the suspect. Honest H0007 risk assessment: `use_simcal` recomputes `S` through simprior weights but adds no new keys and no new qproj — structurally it still opens a second gradient path into the shared `qproj`/`kproj` (autograd from `simcal.out` through recomputed `S`), so the shared-projection pattern is present. The empirically fatal element is absent: no from-scratch key projection, no collapsed all-negative `Sv` dragging the query space — its gradients derive from reductions over the already-trusted h3 `S` volume at ~0.2k params with a clean-null design (uniform `w`, `tau` near init, `out` near zero reads as rejection). Risk is therefore lower but nonzero. N0008 verdict must inspect `simprior.qproj`/`kproj` norms + both temps: if N0008 also suppresses q/k or stalls temp, the family expands from "second similarity bank" to "any second gradient path into simprior projections" and even calibration-only becomes suspect.

**(3) Booking discipline:** loss invariant holds — stay architectural. Objectness-gated readout not booked now (N0004 precedent: failed the new-falsifier gate per §11; a fresh attempt needs a NEW falsifier plus pre-condenser-gating disambiguation, deferred). Simcal-followup conditional on N0008 outcome not booked now (N0008 undecided — booking it here is pre-commit). One decidable booking below (decoder-side, touches neither similarity nor shared projections); remainder left to N0008 synthesis. Total 1 ≤ K_SYNTH=2.

## 5. Bookings proposed (1 of ≤2; format gate verified)

**Booking 1 — NEW `use_hires` (decoder-side detail residual; no similarity, no shared projection):**

IF high-resolution decoder detail residual via use_hires IN a solo single-switch child of live parent N0002_h0001 trained under the canonical protocol at seed 20260830 for 32ep/1800s with the module constructed AFTER all parent modules with append-only RNG and decoder in_ch untouched at 15k added params, THEN final EMA val MAE falls at least 0.30 below live parent 22.5641 to 22.26 or lower AND gt>500 dense-image mean |Δ| falls below the parent 511.7 baseline, BECAUSE a zero-init post-decoder detail pass over joint density logits plus frozen h2-derived fine context restores small-peak sharpness that the single 96x96 cond path smooths away without touching the confirmed similarity readout and without sharing any projection so simprior gradients stay isolated, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol at seed 20260830 AND gt>500 dense-image mean |Δ| is not reduced below the parent 511.7 baseline.

- **Why this and not a bank:** attacks the dense-tail symptom (smeared peaks) at the decoder interface where N0007/N0005 never operated, under the shared-projection ban (§4.1) by construction — own parameters only, frozen-h2 read-only context, zero-init additive residual, decoder `in_ch` untouched. Clean null reads as mechanism rejection (residual norm near zero), engaged-but-worse reads as decoder-capacity rejection; neither confounds the simprior readout.
- **Format gate self-check:** markers IF-IN-THEN-BECAUSE-DISPROVED-IF in order; no hedging; falsifier carries numbers (22.26 global + 511.7 dense secondary) bound to live parent 22.5641; canonical seed/protocol stated. Verified: `validate()` → errors [], warnings []; `novelty_check.py` → top sim 0.608 (H0007) < 0.82 NOVEL, no structural twin (decoder detail residual vs all similarity/verification ledger entries H0001–H0007). Not opposite-of-existing → new id correct.
- **Deliberately unbooked now:** simcal-followup (conditional on N0008 verdict), objectness-gated readout (needs NEW falsifier per §11), any residual-into-cond similarity variant (family deprioritized per §4.2).

## 6. Calibration bin table (`python3 scripts/discovery.py calibration`, 2026-09-20)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)      4       1     25%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall          4       1     25%      -
reliability error (weighted |rate − pred_conf|): 0.250
WARNING: reliability error > 0.20 — confidence at test is drifting

=== Current Standings ===
hyp       confclass       tests
H0001    0.585uncertain       1
H0002    0.420uncertain       1
H0003    0.405uncertain       1
H0004    0.500uncertain       0
H0005    0.415uncertain       1
H0006    0.500uncertain       0
H0007    0.500uncertain       0
```

- Read: 4 tested hypotheses, 1 confirm (25% vs 0.50 mean pred_conf in the only populated bin); reliability error 0.250 (WARNING — confidence at test drifting, small-n bin). H0006 verdict not yet ledger-recorded (Lead records via CLI) — table shifts on booking. H0007 (N0008) untested.

## 7. Recommended evidence event (for the Lead to record via CLI — NOT recorded by this subagent)

Strength justification: w=0.95 — maximal because the quantitative miss is decisive (+25.28 vs parent = 84× bar magnitude wrong-way, bar missed by 25.58, clean 32/32 `done` pass with zero harness confound) AND the mechanism is directly supported by trained-weight evidence (pool-attn uniform at 2.1971, all-negative zero-discrimination `Sv`, shared q/k frozen at half parent norms, temp unsharpened) with the N0005 same-attractor replication; docked nothing further — single-run scope already priced into the catastrophic margin.

```
python3 scripts/discovery.py evidence H0006 --type contradicts --strength 0.95 --node N0007_h0006 --note "N0007_h0006 child 47.8439 @ep32 vs live parent N0002_h0001 22.5641: +25.2798 WORSE, bar <=22.26 missed by +25.5839; 32/32 clean done pass, budget_hit false. Mechanism engaged-but-corrupt: pool_attn entropy 2.1971 uniform, Sv all-negative zero-discrimination, shared simprior qproj/kproj frozen at half parent norms (3.52/3.54 vs 5.73/7.35), temp unsharpened (0.0547 vs 0.0393); same train~14/val~48 attractor as N0005 — second-similarity-residual family suspect."
```

## 8. Live-parent reference (explicit, unchanged)

**N0002_h0001 (22.5641 @ep30) remains the live-parent reference — all future falsifier bars bind to it, not to this node.** Confirm iff child final val MAE **≤22.26** (≥0.30 below 22.5641) AND gt>500 dense mean |Δ| below 511.7, under the canonical protocol at seed 20260830.

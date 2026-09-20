# Diagnostic — N0007_h0006 (H0006 `use_h2pool`) · functional-failure triage

- **Node:** N0007_h0006 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Status:** `done`
  (`budget_hit: false`, 32/32 epochs, best **47.8439 @ep32**), `elapsed` 1803.67 s /
  `elapsed_s` 1809.5 s vs τ_max 1800 s — wall clock slightly over ceiling but runner
  marked `done`, not `timeout`. Result is complete (32 finite val points) and usable
  as a verdict: it misses the 22.26 bar by **+25.58** and regresses **+25.28** vs parent.
- **Question asked:** optimization divergence (loss explosion? gradient blow through new
  H2Pool path at 48×48?) vs representation collapse vs harness artifact. Contrast with
  N0006 (marginal timeout, converged 23.58 — valid refutation) and N0005_h0004
  (timeout 48.12, same second-bank family).
- **Verdict: OPTIMIZATION DIVERGENCE via joint-optimization interference / cond_map
  conditioning corruption — NOT loss explosion, NOT collapse-to-null, NOT harness
  artifact, NOT compute blow-up.** Details below.
- **Sources read (all verified, no recollection):** local `info.json` / `result.json` /
  `idea.md` / `config.toml` / `model.py` for N0007; parent N0002 + siblings N0005/N0006
  `info.json` + `result.json`; server TB `val/mae`, `val/rmse`, `train/loss_epoch`,
  `train/mae`, `lr-AdamW` scalars + wall times via single-shot ssh (no sleep-loops,
  no tmux); server `best.pth` param norms for N0002/N0007/N0005 via single-shot ssh.

## 1. Timing anatomy — budget fine, no compute blow-up (H2Pool cost claim validated)

| Run | TB span (ep1→ep32 val) | /31 | result.json elapsed | Status | Best |
|---|---|---|---|---|---|
| N0002 parent | 1680.7 s | **54.21 s/ep** | 1749.53 s | done | 22.5641 @ep30 |
| N0005_h0004 | 1737.7 s | **56.06 s/ep** (+1.85) | 1809.66 s | timeout | 48.1201 @ep14 |
| N0006_h0005 | 1738.6 s | **56.08 s/ep** (+1.87) | 1809.83 s | timeout | 23.5802 @ep31 |
| N0007_h0006 | 1733.4 s | **55.92 s/ep** (+1.71) | 1803.67 s | done | 47.8439 @ep32 |

- N0007 per-epoch wall deltas are flat ~55–56 s from epoch 1 onward — uniform drag,
  no progressive slowdown, no tail spike. The +1.7 s/ep matches N0005 (+1.85) and N0006
  (+1.87) to ±0.2 s/ep despite three different code deltas → **common environmental
  offset, not H2Pool cost** (same conclusion as the N0006 diagnostic §2). The booked
  <0.5 s/ep claim is consistent with this: no `_BudgetStop` pathology, `budget_hit:
  false`, 32/32 epochs, best.pth exists.
- Contrast with true compute pathology: there is none here. N0005's timeout came with
  the same flat +1.85 s/ep (also environmental); its 48.12 was uninterpretable only in
  the sense of being garbage-converged, not truncated — same as N0007's 47.84 `done`.

## 2. When MAE left the parent range — immediately (ep1), locked by ep5–6

Server TB `val/mae` (32 pts):

```
N0007: 219.64 | 104.46 | 76.00 | 58.22 | 53.33 | 54.85 | 50.11 | 48.54 | 48.28 | 48.28
       48.27 | 48.32 | 48.44 | 48.54 | 48.73 | 48.77 | 48.75 | 48.69 | 48.58 | 48.41
       48.29 | 48.23 | 48.11 | 48.06 | 48.01 | 47.96 | 47.91 | 47.89 | 47.88 | 47.86
       47.85 | 47.84(best@ep32)
Parent:153.29 | 70.05 | 48.79 | 39.12 | 33.50 | 30.30 | 28.41 | 27.27 | 26.28 | 25.41 ...
       → 22.56 @ep30
N0005: 212.08 | 140.07 | 146.70 |108.11 | 69.32 | 55.53 | 53.44 | 50.84 | 49.05 | 48.53 ...
       → flat 48.12–48.18 from ep12 on
```

- N0007 never enters the parent range: ep1 already **+66.3 above parent ep1**
  (219.64 vs 153.29), ep2 +34.4, ep3 +27.2, ep4 +19.1, ep5 +19.8; ep6 worsens
  (53.33→54.85) while parent descends 33.5→30.3. From ep8 onward val is frozen in a
  47.8–48.8 band (range 0.97 over 25 epochs) while the parent grinds 27.3→22.6.
- Best@ep32 is the min of a flat tail (ep28–32: 47.894→47.844, slope −0.01/ep), not a
  lucky dip — but the tail sits **+25.3 above the parent**, so plateau-trustworthiness
  (N0006-style) does not save it. No epoch is within 25 of the 22.26 bar.
- The zero-init step-0 identity claim is not refuted by the ep1 gap: 228 train steps
  separate step-0 from the ep1 val point, plenty for divergent gradients to separate
  trajectories. The gap proves fast divergence, not a forward-pass bug.

## 3. Train-loss trajectory — optimizer never fits (the load-bearing divergence evidence)

Server TB `train/loss_epoch` (32 pts, all finite):

```
N0007: 19.62 | 14.29 | 14.28 | 14.26 | 15.00 | 14.44 | 14.31 | 14.00 | 14.03 | 13.93
       14.01 | 13.94 | 13.96 | 14.14 | 14.15 | 14.07 | 14.02 | 13.93 | 13.93 | 13.95
       13.96 | 13.92 | 13.97 | 13.95 | 13.93 | 13.94 | 13.99 | 13.96 | 13.96 | 13.94
       13.95 | 13.93(final)
Parent:38.02 | 8.65 | 8.41 | 7.79 | 6.99 | 6.88 | 6.85 | 6.55 | 6.15 | 5.49 ... → 2.6183
N0005: 20.53 | 14.61 | 14.47 | 14.52 | 14.33 | 14.54 | 14.66 | 14.21 | 14.26 | 14.18 ...
       → flat ~14.0–14.6, final 13.81
N0006: → final 2.9459 (only +0.33 above parent — fits train, fails val = used-and-harmful gate)
```

- N0007 train stalls from ep2: 13.9–15.0 band for all 32 epochs, final **13.9336 vs
  parent 2.6183 (+11.32, 5.3×)**. The model cannot fit even the training set — this is
  optimization failure, not a generalization gap. N0006 is the contrast: it fits train
  (2.95) and fails val modestly (+1.02) = engaged-but-harmful gate. N0007/N0005 fail
  train catastrophically = divergence.
- N0005 is the same attractor: train ~14, val ~48.1, despite a completely different bank
  (75-key token-max full-grid vs 3-key h2-pooled 48×48). Two bank designs converging to
  train≈14 / val≈48.1–47.8 / RMSE≈128.5 implicates the **shared element — a second
  similarity residual into `cond_map`** — not h2 specifics, not 48×48 upsampling.
- `val/rmse`: N0007 min 123.82 @ep4, final 128.20; N0005 min 123.63, final 128.56;
  parent best 88.92 / final 89.24. RMSE/MAE 2.68 (vs parent 3.95): both metrics degrade
  jointly — no hidden tail rescue, no catastrophic-image signature beyond the global blow-up.

## 4. Ruling out the alternatives

- **NOT loss explosion / gradient blow-up through the H2Pool path:** 32/32 finite points
  for `val/mae`, `val/rmse`, `train/loss_epoch`; no spikes (largest post-ep8 val move
  <0.2; train band ±0.5); LR cosine normal (1e-3→0); `grad_clip=1.0` in force. Checkpoint
  norms are SMALLER than parent, not blown up: shared `simprior.qproj` 3.52 (parent 5.73),
  `simprior.kproj` 3.54 (parent 7.35), `simprior.out` 0.36 (parent 0.59), temps 0.052–0.055
  (parent 0.039, i.e. never sharpened). Explosion would show NaN/Inf, spikes, or inflated
  norms — none present.
- **NOT representation collapse to null / dead branch:** `h2pool.out` weight norm 0.197 +
  bias norm 0.313 (moved well off the zero-init), `h2pool.temp` 0.0524 (moved off 0.07),
  `pool_attn` norm 0.44, `kproj` 3.55 — the branch ENGAGED and learned. A clean null would
  keep `out≈0` and reproduce the parent; instead the engaged branch dragged the parent
  branch down with it (`simprior.out` 0.36 vs 0.59, temp stuck high 0.0547 vs 0.0393).
  N0005 shows the identical pattern (simbank `out` 0.46 engaged, simprior `out` 0.26
  suppressed, temps ~0.056–0.065 unsharpened). This is interference, not death.
- **NOT harness artifact:** seed 20260830 + `augment=false` confirmed in hparams.yaml;
  `use_h2pool=true` + `use_simprior=true` inherited; solo-switch (only H0006 tested);
  32/32 epochs completed; `train/mae` all-NaN is the pre-existing logger artifact (parent
  N0002 identical per N0006 diagnostic §3 — zero signal); no OOM/traceback; checkpoint
  `best.pth` epoch 32 / 47.84 consistent with TB min exactly.

## 5. Failure-mode verdict + what NOT to do

- **Verdict: optimization divergence (joint-interference).** The additive second-bank
  residual corrupts `cond_map` conditioning from the first epochs; the shared
  `simprior.qproj` (receiving gradients from both the 96×96 simprior path and the 48×48
  h2pool path) is dragged to a low-norm under-grown state; both temps stall unsharpened;
  train freezes at ~14 and val at ~48. Same attractor as N0005 despite different bank
  geometry → the bank *form family* (second K-normalized `[top1, consensus]` residual
  into cond) is the suspect, not h2 resolution. **REFUTE H0006** on the global bar
  (47.84 vs required ≤22.26; +25.28 vs parent — 800×+ the 0.03-scale paired precision in
  the wrong direction). The gt>500 dense-tail conjunct is moot: with train +11.3 and
  global +25.3, no tail slice can rescue the mechanism read.
- **Contrast with N0006 (reference):** N0006 = marginal wall-clock overrun (+9.8 s,
  `timeout` technicality), converged smooth curve, train 2.95≈parent, best@ep31 interior
  plateau, valid used-and-harmful refutation. N0007 = `done`, complete curve, but
  functionally blown up (train 5×, val +25) — a mechanism failure, not a timing
  technicality. Do not treat them as the same "timeout family".
- Do NOT re-run N0007 (outcome shopping; §5.1 — the number is flat-tail-robust and
  25 MAE from the bar). Do NOT book "H2Pool is expensive" (timing parity with N0005/N0006
  implicates environment). Do NOT retry any second-similarity-bank-into-cond_map variant
  (75-key, h2-pooled, or otherwise) without a NEW falsifier and an interference analysis
  (separate qproj? detach? pre-trained warm-start?) — §11 no-silent-retry applies to the
  family, not just the instance. Seed stays 20260830; no cross-seed anything.

## Data sources (every number traced)

1. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0007_h0006/result.json` + `info.json`
   (read locally) — best 47.843936920166016 @ep32, 32 ep, elapsed 1803.67/1809.5,
   `budget_hit: false`, status `done`, H0006 solo.
2. Server TB dump of `/data/cac/tree/N0001_champion/N0002_h0001/N0007_h0006/run/latest/tb/t/events*`
   — full `val/mae`, `val/rmse`, `train/loss_epoch` 32-pt series + wall times
   (span 1733.4 s → 55.92 s/ep); `train/mae` all-NaN; `lr-AdamW` 1e-3→0.
3. Server TB dumps of parent N0002 + N0005_h0004 same tags — parent train 38.02→2.6183 /
   val 153.29→22.5641; N0005 train 20.53→~14 flat / val 212.08→48.12.
4. Server `best.pth` norm dump — N0007 h2pool `out` 0.197/bias 0.313 engaged, temp 0.0524;
   simprior `out` 0.361 (parent 0.587), temp 0.0547 (parent 0.0393), qproj 3.52 (parent 5.73);
   N0005 same suppression pattern.
5. `idea.md` §4–5 (bar 22.26 + dense-tail conjunct), `config.toml` (seed 20260830,
   augment=false, use_h2pool=true), `model.py` H2Pool + shared-qproj wiring (lines 232–269,
   177–180), N0006 `feedback/diagnostic.md` + `feedback/quant.md` (marginal-overrun reference).

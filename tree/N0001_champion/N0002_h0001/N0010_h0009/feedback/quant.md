# Quantitative verdict — N0010_h0009 (H0009 `use_densexp`) vs N0002_h0001

- **Node:** N0010_h0009 · **Parent:** N0002_h0001 · **Hypothesis:** H0009
- **Pre-registered falsifier (idea.md:29):** confirm iff child final EMA val MAE
  ≤ **22.26** at seed 20260830 under 32ep/1800s canonical protocol (live parent
  22.5641 @ep30; ≥0.30 lower) AND gt>500 dense-image mean |Δ| below parent 511.7.
- **Seed/protocol check:** child `config.toml:9` `seed = 20260830`,
  `augment = false` (`config.toml:33`); child `use_densexp = true`
  (`config.toml:28`), `use_simprior = true` inherited (`config.toml:27`).
  Same-seed paired contrast per AGENTS.md §6 — valid.

## 1. Parent-vs-child table (all from `result.json` files read locally)

| Metric | Parent N0002_h0001 | Child N0010_h0009 |
|---|---|---|
| Best val MAE | 22.564146041870117 @ep30 | 24.049957275390625 @ep32 |
| Final val MAE (ep32, TB) | 22.570467 | 24.049957 (best = final) |
| Epochs done | 32 / 32 | 32 / 32 |
| Wall seconds (`elapsed_s`) | 1755.0 (`elapsed` 1749.53) | 1794.9 (`elapsed` 1789.42) |
| `budget_hit` | false | false |
| Status | done (`code: ok`) | done (`code: ok`, `info.json: status done`) |
| `config_sha256` | 4d4c9baec459f657 | 8f8d14b1c6e058f0 |
| `model_sha256` | 611be2a9d70768b5 | 70410fe91dd6d346 |

- **Margin vs parent:** 24.04995728 − 22.56414604 = **+1.4858 worse** (bar
  demanded −0.30; child moves ~4.95× the bar magnitude in the wrong direction,
  ~50× the 0.03-scale paired-contrast precision).
- **Miss vs bar:** 24.04995728 − 22.26 = **+1.7900 above the 22.26 bar**.
- **Wall-clock:** +39.9 s (+2.3%) vs parent; `elapsed_s` 1794.9 is 5.1 s under
  τ_max 1800 — clean full-budget run, no timeout event.

## 2. Child full val/MAE curve (TB `val/mae`, 32 pts, server events file dump)

```
ep: 01 270.5622 | 02 138.1341 | 03 94.7896 | 04 70.7976 | 05 57.4545 | 06 47.2147
ep: 07 39.8305 | 08 34.7356 | 09 31.4848 | 10 29.6687 | 11 27.9815 | 12 27.0593
ep: 13 26.5348 | 14 26.0800 | 15 25.7206 | 16 25.2952 | 17 24.9702 | 18 24.6996
ep: 19 24.5105 | 20 24.3748 | 21 24.2680 | 22 24.2011 | 23 24.1564 | 24 24.1237
ep: 25 24.0986 | 26 24.0819 | 27 24.0725 | 28 24.0654 | 29 24.0617 | 30 24.0560
ep: 31 24.0520 | 32 24.0500 (best)
```

- **Never crosses the bar:** minimum over all 32 epochs is 24.0500 (ep32);
  closest approach still **1.7900 above 22.26**. No epoch is within 1.7 of
  the bar.
- **Never matches the parent:** child best (24.0500) sits between the parent's
  ep12 level (24.2563) and ep13 level (23.9073) — the child at ep32 performs
  where the parent was between ep12 and ep13, then plateaus.
- **Separation timing:** child trails from ep01 on: +117.2748 @ep01 (270.5622
  vs 153.2874), +68.0807 @ep02, +46.0014 @ep03, +31.6759 @ep04, +23.9536 @ep05,
  +16.9194 @ep06, +11.4250 @ep07, +7.4697 @ep08, +5.2003 @ep09, +4.2595 @ep10,
  +3.3064 @ep11, +2.8030 @ep12, then a +2.5–2.6 band ep13–14 narrowing to
  +1.4858 at the end.
- **Curve shape:** monotone descent for all 32 epochs (no uptick anywhere —
  every epoch improves on the last), but gains shrink to ~0.004/ep by ep28 and
  the last 5 readings all sit in 24.0500–24.0654. No U-turn, no spike, no
  late recovery.

## 3. Best-epoch edge-fluke check (ep32/32, final epoch — floor, not a dip)

- ep30 24.055971 · ep31 24.051979 · **ep32 24.049957 (best = final)**.
- **Best-vs-final:** 24.049957 − 24.049957 = **+0.0000** — best IS the final
  reading, so no overfit uptick by construction.
- **Last-5-epoch range (ep28–32):** 24.065378 − 24.049957 = **0.0154** — deep
  plateau; late descent rate ≈ −0.0039/ep. At that crawl rate closing the 1.79
  bar gap would take ~460 further epochs. The "best @ep32" is a flat floor
  ~1.79 above the bar, not a lucky dip — and it is also not a still-descending
  optimum that matters: one more epoch at −0.002–0.004 moves nothing against
  a 1.79 miss.
- **Late-best at 32/32 (100% of schedule) vs parent best at 30/32** — training
  converges to a floor under the added module; it does not degrade with more
  steps, but the floor is 1.49 above the parent.

## 4. Train loss vs parent + RMSE (optimization-failure reading)

- **Train `train/loss_epoch` final:** child **5.345045 @ep32** (minimum
  5.319098 @ep22, +0.0259 above min at the end) vs parent **2.618342 @ep32** —
  child **+2.7267 worse on train**. The expert residual does not even fit the
  training set better; the added ~4.3k params made optimization harder, not
  more expressive in a useful direction.
- **Child train-loss shape:** 19.8401 → 12.5100 → 11.5899 → 10.0756 → 8.3347,
  then 8.3–8.6 wobble ep06–09, descent to 5.3191 @ep22, then 5.32–5.35 flat
  for the last 10 epochs. Parent descends every late epoch to 2.6183.
  Train-worse AND val-worse by wide margins = underfit/optimization drag,
  not overfit. (Numeric note: child train starts −18.18 BETTER than the
  parent @ep01, 19.8401 vs 38.0236, while child val starts +117.27 WORSE —
  train/val split from the first epoch; causal attribution left to causal.md.)
- **Val RMSE:** child best = final **84.149635 @ep32** (monotone descent all
  32 epochs: 281.7592 → … → 84.1496); parent best 88.923416 @ep28, final
  89.239967. Child RMSE is **−4.7738 better (best-vs-best)** / **−5.0903
  better (final-vs-final)** — the one scalar that moves toward the parent.
  RMSE/MAE ratio falls (parent final 3.95 → child final 3.50) while MAE
  degrades: squared-error mass shrinks while mean absolute error grows, i.e.
  the error distribution changed shape but the global MAE bar still misses by
  1.79. Whether any of that RMSE move sits on the gt>500 tail is not
  evaluable from these scalars (no per-image dump) — left to qual/causal.
- **`train/mae`:** all-NaN in both parent and child (32/32 NaN each) —
  pre-existing logging artifact, non-deciding (same pattern as
  N0004/N0005/N0006/N0007/N0008/N0009).

## 5. Verdict vs pre-registered falsifier (numbers only)

- **REFUTE H0009.** Child best val MAE **24.0500 @ep32** exceeds the 22.26 bar
  by **1.7900** and is **1.4858 worse than the live parent** same-seed. The
  MAE conjunct of the dual falsifier fails outright (~50× the 0.03-scale
  paired-contrast precision in the wrong direction, on a clean late-plateau
  number with no truncation confound). The gt>500 dense-tail conjunct is not
  evaluable from quant scalars (no per-image dump in this run's artifacts)
  and is left to qual/causal — but the global-bar miss alone decides the
  quantitative verdict: nothing in these numbers supports confirmation.

## 6. Numeric anomalies

- **MAE-worse / RMSE-better split is the anomaly — and it does not help the
  hypothesis quantitatively.** Child val MAE +1.4858 worse while child val
  RMSE −5.0903 better (final-vs-final), with monotone-descent RMSE (32/32
  finite, no rise) unlike the N0008 MAE/RMSE pre-collapse divergence. The bar
  is written in MAE, so the split cannot rescue the verdict; it is recorded
  here as a distribution-shape observation for qual/causal to attribute (or
  dismiss) with per-image evidence.
- **Otherwise none that help the hypothesis.** No NaN/Inf (32/32 finite points
  for `val/mae`, `val/rmse`, `train/loss_epoch`); no spikes; no OOM/traceback;
  `budget_hit: false`, `status: done`, 32/32 epochs recorded — a clean
  full-schedule run whose floor is 1.79 above the bar.

## Data sources (every number traced)

1. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0010_h0009/result.json`
   (read locally) — best_mae 24.049957275390625, best_epoch 32, epochs 32,
   n_epochs_done 32, elapsed 1789.4179482460022, elapsed_s 1794.9,
   budget_hit false, code ok, shas 8f8d14b1c6e058f0 / 70410fe91dd6d346.
2. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/result.json` (read locally)
   — parent best 22.564146041870117 @ep30, elapsed/elapsed_s, shas.
3. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0010_h0009/info.json`
   (read locally) — status done, tested `["H0009"]`, train_seconds 1794.9,
   best_metric 24.049957275390625, epochs 32.
4. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0010_h0009/config.toml`
   (read locally) — seed 20260830, augment false, use_densexp true,
   use_simprior true.
5. Server TB direct `EventAccumulator` dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/N0010_h0009/run/latest/tb/t/events.out.tfevents.1789903448.*`
   — full child `val/mae`, `val/rmse`, `train/loss_epoch` series (32 pts
   each) plus `train/mae` 32/32-NaN check.
6. Server TB direct dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/run/latest/tb/t/events.out.tfevents.1789881056.*`
   — parent `val/mae`/`val/rmse`/`train/loss_epoch` full 32-pt series.
7. `idea.md:29` — falsifier bar 22.26 + dense-tail conjunct (pre-registered).

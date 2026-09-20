# Quantitative verdict — N0009_h0008 (H0008 `use_hires`) vs N0002_h0001

- **Node:** N0009_h0008 · **Parent:** N0002_h0001 · **Hypothesis:** H0008
- **Pre-registered falsifier (idea.md:4-6):** confirm iff child final EMA val MAE
  ≤ **22.26** at seed 20260830 under 32ep/1800s canonical protocol (live parent
  22.5641 @ep30; ≥0.30 lower) AND gt>500 dense-image mean |Δ| below parent 511.7.
- **Seed/protocol check:** child `config.toml:9` `seed = 20260830`,
  `augment = false` (`config.toml:33`); child `use_hires = true`
  (`config.toml:28`), `use_simprior = true` inherited (`config.toml:27`).
  Same-seed paired contrast per AGENTS.md §6 — valid.

## 1. Parent-vs-child table (all from `result.json` files read locally)

| Metric | Parent N0002_h0001 | Child N0009_h0008 |
|---|---|---|
| Best val MAE | 22.564146041870117 @ep30 | 26.453519821166992 @ep31 |
| Final val MAE (ep32, TB) | 22.5705 | 26.4536 |
| Epochs done | 32 / 32 | 32 / 32 |
| Wall seconds (`elapsed_s`) | 1755.0 (`elapsed` 1749.53) | 1815.2 (`elapsed` 1809.75) |
| `budget_hit` | false | true |
| Status | done (`code: ok`) | timeout (`info.json`, `code: ok`) |
| `config_sha256` | 4d4c9baec459f657 | cb9ed415419598f2 |
| `model_sha256` | 611be2a9d70768b5 | dcf9ce12b2cec929 |

- **Margin vs parent:** 26.45351982 − 22.56414604 = **+3.8894 worse** (bar
  demanded −0.30; child moves ~13.0× the bar magnitude in the wrong direction,
  ~130× the 0.03-scale paired-contrast precision).
- **Miss vs bar:** 26.45351982 − 22.26 = **+4.1935 above the 22.26 bar**.
- **Wall-clock:** trainer-`elapsed` +60.22 s (+3.44%) vs parent (56.55 vs
  54.67 s/ep, +1.88 s/ep); `elapsed` exceeds τ_max by 9.75 s, `elapsed_s` by
  15.2 s — yet all 32 epochs completed with best @ep31 and a following
  evaluated ep32, so the timeout is a wall-clock margin technicality, not a
  truncation (same marginal-overrun position as N0006_h0005).

## 2. Child full val/MAE curve (TB `val/mae`, 32 pts, server events file dump)

```
ep: 01 141.4797 | 02 76.1503 | 03 54.9702 | 04 44.4093 | 05 37.6764 | 06 33.6221
ep: 07 31.0434 | 08 29.3547 | 09 28.3240 | 10 27.6445 | 11 27.2385 | 12 26.9835
ep: 13 26.8163 | 14 26.7048 | 15 26.6315 | 16 26.5774 | 17 26.5357 | 18 26.5128
ep: 19 26.4979 | 20 26.4861 | 21 26.4767 | 22 26.4698 | 23 26.4634 | 24 26.4648
ep: 25 26.4617 | 26 26.4596 | 27 26.4584 | 28 26.4571 | 29 26.4559 | 30 26.4541
ep: 31 26.4535 (best) | 32 26.4536
```

- **Never crosses the bar:** minimum over all 32 epochs is 26.4535 (ep31);
  closest approach still **4.1935 above 22.26**. No epoch is within 4.1 of
  the bar.
- **Never matches the parent:** child best (26.4535) sits **above the parent's
  ep09 level** (parent 26.2845 @ep09; parent ep08 27.2659) — the child at ep31
  performs where the parent was between ep08 and ep09, then plateaus.
- **Separation timing:** child leads only at ep01 (141.4797 vs 153.2874,
  −11.8077), then trails from ep02 on: +6.0969 @ep02, +6.1820 @ep03,
  +5.2877 @ep04, +4.1754 @ep05, +3.3269 @ep06, a +2.0–2.7 band ep07–12,
  widening to +3.8894 at the end.
- **Curve shape:** monotone descent into an early floor — gains shrink to
  ~0.2/ep by ep10 and the last 8 readings all sit in 26.4535–26.4617. No
  U-turn, no spike, no late recovery.

## 3. Best-epoch edge-fluke check (ep31/32, late — floor, not a dip)

- ep29 26.4559 · ep30 26.4541 · **ep31 26.4535 (best)** · ep32 26.4536.
- **Best-vs-final:** 26.4536 − 26.4535 = **+0.0001** — no overfit uptick.
- **Last-5-epoch range (ep28–32):** 26.4571 − 26.4535 = **0.0036** — deep
  plateau; the "best @ep31" is not a lucky dip, it is a flat tail ~4.19 above
  the bar. Late-best at 31/32 (97% of schedule) vs parent best at 30/32 —
  training converges to a floor, it does not degrade with more steps.
- **Timeout confound check:** 32/32 `val/mae` points present, ep32 evaluated
  after the best — `_BudgetStop` fired during/after the final epoch and best
  selection is unaffected. Nothing about the flat tail suggests further epochs
  would close a 3.89 gap at a 0.004/5ep crawl rate.

## 4. Train loss vs parent + RMSE (optimization-failure reading)

- **Train `train/loss_epoch` final:** child **6.0013 @ep32** (5.9618 @ep31)
  vs parent **2.6183 @ep32** — child **+3.3830 worse on train**. The residual
  does not even fit the training set better; the added ~15.4k params made
  optimization harder, not more expressive in a useful direction.
- **Child train-loss shape:** 48.9357 → 8.3986 → 7.6544 → 8.2929 (uptick ep04)
  → 6.6447 → 5.9637, then 5.92–6.00 flat for the remaining ~26 epochs
  (minimum 5.9248 @ep12 — already +0.3265 worse than the parent's same-epoch
  5.5983). Parent descends every late epoch to 2.6183. Train-worse AND
  val-worse by wide margins = underfit/optimization drag, not overfit.
- **Val RMSE:** child best **96.6710 @ep12**, final **96.8673**; parent best
  88.9234 @ep28, final 89.2400. Child worse by **+7.7476 (best)** / **+7.6273
  (final)**. Child RMSE rises ep12→ep32 (+0.1963) while child MAE still falls
  (26.9835 → 26.4536) — mild MAE/RMSE divergence on a flat floor, no
  tail-rescue signal. RMSE/MAE ratio falls (parent final 3.95 → child final
  3.66) while both metrics degrade — the loss is broad-based, not
  tail-concentrated.
- **`train/mae`:** all-NaN in both parent and child (32/32 NaN each) —
  pre-existing logging artifact, non-deciding (same pattern as
  N0004/N0006/N0007/N0008).

## 5. Verdict vs pre-registered falsifier (numbers only)

- **REFUTE H0008.** Child best val MAE **26.4535 @ep31** exceeds the 22.26 bar
  by **4.1935** and is **3.8894 worse than the live parent** same-seed. The
  MAE conjunct of the dual falsifier fails outright (~130× the 0.03-scale
  paired-contrast precision in the wrong direction, on a clean late-plateau
  number with no truncation confound). The gt>500 dense-tail conjunct is not
  evaluable from quant scalars (no per-image dump in this run's artifacts)
  and is left to qual/causal — but the global-bar miss alone decides the
  quantitative verdict: nothing in these numbers supports confirmation.

## 6. Numeric anomalies

- **None that help the hypothesis.** No NaN/Inf (32/32 finite points for
  `val/mae`, `val/rmse`, `train/loss_epoch`); no spikes beyond the benign
  +0.6385 train uptick ep03→ep04 (7.6544 → 8.2929, re-descends next epoch);
  `budget_hit: true` / `status: timeout` reflects the 1800 s wall clock, not
  a crash — all 32 epochs completed and the best/final readings are valid.

## Data sources (every number traced)

1. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0009_h0008/result.json`
   (read locally) — best_mae 26.453519821166992, best_epoch 31, epochs 32,
   n_epochs_done 32, elapsed 1809.7473533153534, elapsed_s 1815.2,
   budget_hit true, code ok, shas cb9ed415419598f2 / dcf9ce12b2cec929.
2. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/result.json` (read locally)
   — parent best 22.564146041870117 @ep30, elapsed/elapsed_s, shas.
3. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0009_h0008/info.json`
   (read locally) — status timeout, tested `["H0008"]`, train_seconds 1815.2.
4. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0009_h0008/config.toml`
   (read locally) — seed 20260830, augment false, use_hires true,
   use_simprior true.
5. Server TB direct `EventAccumulator` dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/N0009_h0008/run/latest/tb/t/events.out.tfevents.1789901622.*`
   — full child `val/mae`, `val/rmse`, `train/loss_epoch` series (32 pts
   each) plus `train/mae` 32/32-NaN check.
6. Server TB direct dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/run/latest/tb/t/events.out.tfevents.1789881056.*`
   — parent `val/mae`/`val/rmse`/`train/loss_epoch` full 32-pt series plus
   `train/mae` 32/32-NaN check.
7. `idea.md:4-6` — falsifier bar 22.26 + dense-tail conjunct (pre-registered).

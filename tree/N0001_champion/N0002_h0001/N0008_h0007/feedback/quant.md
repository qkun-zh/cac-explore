# Quantitative verdict — N0008_h0007 (H0007 `use_simcal`) vs N0002_h0001

- **Node:** N0008_h0007 · **Parent:** N0002_h0001 · **Hypothesis:** H0007
- **Pre-registered falsifier (idea.md:4, §5):** confirm iff child final EMA val MAE
  ≤ **22.26** at seed 20260830 under 32ep/1800s canonical protocol (live parent
  22.5641 @ep30; ≥0.30 lower) AND gt>500 dense-image mean |Δ| below parent 511.7.
- **Seed/protocol check:** child `config.toml:9` `seed = 20260830`,
  `augment = false` (`config.toml:33`); child `use_simcal = true`
  (`config.toml:28`), `use_simprior = true` inherited (`config.toml:27`).
  Same-seed paired contrast per AGENTS.md §6 — valid.

## 1. Parent-vs-child table (all from `result.json` files read locally)

| Metric | Parent N0002_h0001 | Child N0008_h0007 |
|---|---|---|
| Best val MAE | 22.564146041870117 @ep30 | 27.637067794799805 @ep09 |
| Final val MAE (ep32, TB) | 22.5705 | NaN (ep10–32 all NaN; see §2) |
| Epochs done | 32 / 32 | 32 / 32 |
| Wall seconds (`elapsed_s`) | 1755.0 (`elapsed` 1749.53) | 1797.8 (`elapsed` 1792.07) |
| `budget_hit` | false | false |
| Status | done (`code: ok`) | done (`code: ok`, `info.json: status done`) |
| `config_sha256` | 4d4c9baec459f657 | 87ed8f2fbf33c621 |
| `model_sha256` | 611be2a9d70768b5 | a2488efa51ec0f15 |

- **Margin vs parent:** 27.63706779 − 22.56414604 = **+5.0729 worse** (bar
  demanded −0.30; child moves 16.9× the bar magnitude in the wrong direction).
- **Miss vs bar:** 27.63706779 − 22.26 = **+5.3771 above the 22.26 bar**.
- **Wall-clock:** +42.8 s (+2.4%) vs parent; `elapsed_s` 1797.8 is 2.2 s under
  τ_max 1800 — clean full-budget run, no timeout event.

## 2. Child full val/MAE curve (TB `val/mae`, server events file dump)

```
ep: 01 139.9451 | 02 67.0162 | 03 48.1929 | 04 39.2804 | 05 33.8597 | 06 30.8333
ep: 07 29.1571 | 08 28.2163 | 09 27.6371 (best) | 10 NaN | 11 NaN | 12 NaN
ep: 13 NaN | 14 NaN | 15 NaN | 16 NaN | 17 NaN | 18 NaN
ep: 19 NaN | 20 NaN | 21 NaN | 22 NaN | 23 NaN | 24 NaN
ep: 25 NaN | 26 NaN | 27 NaN | 28 NaN | 29 NaN | 30 NaN
ep: 31 NaN | 32 NaN
```

- **Never crosses the bar:** minimum over all 32 epochs is 27.6371 (ep09);
  closest approach still **5.3771 above 22.26**. No finite epoch is within 5.3
  of the bar.
- **Never matches the parent:** child best (27.6371) sits **above the parent's
  ep08 level** (parent 27.2659 @ep08; parent ep09 26.2845) — the child at ep09
  performs where the parent was at ~ep08, then collapses.
- **Collapse shape:** 9 finite points of normal descent (139.95 → 27.64),
  then `train/loss` (step-level) goes NaN at step 2249 (first NaN; last finite
  step 2199 val 2.8572) — i.e. inside ep10 — and every epoch-aggregated series
  (`val/mae`, `val/rmse`, `train/loss_epoch`) is NaN for ep10–32 (23 NaN
  points each). The runner still iterated all 32 epochs (`epoch` tag reaches
  31.0 at step 7295) with NaN losses, so `best_mae` froze at ep09 and
  `result.json` reports `done` / `budget_hit: false` over a diverged tail.

## 3. Best-epoch edge-fluke check (ep09, early — degradation confirmed)

- ep07 29.157143 · ep08 28.216259 · **ep09 27.637068 (best)** · ep10 NaN …
  ep32 NaN.
- **Best-vs-final:** final (ep32) is NaN — best-vs-final gap is not a plateau
  (+0.00x) but a **catastrophic divergence**: +23 NaN epochs after the best.
  Continued training cannot rescue a NaN tail; the "best @ep09" is the last
  finite point before collapse, not an optimum.
- **Early-peak reading:** best at ep09/32 (28% of schedule) vs parent best at
  ep30/32 — training degrades (to non-numeric) with more steps under the added
  module.

## 4. Train loss vs parent + RMSE (optimization-failure reading)

- **Train `train/loss_epoch`:** child ep01 41.897808, ep02 8.553652,
  ep03 8.650422, ep04 8.818274, ep05 7.204623, ep06 6.948510, ep07 5.958570,
  ep08 5.840749, ep09 5.848253, ep10–32 NaN; parent final **2.618342 @ep32**
  (parent ep09 6.146763). Child train loss at ep09 (5.8483) is already above
  the parent's ep09 (6.1468 → child −0.2985 better at that single epoch) but
  the run never reaches the parent's late-train regime — it goes NaN instead.
- **Val RMSE:** child 164.952026, 117.326256, 106.679527, 100.807899,
  97.120697, 95.484283, 95.837914, 96.630676, **97.468857 @ep09**, then NaN;
  parent best 88.9234 @ep28, final 89.2400. Child RMSE **rises ep06→ep09**
  (95.4843 → 97.4689, +1.9846) while child MAE still falls (30.8333 →
  27.6371) — MAE/RMSE diverge before the NaN, consistent with instability
  rather than coherent learning. At ep09 child RMSE already +8.23 above the
  parent final.
- **`train/mae`:** all-NaN in both parent and child (32/32 NaN) — pre-existing
  logging artifact, non-deciding (same pattern as N0004/N0006/N0007).

## 5. Verdict vs pre-registered falsifier (numbers only)

- **REFUTE H0007.** Child best val MAE **27.6371 @ep09** exceeds the 22.26 bar
  by **5.3771** and is **5.0729 worse than the live parent** same-seed. The
  MAE conjunct of the dual falsifier fails outright (~169× the 0.03-scale
  paired-contrast precision in the wrong direction, then diverges to NaN).
  The gt>500 dense-tail conjunct is not evaluable from quant scalars (no
  per-image dump in this run's artifacts) and is left to qual/causal — but
  the global-bar miss alone decides the quantitative verdict: nothing in
  these numbers supports confirmation.

## 6. Numeric anomalies

- **NaN collapse ep10→ep32 is the anomaly — and it hurts the hypothesis.**
  No NaN/Inf in the first 9 epochs; from step 2249 (`train/loss`) / ep10
  (all epoch tags) every loss/metric is NaN through ep32, yet the run
  reports 32/32 `done`. This is a training-divergence signature of the added
  `use_simcal` path (statistics division / SNR-softmax / extra residual —
  causal attribution left to causal.md), not a logging artifact: the parent
  under the identical harness is finite for all 32 epochs.
- **No budget/timeout confound:** `budget_hit: false`, `elapsed_s` 1797.8 <
  1800, 32/32 epochs recorded — the NaN tail is not a `_BudgetStop` truncation.

## Data sources (every number traced)

1. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0008_h0007/result.json`
   (read locally) — best_mae 27.637067794799805, best_epoch 9, epochs 32,
   n_epochs_done 32, elapsed 1792.0685770511627, elapsed_s 1797.8,
   budget_hit false, code ok, shas 87ed8f2fbf33c621 / a2488efa51ec0f15.
2. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/result.json` (read locally)
   — parent best 22.564146041870117 @ep30, elapsed/elapsed_s, shas.
3. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0008_h0007/info.json`
   (read locally) — status done, tested `["H0007"]`, train_seconds 1797.8.
4. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0008_h0007/config.toml`
   (read locally) — seed 20260830, augment false, use_simcal true,
   use_simprior true.
5. Server TB direct `EventAccumulator` dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/N0008_h0007/run/latest/tb/t/events.out.tfevents.1789897873.*`
   — child `val/mae` (9 finite + 23 NaN), `val/rmse`, `train/loss_epoch`,
   `train/loss` (145 pts; NaN from step 2249), `epoch`/`lr-AdamW` tails.
6. Server TB direct dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/run/latest/tb/t/events.out.tfevents.1789881056.*`
   — parent `val/mae`/`val/rmse`/`train/loss_epoch` full 32-pt series.
7. `idea.md:4` + §5 — falsifier bar 22.26 + dense-tail conjunct (pre-registered).

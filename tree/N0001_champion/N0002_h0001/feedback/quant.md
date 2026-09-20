# Quantitative verdict — N0002_h0001 (H0001 `use_simprior`) vs N0001_champion

- **Node:** N0002_h0001 · **Parent:** N0001_champion · **Hypothesis:** H0001
- **Pre-registered falsifier (idea.md:4-6):** confirm iff child final val MAE ≤ **22.99**
  at seed 20260830 under 32ep/1800s canonical protocol (live parent 23.293; ≥0.30 lower).
- **Seed/protocol check:** `seed = 20260830`, `augment = false` in both
  `config.toml` files; child `use_simprior = true` (parent default false). Same-seed
  paired contrast per AGENTS.md §6 — valid.

## 1. Parent-vs-child table (all from `result.json` files read locally)

| Metric | Parent N0001_champion | Child N0002_h0001 |
|---|---|---|
| Best val MAE | 23.293153762817383 @ep26 | 22.564146041870117 @ep30 |
| Final val MAE (ep32) | 23.3364 (TB) | 22.5705 (TB) |
| Epochs done | 32 / 32 | 32 / 32 |
| Wall seconds (`elapsed_s`) | 1699.4 | 1755.0 (`elapsed` 1749.53; `info.json` `train_seconds` 1755.0) |
| `budget_hit` | false | false |
| Status | (baseline) | done (`info.json`, `code: ok`) |
| `config_sha256` | 679f37c26bc72103 | 4d4c9baec459f657 |
| `model_sha256` | eaf2a8b93c280cba | 611be2a9d70768b5 |

- **Margin:** 23.29315376 − 22.56414604 = **0.7290** (≥0.30 bar by **2.43×**).
- **Clearance below bar:** 22.99 − 22.56414604 = **0.4259**.
- **Wall-clock overhead:** +55.6 s (+3.3%) vs parent — small, no budget pressure
  (45 s headroom under τ_max=1800 s).

## 2. Child full val/MAE curve (TB `val/mae`, 32 pts, server events file)

```
ep: 01 153.2874 | 02 70.0534 | 03 48.7882 | 04 39.1216 | 05 33.5010 | 06 30.2952
ep: 07 28.4055 | 08 27.2659 | 09 26.2845 | 10 25.4092 | 11 24.6751 | 12 24.2563
ep: 13 23.9073 | 14 23.5598 | 15 23.4425 | 16 23.2482 | 17 23.0597 | 18 22.9224
ep: 19 22.9487 | 20 22.9341 | 21 22.9015 | 22 22.8883 | 23 22.7773 | 24 22.7200
ep: 25 22.6848 | 26 22.6515 | 27 22.6285 | 28 22.5923 | 29 22.5659 | 30 22.5641
ep: 31 22.5733 | 32 22.5705
```

- Curve tail (eps 24–32, `curve.py` + TB agree): 22.72, 22.685, 22.652, 22.629,
  22.592, 22.566, 22.564, 22.573, 22.570. `curve.py` summary:
  `epochs=32 best=22.5641@ep30 last=22.5705` — matches `result.json` to 4 dp.

## 3. Bar-crossing epoch

- **First epoch ≤ 22.99: ep18 (22.9224).** Ep17 = 23.0597 (above bar).
- **Never recrosses:** every epoch 18–32 stays sub-bar (max 22.9487 @ep19).
- Strictly decreasing ep19→ep30 (12-epoch monotone descent); largest single-epoch
  uptick after ep10 is ep18→ep19, +0.0263 — noise, not a spike.

## 4. Plateau / overfit reading

- **Child last-5-epoch range (ep28–32):** max 22.5923 − min 22.5641 = **0.0282**
  (≈4% of the 0.729 margin) — deep plateau but slope still negative into ep30.
- **Child best-vs-final:** 22.5705 − 22.5641 = **+0.0064** — no overfit uptick;
  final is within 0.01 of best.
- **Parent last-5 range (ep28–32):** 23.3364 − 23.3070 = **0.0294**, but
  parent best-vs-final = +0.0432 (drift up after ep26) — classic mild overfit.
  The child extends the descent 4 epochs past the parent's best epoch and lands
  0.73 lower.
- **Train loss (`train/loss_epoch`)** falls monotonically 38.02 → 2.62 across all
  32 epochs — still descending at the end; optimization did not stall.
- **RMSE check:** child best RMSE 88.9234 @ep28, final 89.24; parent best 90.6424
  @ep24, final 91.7522. RMSE gain −1.72 (−1.9%), MAE gain −0.73 (−3.1%);
  RMSE/MAE ratio flat (parent final 3.93 → child final 3.95), so the win is spread
  across the bulk, not just catastrophe images.

## 5. Verdict vs pre-registered falsifier

- **CONFIRM H0001.** Child best val MAE **22.5641 @ep30** ≤ 22.99 bar;
  margin over live parent (23.2932 @ep26) is **0.7290**, clearing the 0.30 bar by
  ~2.4×. Same seed (20260830), full 32-epoch canonical run, `budget_hit: false`,
  `status: done`.

## 6. Numeric anomalies

- **None.** No NaN/Inf in TB series (32/32 finite points for `val/mae`,
  `val/rmse`, `train/loss_epoch`); no spikes (max late uptick +0.026);
  no `_BudgetStop`/timeout/OOM/traceback in `chain_run.log`. The single grep hit
  for `error|...` is a benign torchmetrics `MeanMetric compute-before-update`
  `UserWarning` (also present in the smoke phase). Log tail confirms
  `` `Trainer.fit` stopped: `max_epochs=32` reached `` and final val/mae 22.570.

## Data sources (every number traced)

1. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/result.json` (read locally) —
   best_mae, best_epoch, epochs, elapsed/elapsed_s, budget_hit, shas.
2. `/home/qkun/cac/tree/N0001_champion/result.json` (read locally) — parent row.
3. Server TB via `curve.py N0002_h0001` + direct `EventAccumulator` dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/run/latest/tb/t/events*` —
   full child `val/mae`, `val/rmse`, `train/loss_epoch` series.
4. Server TB direct dump of `/data/cac/tree/N0001_champion/run/latest/tb/t/events*`
   (note: `curve.py N0001_champion` prints "no tb yet" — path bug: it looks under
   `.../N0001_champion/N0001_champion/`; the parent events file exists at the path
   above) — full parent `val/mae` + `val/rmse` series.
5. `/data/cac/tree/N0001_champion/N0002_h0001/chain_run.log` (grep + tail) —
   stop reason, final val/mae line, anomaly scan.
6. `/data/cac/tree/N0001_champion/N0002_h0001/info.json` (server) — status done,
   best_metric 22.564146041870117, train_seconds 1755.0.
7. Local `config.toml` files (both nodes) — seed 20260830, augment=false,
   child `use_simprior = true`.
8. `idea.md:4-6` — falsifier bar 22.99 (pre-registered, not re-derived).

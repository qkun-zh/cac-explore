# Quantitative verdict — N0006_h0005 (H0005 `use_verify`) vs N0002_h0001

- **Node:** N0006_h0005 · **Parent:** N0002_h0001 · **Hypothesis:** H0005
- **Pre-registered falsifier (idea.md:4-6):** confirm iff child final EMA val MAE
  ≤ **22.26** at seed 20260830 under 32ep/1800s canonical protocol (live parent
  22.5641 @ep30; ≥0.30 lower) AND gt>500 dense-image mean |Δ| below parent 511.7.
- **Seed/protocol check:** `seed = 20260830`, `augment = false` in both
  `config.toml` files; child `use_verify = true` (parent default false), child
  `use_simprior = true` inherited. Same-seed paired contrast per AGENTS.md §6 —
  valid.

## 1. Parent-vs-child table (all from `result.json` files read locally; server copy identical)

| Metric | Parent N0002_h0001 | Child N0006_h0005 |
|---|---|---|
| Best val MAE | 22.564146041870117 @ep30 | 23.58022117614746 @ep31 |
| Final val MAE (ep32, TB) | 22.5705 | 23.581057 |
| Epochs done | 32 / 32 | 32 / 32 |
| Wall seconds (`elapsed_s`) | 1755.0 (`elapsed` 1749.53) | 1815.4 (`elapsed` 1809.83) |
| `budget_hit` | false | true |
| Status | done (`code: ok`) | timeout (`info.json`, `code: ok`) |
| `config_sha256` | 4d4c9baec459f657 | 88b8f37902f414f1 |
| `model_sha256` | 611be2a9d70768b5 | d239d6413086b852 |

- **Margin vs parent:** 23.58022118 − 22.56414604 = **+1.0161 worse** (bar
  demanded −0.30; child moves 3.39× the bar magnitude in the wrong direction).
- **Miss vs bar:** 23.58022118 − 22.26 = **+1.3202 above the 22.26 bar**.
- **Wall-clock:** +60.4 s (+3.4%) vs parent; `elapsed_s` 1815.4 exceeds τ_max by
  15.4 s — the run still completed all 32 epochs (log shows both `[budget]
  exceeded 1800s — stopping early` and `` `Trainer.fit` stopped:
  `max_epochs=32` reached ``; best comes from ep31, final ep32 recorded).

## 2. Child full val/MAE curve (TB `val/mae`, 32 pts, server events file dump)

```
ep: 01 136.7200 | 02 66.9365 | 03 47.6049 | 04 38.3858 | 05 33.1605 | 06 30.1898
ep: 07 28.0150 | 08 26.5397 | 09 25.6442 | 10 25.0957 | 11 24.8769 | 12 24.6071
ep: 13 24.4650 | 14 24.2548 | 15 24.0408 | 16 23.9010 | 17 23.8640 | 18 23.8410
ep: 19 23.7142 | 20 23.6736 | 21 23.7800 | 22 23.7730 | 23 23.7419 | 24 23.7065
ep: 25 23.6821 | 26 23.6680 | 27 23.6359 | 28 23.5891 | 29 23.5813 | 30 23.5810
ep: 31 23.5802 | 32 23.5811
```

- **Never crosses the bar:** minimum over all 32 epochs is 23.5802 (ep31);
  closest approach still **1.3202 above 22.26**. No epoch is even within 1.3 of
  the bar.
- **Never matches the parent:** child best (23.5802) sits **above the parent's
  ep14 level** (parent 23.5598 @ep14) — the child at ep32 performs where the
  parent was at ep14, then plateaus.
- **Curve shape:** descent stalls after ep20 (23.6736); eps 21–22 tick up
  (+0.1064 ep20→ep21, the largest late uptick), then grind down only 0.20 over
  the last 10 epochs into a flat tail.

## 3. Best-epoch edge-fluke check (ep31/32)

- ep29 23.581274 · ep30 23.580952 · **ep31 23.580221 (best)** · ep32 23.581057.
- **Best-vs-final:** 23.581057 − 23.580221 = **+0.0008** — no overfit uptick.
- **Last-5-epoch range (ep28–32):** 23.589106 − 23.580221 = **0.0089** — deep
  plateau; the "best @ep31" is not a lucky dip, it is a flat tail ~1.32 above
  the bar. Extending training would not plausibly close a 1.32 gap at a
  0.009/5ep crawl rate.

## 4. Train loss vs parent + RMSE (optimization-failure reading)

- **Train `train/loss_epoch` final:** child **2.945921 @ep32** (2.943310 @ep31)
  vs parent **2.618342 @ep32** — child **+0.3276 worse on train**. The gate does
  not even fit the training set better; the added 19,713 params made
  optimization harder, not more expressive in a useful direction.
- **Child train-loss tail (last 6):** 3.5097, 3.5840, 3.2983, 3.2188, 3.1416,
  3.0440, 2.9641, 2.9433, 2.9459 — still descending but ~0.33 above the
  parent's endpoint at every late epoch.
- **Val RMSE:** child best **92.3955 @ep19**, final **93.0648**; parent best
  88.9234 @ep28, final 89.2400. Child worse by **+3.4721 (best)** / **+3.8248
  (final)**. RMSE/MAE ratio rises (parent final 3.95 → child final 3.97) while
  both metrics degrade — no catastrophe-image rescue hides in the RMSE; the
  loss is uniform, not tail-concentrated.

## 5. Verdict vs pre-registered falsifier (numbers only)

- **REFUTE H0005.** Child best val MAE **23.5802 @ep31** exceeds the 22.26 bar
  by **1.3202** and is **1.0161 worse than the live parent** same-seed. The
  MAE conjunct of the dual falsifier fails outright (44× the 0.03-scale
  paired-contrast precision in the wrong direction). The gt>500 dense-tail
  conjunct is not evaluable from quant scalars (no per-image dump in this
  run's artifacts) and is left to qual/causal — but the global-bar miss alone
  decides the quantitative verdict: nothing in these numbers supports
  confirmation.

## 6. Numeric anomalies

- **None that help the hypothesis.** No NaN/Inf (32/32 finite points for
  `val/mae`, `val/rmse`, `train/loss_epoch`); no spikes beyond the benign
  +0.1064 ep20→ep21 bump; no OOM/traceback (log tail is NumPy-writability
  `UserWarning` noise + clean budget/max-epoch stop lines). `budget_hit: true`
  / `status: timeout` reflects the 1800 s wall clock, not a crash — all 32
  epochs completed and the best/final readings are valid.

## Data sources (every number traced)

1. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0006_h0005/result.json`
   (read locally; server copy `cat` identical) — best_mae, best_epoch, epochs,
   elapsed/elapsed_s, budget_hit, shas.
2. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/result.json` (read locally)
   — parent row.
3. Server TB direct `EventAccumulator` dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/N0006_h0005/run/latest/tb/t/events*`
   — full child `val/mae`, `val/rmse`, `train/loss_epoch` series (32 pts each).
4. Server TB direct dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/run/latest/tb/t/events*` — parent
   `val/mae` tail, `val/rmse` best/final, `train/loss_epoch` final 2.618342.
5. `/tmp/train-n0006.log` (server tail) — `[budget] exceeded 1800s`, final
   `val/mae: 23.581`, `` `Trainer.fit` stopped: `max_epochs=32` reached ``.
6. `/tmp/verify_n0006.log` (server) — solo-switch check (`delta=19713`,
   `VERIFY_ALL_OK`).
7. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0006_h0005/info.json`
   (read locally) — status timeout, tested `["H0005"]`.
8. `idea.md:4-6` — falsifier bar 22.26 + dense-tail conjunct (pre-registered).

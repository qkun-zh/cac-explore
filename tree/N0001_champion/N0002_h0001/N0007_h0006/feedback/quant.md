# Quantitative verdict — N0007_h0006 (H0006 `use_h2pool`) vs N0002_h0001

- **Node:** N0007_h0006 · **Parent:** N0002_h0001 · **Hypothesis:** H0006
- **Pre-registered falsifier (idea.md:4, §5):** confirm iff child final EMA val MAE
  ≤ **22.26** at seed 20260830 under 32ep/1800s canonical protocol (live parent
  22.5641 @ep30; ≥0.30 lower) AND gt>500 dense-image mean |Δ| below parent 511.7.
- **Seed/protocol check:** `seed = 20260830`, `augment = false` in child
  `config.toml`; child `use_h2pool = true` (parent default false), child
  `use_simprior = true` inherited. Same-seed paired contrast per AGENTS.md §6 —
  valid.

## 1. Parent-vs-child table (all from `result.json` files read locally)

| Metric | Parent N0002_h0001 | Child N0007_h0006 |
|---|---|---|
| Best val MAE | 22.564146041870117 @ep30 | 47.843936920166016 @ep32 |
| Final val MAE (ep32, TB) | 22.570467 | 47.843937 (= best) |
| Epochs done | 32 / 32 | 32 / 32 |
| Wall seconds (`elapsed_s`) | 1755.0 (`elapsed` 1749.53) | 1809.5 (`elapsed` 1803.67) |
| `budget_hit` | false | false |
| Status | done (`code: ok`) | done (`code: ok`) |
| `config_sha256` | 4d4c9baec459f657 | 4ba6d04d77ab6a88 |
| `model_sha256` | 611be2a9d70768b5 | e709b1c83c4c0055 |

- **Margin vs parent:** 47.84393692 − 22.56414604 = **+25.2798 worse** (bar
  demanded −0.30; child moves 84.3× the bar magnitude in the wrong direction).
- **Miss vs bar:** 47.84393692 − 22.26 = **+25.5839 above the 22.26 bar**.
- **Level:** child 47.84 is **2.12× the parent level** — catastrophic blow-up,
  same family as the N0005_h0004 timeout point (48.12; child 0.28 below that
  uninterpretable mark, but this run completed).
- **Wall-clock:** +54.5 s (+3.1%) vs parent; `elapsed_s` 1809.5 exceeds τ_max by
  9.5 s yet `budget_hit: false`, `status: done` — same marginal/environmental
  overrun class as N0006 (+9.8 s); all 32 epochs completed, readings valid.

## 2. Child full val/MAE curve (TB `val/mae`, 32 pts, server events file dump)

```
ep: 01 219.6360 | 02 104.4560 | 03 75.9999 | 04 58.2163 | 05 53.3305 | 06 54.8502
ep: 07 50.1067 | 08 48.5376 | 09 48.2841 | 10 48.2827 | 11 48.2661 | 12 48.3158
ep: 13 48.4371 | 14 48.5433 | 15 48.7322 | 16 48.7667 | 17 48.7487 | 18 48.6914
ep: 19 48.5762 | 20 48.4143 | 21 48.2949 | 22 48.2269 | 23 48.1146 | 24 48.0558
ep: 25 48.0149 | 26 47.9632 | 27 47.9057 | 28 47.8942 | 29 47.8810 | 30 47.8627
ep: 31 47.8518 | 32 47.8439 (best)
```

- **Never crosses the bar:** minimum over all 32 epochs is 47.8439 (ep32);
  closest approach still **25.5839 above 22.26**. No epoch is within 25 of the
  bar or within 25.7 of the parent.
- **Curve shape:** violent early collapse (219.64 → 48.27 by ep11), a +0.50
  mid-training hump peaking ep16 (48.7667), then a 0.42 grind down over the last
  16 epochs into a flat tail at ~47.9. The run converges — to a level more than
  double the parent.
- **Plateau-trustworthy level, moot for the verdict:** last-21-epoch drift
  (ep11→ep32) is only 0.4222, so the ~48 level is stable, not a transient — but
  at 84× the bar magnitude in the wrong direction no plateau subtlety matters.

## 3. Best-epoch edge-fluke check (ep32/32)

- ep30 47.862694 · ep31 47.851799 · **ep32 47.843937 (best = final)**.
- **Best-vs-final:** **0.000000** — identical by construction (best_epoch 32 =
  last epoch).
- **Edge check: TRUE (truncated).** Best sits exactly on the final epoch, so
  continued training cannot be ruled out from this run alone. Last-3-epoch crawl
  is 0.0188 total; closing a 25.58 gap at that rate is not plausible, and the
  edge does not soften the verdict — it only notes the run gives no
  post-plateau evidence either way.

## 4. Train loss vs parent + RMSE (optimization-failure reading)

- **Train `train/loss_epoch` final:** child **13.933559 @ep32** vs parent
  **2.618342 @ep32** — child **+11.3152 worse on train (5.3× the parent
  level)**. Child train loss never drops below 13.93 after ep10 (ep10 13.928304
  is the run minimum; tail oscillates 13.93–13.99 with zero late descent). The
  branch does not even fit the training set — train-worse + val-worse, the
  N0006-style corruption signature named in idea.md §4, not a resolution gain.
- **Val RMSE:** child best **123.823975 @ep04**, final **128.202408**; parent
  best 88.923416 @ep28, final 89.239967. Child worse by **+34.9006 (best)** /
  **+38.9624 (final)**. RMSE/MAE ratio falls (parent final 3.95 → child final
  2.68) while both metrics degrade catastrophically — no catastrophe-image
  rescue hides in the RMSE; the failure is global, not tail-concentrated.
- **RMSE best-vs-final divergence:** child RMSE best occurs at ep04 while MAE
  best sits at ep32 — the two metrics disagree on the optimum by 28 epochs,
  consistent with a corrupted evidence path rather than coherent learning.

## 5. Verdict vs pre-registered falsifier (numbers only)

- **REFUTE H0006.** Child best val MAE **47.8439 @ep32** exceeds the 22.26 bar
  by **25.5839** and is **25.2798 worse than the live parent** same-seed. The
  MAE conjunct of the dual falsifier fails outright (~845× the 0.03-scale
  paired-contrast precision in the wrong direction). The gt>500 dense-tail
  conjunct is not evaluable from quant scalars (no per-image dump in this
  run's artifacts) and is left to qual/causal — but the global-bar miss alone
  decides the quantitative verdict: nothing in these numbers supports
  confirmation.

## 6. Numeric anomalies

- **None that help the hypothesis.** No NaN/Inf in the deciding series (32/32
  finite points for `val/mae`, `val/rmse`, `train/loss_epoch`); `train/mae` is
  all-NaN (train MAE simply not logged under this protocol — logging artifact,
  non-deciding). No OOM/traceback; `budget_hit: false` / `status: done` with
  32/32 epochs — the 9.5 s τ_max overrun is wall-clock noise, not a stop event.

## Data sources (every number traced)

1. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0007_h0006/result.json`
   (read locally) — best_mae, best_epoch, epochs, elapsed/elapsed_s,
   budget_hit, shas.
2. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/result.json` (read locally)
   — parent row.
3. `/home/qkun/cac/tree/N0001_champion/N0002_h0001/N0007_h0006/info.json`
   (read locally) — status done, tested `["H0006"]`, train_seconds 1809.5.
4. Server TB direct `EventAccumulator` dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/N0007_h0006/run/latest/tb/t/events.out.tfevents.1789896053.*`
   — full child `val/mae`, `val/rmse`, `train/loss_epoch` series (32 pts each).
5. Server TB direct dump of
   `/data/cac/tree/N0001_champion/N0002_h0001/run/latest/tb/t/events.out.tfevents.1789881056.*`
   — parent `val/mae` best/final, `val/rmse` best/final, `train/loss_epoch`
   final 2.618342.
6. `idea.md:4` + §5 — falsifier bar 22.26 + dense-tail conjunct (pre-registered).

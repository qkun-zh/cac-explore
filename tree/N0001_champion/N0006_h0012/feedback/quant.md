# N0006_h0012 — Quantitative feedback (H0012 SOLO)

Sources actually read (no recollection): `result.json`, `info.json`, `config.toml`,
`model.py`, `idea.md`, `run/latest/exec.log` (child) and
`tree/N0001_champion/run/latest/run_node.log` (champion), both TB event files
(local `EventAccumulator`, tensorboard 2.21.0), and the server copies
(`/data/cac/tree/...`, `result.json` byte-identical to local; `best.pth` loaded
under `/data/miniconda/envs/cac/bin/python`). Delta convention: **child − champion**;
negative = child better (lower MAE).

## 1. Exact final numbers

| field | N0006_h0012 (H0012 on) | N0001_champion (parent) |
|---|---|---|
| best_mae | **21.49515724182129** (TB `val/mae` step 6383) | 21.458858489990234 (step 7295) |
| best_epoch | 28 | 32 |
| last-epoch val/mae | 21.671781539916992 (ep32) | 21.458858489990234 (ep32) |
| n_epochs_done / epochs | 32 / 32 | 32 / 32 |
| budget_hit | false (τ_max 1800 s) | false (τ_max 1800 s) |
| elapsed_s (wall) | 1771.0 | 1704.5 |
| elapsed (trainer loop) | 1765.6064641475677 | 1699.1245362758636 |
| config_sha256 | `ff16897f27a830f7` | `7954b5f6bb4829ab` |
| model_sha256 | `ba9650ed47d01c7e` | `eaf2a8b93c280cba` |

Both hashes re-verified: `sha256(config.toml)[:16]` / `sha256(model.py)[:16]` match the
recorded values for both runs. Config delta = `use_count_temp = true` only; model delta =
`CountPrior` MLP + temperature scaling path, single switch.

## 2. Epoch-by-epoch child vs champion (`val/mae`)

| ep | child | champ | Δ | | ep | child | champ | Δ |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 151.2546 | 157.3232 | **−6.0686** | | 17 | 23.3585 | 23.1817 | +0.1769 |
| 2 | 74.9010 | 76.2375 | **−1.3365** | | 18 | 23.0512 | 22.9429 | +0.1084 |
| 3 | 53.6889 | 53.2973 | +0.3917 | | 19 | 22.8534 | 22.7674 | +0.0860 |
| 4 | 42.4734 | 42.5154 | **−0.0420** | | 20 | 22.5524 | 22.6830 | **−0.1306** |
| 5 | 35.9252 | 35.7347 | +0.1905 | | 21 | 22.2128 | 22.4579 | **−0.2451** |
| 6 | 31.8090 | 31.3749 | +0.4341 | | 22 | 21.9408 | 22.2909 | **−0.3502** |
| 7 | 29.3477 | 28.7491 | +0.5986 | | 23 | 21.7400 | 22.0963 | **−0.3563** |
| 8 | 27.8061 | 27.0146 | +0.7915 | | 24 | 21.6532 | 21.8620 | **−0.2088** |
| 9 | 26.5801 | 25.9413 | +0.6388 | | 25 | 21.5716 | 21.7278 | **−0.1562** |
| 10 | 25.7510 | 25.1930 | +0.5580 | | 26 | 21.5408 | 21.6732 | **−0.1323** |
| 11 | 25.2333 | 24.7454 | +0.4879 | | 27 | 21.5147 | 21.6029 | **−0.0883** |
| 12 | 24.7589 | 24.3498 | +0.4091 | | 28 | 21.4952 | 21.5615 | **−0.0664** |
| 13 | 24.4825 | 24.1115 | +0.3710 | | 29 | 21.5374 | 21.5198 | +0.0177 |
| 14 | 24.2215 | 23.7738 | +0.4477 | | 30 | 21.5923 | 21.4860 | +0.1063 |
| 15 | 23.9233 | 23.6001 | +0.3232 | | 31 | 21.6369 | 21.4686 | +0.1683 |
| 16 | 23.5461 | 23.4119 | +0.1342 | | 32 | 21.6718 | 21.4589 | +0.2129 |

- **Child leads in 12/32 epochs.** Magnitudes: ep1 −6.0686, ep2 −1.3365, then ep23
  −0.3563, ep22 −0.3502, ep21 −0.2451, ep24 −0.2088, ep25 −0.1562, ep26 −0.1323, ep20
  −0.1306, ep27 −0.0883, ep28 −0.0664, ep4 −0.0420. Child loses 20/32; worst ep8 +0.7915,
  ep9 +0.6388, ep7 +0.5986, ep10 +0.5580.
- **Regime structure**: losing stretch ep5–19 (15 straight epochs, mean Δ +0.359), leading
  stretch ep20–28 (9 straight, mean Δ −0.193), losing again ep29–32 (mean Δ +0.126). All-32
  mean Δ = −0.0790, but that is ep1–2 dominated; excluding ep1–2 the mean is +0.1625.
- **Tail divergence**: champion ep26→32 falls monotonically 21.6732→21.4589 (net −0.2143,
  −0.0357/epoch) and its best is its last epoch. Child ep26→28 falls only 21.5408→21.4952
  (net −0.0457, −0.0229/epoch), then ep28→32 rises monotonically +0.1766 (21.4952→21.6718,
  +0.0442/epoch). Swing of the gap from ep26 (−0.1323) to ep32 (+0.2129) = 0.3452.
- Val RMSE tail agrees directionally: child ep28 80.5869 vs champion ep28 80.9238 (child
  better), but champion continues to 79.7708 while child ends 80.5307 (Δ +0.7599 final).
- **Stream-confound caveat (important)**: tau ≡ 1 at initialization by construction
  (`CountPrior` last layer zero-init → z = 0 → clamp(1+z) = 1), so step-0 child == champion,
  yet ep1 train loss is 65.9783 (child) vs 50.3046 (champion) and ep1 val MAE differs by
  −6.07. The two runs did not share the same epoch-1 data/augmentation stream (extra init
  draws shifting a global RNG and/or residual nondeterminism), so early-epoch and even later
  per-epoch deltas mix mechanism with trajectory divergence. Train loss is lower for the
  child from ep20 onward (ep32 2.5294 vs 2.7062) but is not comparable across streams.

## 3. Best-vs-best / best-vs-final vs the two lines

Champion line = 21.458858489990234 (ep32; its best = its final). Pre-registered bar =
21.4589 − 0.40 = **21.0589**.

| quantity | value | vs champion 21.4589 | vs bar 21.0589 |
|---|---|---|---|
| child best (ep28) | 21.49515724182129 | **+0.0363 worse** | **+0.4363 above bar** |
| child final (ep32) | 21.671781539916992 | +0.2129 worse | +0.6129 above bar |
| champion final/best | 21.458858489990234 | 0 | +0.4000 above bar |

The bar demanded a −0.40 move; the observed best-epoch move is +0.0363 (opposite direction),
so the shortfall equals 0.4363. Even oracle early-stopping at ep28 (the global best, matching
`best_epoch`) cannot reach the bar; the best-vs-best gap itself is +0.0363.

## 4. Overhead dissection (1771.0 vs 1704.5 s, +66.5 s, +3.90%)

- Identical harness overhead: `elapsed_s − elapsed` = +5.39 s (child) vs +5.38 s (champion);
  the whole +66.5 s sits in the trainer loop (+66.482 s).
- Epoch-interval wall times from TB (ep2..32, 31 intervals): child 54.74 ± 0.24 s vs
  champion 52.68 ± 0.13 s; Δ = +2.07 ± 0.24 s/epoch, child slower in **31/31** intervals
  (min +1.66, max +2.47). Final-epoch throughput 5.06 vs 5.28 it/s (−4.2%). This is a steady
  throughput difference, not one-off stalls.
- Params: child 31,349,476 − champion 31,324,771 = **+24,705 (+0.0788%)**; the only new
  tensors are `head.count_prior.mlp.{0,2}.{weight,bias}` (64×384, 64, 1×64, 1). Total is
  under the 32 M cap.
- FLOPs estimate (conservative): CountPrior forward ≈ 24,640 MACs/sample (≈49.3 kFLOPs),
  plus one GAP over fine (128×96×96 ≈ 1.18 M adds); forward+backward ≲ 3.7 MFLOPs/sample
  even counting the GAP at 1 FLOP/add and 3× factor. Backbone forward alone is ~13.1 GFLOPs
  at 384² (ConvNeXt-Tiny class, scaled from 224²), so a full fwd+bwd step is ≳39 GFLOPs/sample.
  Added arithmetic is ≤ ~0.01% of step compute, and ≤ ~0.0135 TFLOPs/epoch against a
  ≥142 TFLOPs/epoch backbone lower bound.
- Conclusion: the +3.9% wall-clock overhead is ~50× the +0.079% parameter share and several
  orders of magnitude above the added-arithmetic share; it is **not explainable by the extra
  parameters/FLOPs as compute**. The uniform 31/31 pattern is equally consistent with a steady
  environment/host effect (shared RTX 3060 pod; data-loader/co-tenant load) as with a fixed
  per-step orchestration cost of the added ops. With n = 1 run per arm and no interleaved
  control, the split between module-attributable overhead and run/environment variance is
  **not identifiable**; a same-session A/B with `use_count_temp` toggled (or the queued
  same-seed champion replicate for the noise floor) is required. Note the budget margin:
  child finished with only 29.0 s (1.61%) headroom vs 95.5 s (5.31%) for the parent.

## 5. Internal consistency checks

- `result.json.best_mae` == min over 32 TB `val/mae` scalars (21.49515724182129 at step 6383),
  exactly bit-equal; `best_epoch` 28 == argmin epoch; exec.log last print 21.672 == TB ep32.
- Server `best.pth` has `epoch = 28`, `best_mae = 21.49515724182129`; checkpoint holds exactly
  the 4 `count_prior` tensors with shapes (64,384), (64,), (1,64), (1,); param count confirms
  +24,705. Written 16:01, final epoch logged 16:04 — never overwritten (ep32 MAE worse).
- Server `result.json` is byte-identical to local; both checksums re-derived from file bytes.
- `n_epochs_done` = `epochs` = 32 = number of TB scalars; `budget_hit=false` consistent with
  1771.0 < 1800 (but see 1.61% headroom above).
- Caveat: no optimizer state in `best.pth`, so training-resume consistency cannot be checked
  (not needed for inference/ledger).

## So what for the next cycle

- Treat 21.4952 (ep28) as the observed best with a +0.4363 distance to the 21.0589 bar; the
  queued same-seed champion replicate (`repro_run.py`) is now needed to bound the run-to-run
  noise floor before any sub-0.30 delta in the next cycles is read as signal.
- The two arms diverged from epoch 1 (train loss 65.98 vs 50.30 while tau ≡ 1 at init), so
  future single-switch comparisons should pin the data/augmentation stream (fixed generator or
  first-batch hash logged) before per-epoch deltas are used as mechanism evidence.
- Budget arithmetic: the child ended 29 s under τ_max with a uniform +2.1 s/epoch; pre-register
  the next 32-epoch node with either a 31-epoch cap or explicit early-stop-at-best, or a slow
  run may convert a `done` into a `timeout` by environment noise alone.

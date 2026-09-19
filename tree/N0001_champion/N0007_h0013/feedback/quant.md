# Quantitative feedback — N0007_h0013 (H0013 solo: learned subpixel upsample in FineFuser)

Role: Quantitative Feedback Agent. Sources actually read (2026-09-19):

- `tree/N0001_champion/N0007_h0013/result.json`, `info.json`, `config.toml`, `model.py`, `idea.md`
- child TB: `tree/N0001_champion/N0007_h0013/run/latest/tb/t/events.out.tfevents.1789805101...14113.0`
- parent TB: `tree/N0001_champion/run/latest/tb/t/events.out.tfevents.1789796237...5824.0`
- `run/latest/exec.log`; parent `run/latest/run_node.log`; `tree/N0001_champion/N0006_h0012/result.json`
- server read-only: `/data/cac/tree/N0001_champion/N0007_h0013/run/latest/best.pth` (torch.load)
- ledger read-only: `memory/hypotheses.jsonl` H0013 evidence (2026-09-19T16:33:42+0800)

Design change is exactly one config key + one code path (diffed): child adds `use_subpixel_up = true`;
`FineFuser.__init__` adds `top_up = Conv2d(128, 512, 3, groups=128) + PixelShuffle(2)`, replacing
`F.interpolate(..., scale_factor=2, bilinear)` on the coarse branch. Params: 5120 new (4608 W + 512 b),
trainable 3.5 M / total 31.3 M. Train-batch counts, LR schedule, and train/loss step indices are
bit-identical to the champion (schedule invariant intact).

## 1. Final numbers

| Quantity | Value | Evidence |
|---|---|---|
| best_mae | 22.454483032226562 | result.json, TB `val/mae` last, info.json `best_metric`, best.pth `best_mae` |
| best_epoch | 32 (= last epoch, argmin of 32-point curve) | result.json, best.pth `epoch = 32` (1-based per `calls/best_checkpoint.py`) |
| last val/mae | 22.454483 (= best; curve still descending) | TB scalar @step 7295 |
| epochs | 32/32 | result.json `epochs` = `n_epochs_done`; TB 32 val points; `epoch` tag max 31 (0-based); 228 steps/epoch, final step 7295 |
| budget_hit | false | result.json; `elapsed_s` 1708.1 s < 1800 s (headroom 91.9 s) |
| elapsed | 1702.651 s (fit timer) / 1708.1 s (run_node outer) | result.json `elapsed` / `elapsed_s`; inner-vs-outer gap 5.45 s = build/val setup |
| config_sha256 | 2300d4dbbde4aef7 | recomputed locally with runner's `sha256[:16]` → match |
| model_sha256 | 559d216c252b657e | recomputed locally → match |
| best.pth epoch | 32, best_mae 22.454483032226562 | torch.load on server |

## 2. Epoch-by-epoch delta vs champion (child − champion, val MAE)

| ep | champ | child | Δ | | ep | champ | child | Δ |
|---|---|---|---|---|---|---|---|---|
| 1 | 157.3232 | 135.4499 | **−21.8732** | | 17 | 23.1817 | 23.9185 | +0.7368 |
| 2 | 76.2375 | 70.9578 | **−5.2797** | | 18 | 22.9429 | 23.7778 | +0.8350 |
| 3 | 53.2973 | 51.6396 | **−1.6576** | | 19 | 22.7674 | 23.6358 | +0.8684 |
| 4 | 42.5154 | 42.0208 | **−0.4945** | | 20 | 22.6830 | 23.5164 | +0.8335 |
| 5 | 35.7347 | 35.4143 | **−0.3204** | | 21 | 22.4579 | 23.3560 | +0.8981 |
| 6 | 31.3749 | 31.0035 | **−0.3714** | | 22 | 22.2909 | 23.2258 | +0.9348 |
| 7 | 28.7491 | 28.0761 | **−0.6731** | | 23 | 22.0963 | 23.0852 | +0.9889 |
| 8 | 27.0146 | 26.3646 | **−0.6500** | | 24 | 21.8620 | 22.9310 | +1.0691 |
| 9 | 25.9413 | 25.4739 | **−0.4674** | | 25 | 21.7278 | 22.8138 | +1.0860 |
| 10 | 25.1930 | 24.8668 | **−0.3262** | | 26 | 21.6732 | 22.7234 | +1.0503 |
| 11 | 24.7454 | 24.6231 | **−0.1222** | | 27 | 21.6029 | 22.6347 | +1.0317 |
| 12 | 24.3498 | 24.4678 | **+0.1180** | | 28 | 21.5615 | 22.5436 | +0.9821 |
| 13 | 24.1115 | 24.2851 | +0.1735 | | 29 | 21.5198 | 22.4968 | +0.9770 |
| 14 | 23.7738 | 24.2221 | +0.4483 | | 30 | 21.4860 | 22.4829 | +0.9969 |
| 15 | 23.6001 | 24.1615 | +0.5614 | | 31 | 21.4686 | 22.4662 | +0.9976 |
| 16 | 23.4119 | 24.0347 | +0.6228 | | 32 | 21.4589 | 22.4545 | **+0.9956** |

- Child leads only ep1–11 (11/32 epochs), monotonically shrinking from −21.87 to −0.12; crossover at ep12.
- Tail deficit is stable, not noisy: ep25–32 mean +1.0147 (sd 0.0355, range +0.9770…+1.0860);
  ep28–32 mean +0.9899. No closing trend (ep29 +0.9770 → ep32 +0.9956).
- Child's 32-epoch best (22.4545) is only champion's ~ep21.5 quality: champion first reaches
  ≤22.4545 at ep22 (22.2909; ep21 = 22.4579) — i.e. the change costs ~10 epochs of progress.
- Train/val divergence: child train loss is lower nearly everywhere (ep1 Δ 50.305→23.296 = −27.01;
  ep32 2.7062→2.4700 = −0.2362; tail mean −0.252) while val MAE is worse → the subpixel branch fits
  the train density target better, generalizes worse. Val RMSE gap is larger than the MAE gap
  (ep32 +4.904: 84.675 vs 79.771; tail mean +4.311), i.e. the excess error sits in fewer, larger misses.
- Confound (data, not verdict): `top_up` draws 5120 RNG values inside `FineFuser.__init__` before
  `ExemplarEncoder`/`Condenser`/`DensityDecoder`/`GCA` are constructed, so with the shared seed the
  child's downstream modules start from different initial weights than the champion. The ep1–11 lead
  (and the ep12 flip) cannot be attributed to the mechanism on this run alone.

## 3. Bar-distance numbers

Champion = N0001_champion: best 21.458858 (ep32) = final 21.458858. Child: best 22.454483 (ep32) = final.

| Reference | Delta (child − ref) | Interpretation |
|---|---|---|
| 21.4589 (parent best) | **+0.9956** | child worse by ~1.0 MAE; 0.9956/0.30 = 3.3× the required improvement, in the wrong direction |
| 21.4589 (parent final, ep32) | **+0.9956** | identical (both runs' best = final epoch) |
| 21.1589 (pre-registered support line = parent − 0.30) | **+1.2956** | required ≤21.1589; miss is 4.3× the 0.30 bar |
| Child relative improvement needed to clear support line | 1.2956 MAE = 5.77% of 22.4545 | — |

H0013's falsifier "final val MAE is not at least 0.30 lower" evaluates to final 22.4545 vs 21.1589.

## 4. Wall-clock

| Run | elapsed_s | vs champion | vs N0006 |
|---|---|---|---|
| N0001_champion | 1704.5 | — | −66.5 s |
| N0007_h0013 (subpixel) | 1708.1 | **+3.6 s (+0.21%)** | −62.9 s |
| N0006_h0012 | 1771.0 | +66.5 s | — |

Per-epoch timing from TB event wall-clocks (`val/mae` inter-event gaps, includes train+val):
champion mean 52.7 s (52.4–52.9, 31 gaps; span 1633.0 s); child mean 52.8 s (52.6–53.1; span 1636.1 s).
Paired total +3.1 s over the 31 gaps ≈ **+0.10 s/epoch (+0.19%)**. Gap is flat across all 32 epochs
(no warmup outlier, no drift, child range 0.5 s). N0006 mean gap 54.7 s — its overhead is 17.8× the
subpixel run's. The subpixel cost (5120 params; ~5.3 M MACs/image grouped 3×3 on 24×24, ≈0.1% of the
frozen backbone) is not visible in wall-clock. Budget: 32 epochs at ~52.8 s + ~72 s setup ⇒ ~33 epochs
inside 1800 s.

## 5. Internal consistency

- result.json `best_mae` = 22.454483032226562 = TB last `val/mae` = info.json `best_metric` =
  best.pth `best_mae`. `best_epoch` 32 = TB argmin (index 31) = best.pth `epoch`.
- 32/32 epochs: TB 32 val points, `n_epochs_done` 32, `epoch` tag max 31, final global step 7295
  (228 steps/epoch), budget not hit, elapsed 1708.1 s < 1800 s. exec.log ends "Epoch 31/31 ... 22.454".
- Both checksums reproduce from the local files with the runner's algorithm (`sha256[:16]`).
- LR sequence (`lr-AdamW`, 32 pts, 0.001 → 3e-6) and `train/loss` step indices (145 pts, final 7249)
  are identical child-vs-champion → optimizer/schedule invariant honored.
- Known harness artifact (not node-specific): `train/mae` is NaN in both runs; `elapsed` (1702.651)
  and `elapsed_s` (1708.1) are two timers (fit vs run_node), difference 5.45 s, both honest.

## So what for the next cycle

- **Retire the subpixel mechanism, not just this config.** The deficit is stable at ~+1.0 MAE from
  ep25 on and the champion's own tail slope is steeper, so no plausible epoch extension closes it;
  if it ever returns it must book as `contradicts` on H0013 with a new falsifier plus the control in
  the next bullet (Hard Rule 11).
- **Make init a controlled variable for any add-a-module hypothesis.** Because new params consume the
  shared RNG stream before downstream head modules are built, every "child adds a block" run starts
  from a different shared-head init than its parent; the child's ep1–11 lead here (−21.87 MAE at ep1)
  is confounded by that. Next coding agent: initialize new modules from a private `torch.Generator`
  after all pre-existing modules are constructed (or record an init-only control run) so architecture
  effects are separable from init effects.
- **Wall-clock is not the binding constraint on this card, MAE is.** Subpixel cost +3.6 s vs N0006's
  +66.5 s; ~33 epochs fit the 1800 s budget. Future bars of 0.30 MAE are 1.4% of the 21.459 baseline
  while observed solo-test misses are +0.036 (H0012) and +1.296 (H0013, vs bar overlap) — the
  `repro_run` champion noise-floor diagnostic should be read before any new 0.3-scale bar is trusted.

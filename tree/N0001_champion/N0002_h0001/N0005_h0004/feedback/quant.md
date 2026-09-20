# N0005_h0004 - Quantitative feedback (H0004 `use_simbank`)

**Verdict: REFUTE.** The pre-registered falsifier is missed. Best EMA val MAE **48.1201 @ep14**, final **48.1849 @ep32**, against the bar `<= 22.26`. Short by **+25.8601** (best) and **+25.9249** (final). No confirm path exists at any logged epoch.

**Timeout protocol fact, separate from the verdict:** `budget_hit=true` forces status `timeout` (`run_node.py:108-109`), yet all 32 scheduled epochs still ran end-of-epoch validation (32 val events; `epochs=32`, `n_epochs_done=32`). The final epoch's training loop was cut at 109/228 batches by `_BudgetStop`, then validation logged 48.185. Best epoch was ep14, 18 epochs before the stop, so the timeout cannot change the verdict.

## 1. Sources actually read

| Source | Path | Used for |
|---|---|---|
| child result.json | `tree/N0001_champion/N0002_h0001/N0005_h0004/result.json` | best_mae 48.12007522583008, best_epoch 14, epochs 32, n_epochs_done 32, elapsed 1809.6606, elapsed_s 1815.2, budget_hit true, shas |
| parent result.json | `tree/N0001_champion/N0002_h0001/result.json` | best_mae 22.564146041870117, best_epoch 30, elapsed 1749.5316, elapsed_s 1755.0, budget_hit false, shas |
| N0004 result.json | `tree/N0001_champion/N0002_h0001/N0004_h0003/result.json` | best_mae 25.566089630126953, best_epoch 13, elapsed_s 1773.3, budget_hit false, shas |
| child info.json | `.../N0005_h0004/info.json` | status `timeout`, best_metric 48.12007522583008, train_seconds 1815.2, epochs 32 |
| parent info.json | `.../N0002_h0001/info.json` | status `done` |
| N0004 info.json | `.../N0004_h0003/info.json` | status `done` |
| child TB events | `/data/cac/tree/N0001_champion/N0002_h0001/N0005_h0004/run/latest/tb/t/events.out.tfevents.1789887370.bwdqkrimegiarobw-snow-59d754d49d-f7svh.11590.0` | full 32-point val/mae, val/rmse, train/loss_epoch |
| parent TB events | `/data/cac/tree/N0001_champion/N0002_h0001/run/latest/tb/t/events.out.tfevents.1789881056.bwdqkrimegiarobw-snow-59d754d49d-f7svh.1670.0` | full 32-point val/mae, val/rmse, train/loss_epoch |
| N0004 TB events | `/data/cac/tree/N0001_champion/N0002_h0001/N0004_h0003/run/latest/tb/t/events.out.tfevents.1789885586.bwdqkrimegiarobw-snow-59d754d49d-f7svh.9960.0` | 32-point val/mae for the sibling comparison |
| chain_run.log | `/data/cac/tree/N0001_champion/N0002_h0001/N0005_h0004/chain_run.log` | budget stop line, fit-stop line, final val 48.185, trailing result JSON |
| child hparams.yaml | `.../run/latest/tb/t/hparams.yaml` | seed 20260830, augment false, epochs 32, use_simbank true |
| checksums | sha256sum on local and server copies | child config e7a9d269d105a9fa, model 319a55e6dc58918a; parent 4d4c9baec459f657 / 611be2a9d70768b5; N0004 769ddfa05d25dfa9 / 93c221209869fe69 |

Curves extracted with `tensorboard.backend.event_processing.event_accumulator.EventAccumulator(path, size_guidance={"scalars": 0}, purge_orphaned_data=False)`, so all 32 points are present (no reservoir downsampling).

## 2. Headline table

| Field | N0005_h0004 (child) | N0002_h0001 (parent) | N0004_h0003 (sibling) |
|---|---:|---:|---:|
| Best val MAE | 48.120075 | 22.564146 | 25.566090 |
| Best epoch | 14 | 30 | 13 |
| Final val MAE (ep32) | 48.184944 | 22.570467 | 26.166784 |
| Epochs scheduled | 32 | 32 | 32 |
| n_epochs_done | 32 | 32 | 32 |
| Val events logged | 32/32 | 32/32 | 32/32 |
| Wall seconds (result `elapsed_s`) | 1815.2 | 1755.0 | 1773.3 |
| Fit seconds (result `elapsed`) | 1809.6606 | 1749.5316 | 1767.6270 |
| `budget_hit` | true | false | false |
| Status (info.json) | **timeout** | done | done |
| `config_sha256` | e7a9d269d105a9fa | 4d4c9baec459f657 | 769ddfa05d25dfa9 |
| `model_sha256` | 319a55e6dc58918a | 611be2a9d70768b5 | 93c221209869fe69 |

Best rows come from result.json (child, parent, N0004) and match the TB minima exactly at the same steps. Final rows are the ep32 `val/mae` events (no result.json field stores final MAE). Config delta child vs parent is exactly one added line, `use_simbank = true` (`diff` exit 1, single `27a28` hunk); model.py differs by 54 added/removed lines (SimBank class plus wiring).

## 3. Full child and parent curves (val/mae, val/rmse), N0004 val/mae

Step 227 + 228*(ep-1) for ep1..31 in all three runs. Ep32 steps: child 7176 (stopped early), parent 7295, N0004 7295.

| ep | TB step | child val/mae | child val/rmse | parent val/mae | parent val/rmse | N0004 val/mae |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 227 | 212.0775 | 228.3391 | 153.2874 | 174.3924 | 152.7835 |
| 2 | 455 | 140.0689 | 166.2508 | 70.0534 | 116.8055 | 69.2195 |
| 3 | 683 | 146.6994 | 171.3909 | 48.7882 | 106.9041 | 49.6918 |
| 4 | 911 | 108.1089 | 143.0917 | 39.1216 | 102.9298 | 40.8047 |
| 5 | 1139 | 69.3205 | 124.6808 | 33.5010 | 100.0118 | 35.1977 |
| 6 | 1367 | 55.5274 | 123.6250 | 30.2952 | 98.0300 | 31.3338 |
| 7 | 1595 | 53.4358 | 123.9916 | 28.4055 | 97.2657 | 28.8931 |
| 8 | 1823 | 50.8420 | 124.8637 | 27.2659 | 96.8390 | 27.4816 |
| 9 | 2051 | 49.0509 | 126.0892 | 26.2845 | 96.3401 | 26.6618 |
| 10 | 2279 | 48.5324 | 126.6988 | 25.4092 | 95.3331 | 25.9875 |
| 11 | 2507 | 48.2737 | 127.2247 | 24.6751 | 94.0627 | 25.6881 |
| 12 | 2735 | 48.1399 | 127.9124 | 24.2563 | 93.2005 | 25.5786 |
| 13 | 2963 | 48.1202 | 128.4674 | 23.9073 | 92.3794 | 25.5661 |
| 14 | 3191 | **48.1201** | 128.6037 | 23.5598 | 91.4805 | 25.5921 |
| 15 | 3419 | 48.1314 | 128.7645 | 23.4425 | 91.0531 | 25.6438 |
| 16 | 3647 | 48.1493 | 128.8987 | 23.2482 | 90.5843 | 25.7179 |
| 17 | 3875 | 48.1565 | 128.9293 | 23.0597 | 90.1668 | 25.7898 |
| 18 | 4103 | 48.1486 | 128.8124 | 22.9224 | 89.9383 | 25.8624 |
| 19 | 4331 | 48.1511 | 128.7641 | 22.9487 | 89.7513 | 25.9220 |
| 20 | 4559 | 48.1567 | 128.7432 | 22.9341 | 89.6870 | 25.9757 |
| 21 | 4787 | 48.1597 | 128.7292 | 22.9015 | 89.5291 | 26.0180 |
| 22 | 5015 | 48.1592 | 128.6849 | 22.8883 | 89.4606 | 26.0532 |
| 23 | 5243 | 48.1556 | 128.5748 | 22.7773 | 89.1640 | 26.0818 |
| 24 | 5471 | 48.1611 | 128.5569 | 22.7200 | 89.1040 | 26.1028 |
| 25 | 5699 | 48.1671 | 128.5620 | 22.6848 | 88.9863 | 26.1212 |
| 26 | 5927 | 48.1679 | 128.5100 | 22.6515 | 88.9433 | 26.1298 |
| 27 | 6155 | 48.1689 | 128.4819 | 22.6285 | 88.9668 | 26.1365 |
| 28 | 6383 | 48.1717 | 128.4809 | 22.5923 | 88.9234 | 26.1453 |
| 29 | 6611 | 48.1796 | 128.5471 | 22.5659 | 88.9828 | 26.1528 |
| 30 | 6839 | 48.1823 | 128.5532 | **22.5641** | 89.0939 | 26.1581 |
| 31 | 7067 | 48.1845 | 128.5583 | 22.5733 | 89.1886 | 26.1628 |
| 32 | c7176 / p7295 / n7295 | 48.1849 | 128.5564 | 22.5705 | 89.2400 | 26.1668 |

### Train-side curve (overfit check)

| ep | child train/loss_epoch | parent train/loss_epoch |
|---:|---:|---:|
| 1 | 20.5252 | 38.0236 |
| 2 | 14.6132 | 8.6532 |
| 3 | 14.4732 | 8.4087 |
| 4 | 14.5234 | 7.7869 |
| 5 | 14.3327 | 6.9885 |
| 6 | 14.5434 | 6.8799 |
| 7 | 14.6597 | 6.8469 |
| 8 | 14.2070 | 6.5521 |
| 9 | 14.2597 | 6.1468 |
| 10 | 14.1750 | 5.4920 |
| 11 | 14.2083 | 5.4525 |
| 12 | 14.2020 | 5.5983 |
| 13 | 14.0027 | 5.2261 |
| 14 | 14.1850 | 5.0677 |
| 15 | 14.1099 | 4.6221 |
| 16 | 13.9905 | 4.5107 |
| 17 | 14.0406 | 4.5179 |
| 18 | 14.0842 | 4.1466 |
| 19 | 14.1329 | 4.1724 |
| 20 | 14.0596 | 3.9857 |
| 21 | 14.0622 | 3.7517 |
| 22 | 14.0398 | 3.7048 |
| 23 | 14.0590 | 3.4890 |
| 24 | 14.0551 | 3.5126 |
| 25 | 14.0042 | 3.2047 |
| 26 | 14.0147 | 3.1010 |
| 27 | 14.0330 | 3.0211 |
| 28 | 14.0394 | 2.8041 |
| 29 | 14.0281 | 2.7641 |
| 30 | 14.0142 | 2.7040 |
| 31 | 14.0225 | 2.6526 |
| 32 | 13.8132 | 2.6183 |

## 4. chain_run.log evidence

Verbatim lines from the child log (the run's own stdout):

```
[budget] exceeded 1800s — stopping early
`Trainer.fit` stopped: `max_epochs=32` reached.
Epoch 31/31 ━━━━━━━╸         109/228 0:00:32 •        4.84it/s v_num: t val/mae:
                                     0:00:25                   48.185
```

The trailing JSON block in the same log is byte-for-byte the child `result.json` contents quoted in §1. The progress line fixes two facts: the last epoch display is `31/31` (0-indexed 32nd) and its training bar stopped at `109/228`, then val/mae 48.185 was printed.

## 5. Quantifications

1. **vs live parent (22.564146): +25.555929 MAE worse** (48.120075 - 22.564146). Relative: **+113.26%**. At every one of the 32 epochs the parent's val/mae is lower (per-epoch gap range in point 6).
2. **vs bar 22.26: best misses by +25.860075, final misses by +25.924944.** Child best is 2.1617x the bar. This is ~1,278x the ~0.02 same-seed paired-contrast precision stated in AGENTS §6.
3. **vs N0004 (25.566090): +22.553985 MAE worse** (48.120075 - 25.566090), **+88.22%** relative. The child is worse than both its parent and its sibling N0004.
4. **Budget overrun: elapsed_s 1815.2 vs 1800 = +15.2 s (+0.84%).** The internal fit clock gives 1809.6606 - 1800 = **+9.661 s**; the two differ because `run_node.py:94` measures t0 to return (incl. smoke/setup) while `runner.py:129` measures the fit only.
5. **Wall vs siblings:** +60.2 s vs parent (1815.2 - 1755.0, +3.43%); +41.9 s vs N0004 (1815.2 - 1773.3, +2.36%).
6. **Paired per-epoch gap child minus parent runs +22.766 (ep9, smallest) to +97.911 (ep3).** The parent's val/mae is lower than the child's at all 32 logged epochs.
7. **Dominance facts.** Parent ep4 = 39.121643, already below the child's best-ever 48.120075, so from ep4 on the parent beats the child's best. Parent ep2 train loss 8.653178 is below the child's all-time train/loss minimum 13.813208; the child never fit the training set as well as the parent did by its second epoch. Final ratio of train losses: 13.813208 / 2.618342 = 5.28x. The child underfit, it did not overfit.
8. **Plateau.** Child ep13..32 val/mae spans 48.120152..48.184944 (spread 0.064869). Parent ep29..32 spans 22.564146..22.573269 (spread 0.009123). The child's last 20 epochs move less than 0.07 MAE while sitting 25.5 MAE above the parent.
9. **RMSE.** Child minimum 123.625038 @ep6, final 128.556351. Parent minimum 88.923416 @ep28, final 89.239967. The child RMSE rose ~4.93 from its own minimum while its MAE fell to the plateau, and parent ep4 RMSE (102.929771) is already below the child's all-time minimum.
10. **Best epoch vs timeout.** Best (ep14, 48.1201) and second-best (ep13, 48.1202) are 18 epochs before the ep32 budget stop; the stopped-epoch val (48.1849) is +0.0649 from the best. The timeout has no bearing on the falsifier miss.

## 6. Timeout protocol fact

- `_BudgetStop` (`src/cac/engine/runner.py:134-146`) checks at each `on_train_batch_end` whether its clock exceeds 1800 s, then sets `trainer.should_stop=True`, sets `hit=True`, and prints the stop line.
- `run_node.py:108-109` maps `budget_hit=true` to `tree.set_status(node, "timeout")` regardless of epoch completion. That is why info.json says `timeout` while result.json says `epochs=32, n_epochs_done=32`.
- Observed completion: 32/32 val events in TB, final epoch bar at 109/228 batches (chain_run.log), child's last logged TB step 7176 vs parent 7295. So all 32 epochs produced validation, and the last epoch trained roughly 48% of its batches before the loop stopped.
- The timeout is a budget label, not a crashed or failed run (no traceback in chain_run.log; the trailing JSON has `"code": "ok"`).

## 7. Verdict detail

Pre-registered rule (idea.md §6, booking header): confirm iff final EMA val MAE `<= 22.26`; otherwise REFUTE. Observed final 48.184944, best 48.120075. Both fail; the best is 2.16x the bar value (48.120075 / 22.26). Same-seed paired protocol held: seed 20260830 (hparams.yaml), augment false, 32ep/1800s, single run, no re-runs.

Scope of the negative (per idea.md §6): this falsifies the specific claim that adding a per-part 1/8 ROI-token similarity bank (`use_simbank`, roi_size 5) on top of the simprior-bearing parent recovers small-object evidence beyond the pooled 1/16 prior. The broader similarity-readout family stays live as stated there. Refuted variants are not retried silently (AGENTS §5 rule 11); any revisit books as `contradicts` on H0004 with a new falsifier.

## 8. Provenance checks performed

- sha256 of child config.toml / model.py computed on the local tree and on `/data/cac/...` server copies: both match result.json (`e7a9d269d105a9fa`, `319a55e6dc58918a`). Same check passed for parent and N0004.
- TB best values equal result.json values at equal steps: child 48.120075 at step 3191 (ep14), parent 22.564146 at step 6839 (ep30), N0004 25.566090 at step 2963 (ep13).
- `chain_run.log` final JSON equals child result.json field-for-field.
- All arithmetic in §5 recomputed from the raw event values listed in §3 and the result.json fields in §2.

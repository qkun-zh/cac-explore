# N0003_h0002 — Quantitative feedback (H0002 `use_gca_cal`)

**Verdict: REFUTE.** Pre-registered falsifier was *final EMA val MAE ≤ 22.99*;
child best **23.2088** misses by **+0.2188**. Paired improvement over the direct
parent N0001_champion (23.2932 @ep26) is only **0.0844**, vs the **0.30** bar.
All numbers below were read from `result.json`, `info.json`, the TensorBoard
events files, `config.toml`, `best.pth`, and `chain_run.log` actually fetched
this session (sources at bottom).

## 1. Parent-vs-child table

| Quantity | Parent `N0001_champion` | Child `N0003_h0002` | Delta (child − parent) |
|---|---|---|---|
| best val MAE | 23.293153762817383 | 23.20878791809082 | **−0.084365844726563** (improve) |
| best epoch | 26 (of 32) | 32 (of 32, final) | — |
| final-epoch val MAE | 23.33643341064453 (ep32) | 23.20878791809082 (ep32) | −0.1276454919 |
| epochs run | 32/32 | 32/32 | — |
| elapsed | 1694.12 s (`elapsed`); 1699.4 s (`elapsed_s`) | 1698.95 s (`elapsed`); 1704.2 s (`elapsed_s`) | — |
| budget_hit | false | false | — |
| config_sha256 | 679f37c26bc72103 | daf2b13a125900ad | — |
| model_sha256 | eaf2a8b93c280cba | 633e362a482b3420 | — |
| status | done | done | — |
| tested_hypotheses | — | ["H0002"] | — |

Live cross-reference (not the direct parent, for context only): new live best
`N0002_h0001` = 22.564146041870117 @ep30. Child is **0.6446 worse** than that
reference. Re-instantiated 0.30 bar off 22.5641 (AGENTS §6) = 22.2641; child
misses that by **+0.9447**. The node's own booked bar remains 22.99.

## 2. Full child val/MAE curve (32 epochs, EMA eval)

Source tag `val/mae`, 32 scalars, steps 227…7295 (one per epoch).

| ep | val MAE | ep | val MAE |
|---|---|---|---|
| 1 | 164.203201 | 17 | 24.317142 |
| 2 | 79.014961 | 18 | 24.098745 |
| 3 | 56.607067 | 19 | 23.869884 |
| 4 | 45.004013 | 20 | 23.747503 |
| 5 | 38.526306 | 21 | 23.575665 |
| 6 | 34.109222 | 22 | 23.452490 |
| 7 | 31.061632 | 23 | 23.323265 |
| 8 | 28.814993 | 24 | 23.259789 |
| 9 | 27.166063 | 25 | 23.216766 |
| 10 | 26.017748 | 26 | 23.227020 |
| 11 | 25.422514 | 27 | 23.224014 |
| 12 | 25.226351 | 28 | 23.215664 |
| 13 | 25.271629 | 29 | 23.221870 |
| 14 | 25.072756 | 30 | 23.216501 |
| 15 | 24.816734 | 31 | 23.213377 |
| 16 | 24.579475 | 32 | **23.208788** (best) |

Parent series for the paired contrast (read from
`/data/cac/tree/N0001_champion/run/latest/tb/t/events*`, direct path because
`curve.py N0001_champion` is path-broken for the root per STATE): ep1 146.9976,
ep2 69.0238, ep3 49.4376, ep4 39.5353, ep5 34.1235, ep6 30.6235, ep7 28.5269,
ep8 27.1416, ep9 26.0856, ep10 25.3461, ep11 24.7792, ep12 24.3675, ep13
24.2335, ep14 24.0380, ep15 23.8607, ep16 23.9168, ep17 23.8016, ep18 23.7184,
ep19 23.6163, ep20 23.5276, ep21 23.5090, ep22 23.4600, ep23 23.3879, ep24
23.3307, ep25 23.3257, ep26 **23.2932**, ep27 23.3194, ep28 23.3075, ep29
23.3070, ep30 23.3221, ep31 23.3341, ep32 23.3364.

## 3. Distance to the 22.99 bar — first / last / closest

The bar is **never approached**. Minimum over all 32 epochs = 23.2088 (ep32),
i.e. still **+0.2188 above 22.99** at the single best point. The only epochs
below 23.30 are ep24–ep32; before ep24 the curve is ≥ 23.2598, so early epochs
(ep1 = 164.20, ep10 = 26.02) are 1.2–141 MAE above the bar. There is no epoch,
including the best, within 0.21 of the registered threshold.

- Distance from best epoch to bar: `23.208788 − 22.99 = +0.218788`.
- Distance from final epoch to bar: same value (best == final).
- Required paired drop: child delivered 0.0844 of the 0.30 needed
  (`0.0844 / 0.30 = 28.1%` of the required margin).

## 4. Plateau reading

Both runs plateau hard over the last third:

- Child ep24–ep32 band: max 23.259789, min 23.208788 → **range 0.0510**.
- Parent ep26–ep32 band: max 23.336433, min 23.293154 → **range 0.0433**.
- The child's entire late gain (ep25 23.2168 → ep32 23.2088) is **0.0080**;
  its last-4-epoch trend (ep28 23.2157 → ep32 23.2088) is **0.0069**.
- Child beats parent's best (23.2932) only from ep24 onward, and only by
  0.0844 at the end — i.e. the intervention's whole effect is inside a plateau
  band comparable to run-to-run noise, not a level shift.

## 5. Best-at-final-epoch (ep32): unresolved trend, quantified

Yes — the best checkpoint is the final epoch, so no early-stop defines the
result and a strictly non-increasing tail is still visible. The slope is far
too shallow to matter:

- Last 7 epochs (ep25→ep32): `23.216766 − 23.208788 = 0.007978` over 7 epochs
  ≈ **0.00114 MAE/epoch**.
- Even crediting the entire ep24→ep32 drop (`23.259789 − 23.208788 = 0.051001`
  over 8 epochs ≈ 0.00638/epoch) and linearly extrapolating, closing the
  remaining 0.2188 to the bar needs **~34–192 more epochs** depending on slope
  used — far beyond the fixed 32ep/1800 s protocol. Cosine LR is also at its
  floor (η_min 1e-06), so the observed slope is the *optimistic* upper bound.
- Conclusion: "still descending" is real but cannot reach 22.99 under this
  protocol; it does not rescue the hypothesis. Verdict stays REFUTE.

## 6. Learned-parameter criterion (from child `best.pth`)

Stored EMA-averaged values (`{"epoch":32, "best_mae":23.20878791809082}`):

| param | init | learned (EMA) | null box (idea.md §6) | null? |
|---|---|---|---|---|
| `gca.s_pos` | 0.02 | **0.008398002944886684** | `|s_pos−0.02|/0.02 < 0.10` → actual 0.580 | **fail (moved)** |
| `gca.s_lin` | 0.0 | **0.02147696353495121** | `|s_lin| < 1e-3` → actual 0.0215 | **fail (moved)** |
| `gca.b_cal` | 0.0 | **−0.11962026357650757** | `|b_cal| < 0.5` → actual 0.1196 | pass |

This is the **moved-but-missed** branch of idea.md §6, not the "coupling was
already optimal" null: the optimizer shrank the softplus branch by ~58% and
added a nonzero signed linear term / negative offset, yet MAE improved by only
0.0844. Under AGENTS rule 1 (no outcome shopping) this is still DISPROVED on
the registered bar; the movement is evidence for a follow-up card, not a rescue.

## 7. Numeric anomalies (all observed, none change the verdict)

1. **`train/mae` scalar is `nan` for all 32 epochs** — in *both* child and
   parent event files. It is a Logging-side artifact (torchmetrics
   `MeanMetric.compute` before `update`, same warning present in
   `run_smoke.log`); `train/loss_epoch` and `val/mae` are finite throughout, so
   training is not NaN. Flag for the runner/logging, not a model failure.
2. **Epoch-1 trajectory diverges from the parent despite "init-equivalent
   forward"** (idea.md §4). Child ep1 `val/mae` = 164.2032 vs parent 146.9976
   (**+17.21 worse**); child first logged `train/loss` (step 49) = 631.64 vs
   parent 3619.43, and `train/loss_epoch` ep1 = 20.93 vs parent 91.30. The
   init math is equal in expectation, but the 3 new params are trainable from
   step 0 and move within epoch 1, so the child does not track the parent early.
   The claim "worst case ≈ the parent" holds only asymptotically, not per-epoch.
3. **`result.json` carries two elapsed fields that disagree**:
   `"elapsed": 1698.9452395439148` vs `"elapsed_s": 1704.2` (~5.25 s apart).
   Same structure in the parent (1694.12 vs 1699.4). Cosmetic; both < 1800 s
   and `budget_hit=false`.
4. Steps for `val/mae` are identically 227…7295 in parent and child (session
   step counter is per-epoch, not wall-clock) — expected, noted so the step
   equality is not misread as a shared event file.

## 8. Bottom line

- Paired contrast vs direct parent: **0.0844 improvement < 0.30 bar → REFUTE**.
- Absolute registered bar: best **23.2088 > 22.99 by 0.2188 → REFUTE**.
- Parameters moved (s_pos −58%, s_lin = 0.0215) yet MAE barely moved: mechanism
  fired, effect negligible — a clean negative, not a null, not a rescue.
- Best-at-final is a real residual slope (~0.0011 MAE/epoch) but cannot close
  0.2188 within the fixed protocol; no basis to re-run or extend.

## Sources (read this session)

- `/home/qkun/cac/tree/N0001_champion/N0003_h0002/result.json`
- `/home/qkun/cac/tree/N0001_champion/result.json` (parent)
- `/home/qkun/cac/tree/N0001_champion/N0002_h0001/result.json` (live best)
- `/home/qkun/cac/tree/N0001_champion/N0003_h0002/idea.md` + `config.toml` + `info.json`
- `ssh cac-server`: `curve.py N0003_h0002` → `best=23.2088@ep32`
- EventAccumulator dump of
  `/data/cac/tree/N0001_champion/N0003_h0002/run/latest/tb/t/events*` (child, 32 val/mae)
  and `/data/cac/tree/N0001_champion/run/latest/tb/t/events*` (parent, 32 val/mae)
- `torch.load` of child `/data/cac/tree/N0001_champion/N0003_h0002/run/latest/best.pth`
- child `/data/cac/tree/N0001_champion/N0003_h0002/chain_run.log` (last line: val/mae 23.209)

# Quantitative feedback — N0011_h0010 (H0010 SOLO, per-exemplar scalar reliability gate)

## 1. Final table (same-seed paired contrast, canonical protocol)

| Node | Test | best val MAE | @ep | val RMSE @best-MAE ep | train loss_epoch @ep32 |
|---|---|---|---|---|---|
| N0001_champion (parent, canonical) | — | 23.2932 | 26 | 90.8030 (best RMSE 90.6424 @ep24) | 2.8345 |
| N0011_h0010 (child, H0010 solo) | `use_exemplar_gate=true` | 23.4581 | 25 | 91.6393 (best RMSE 91.6063 @ep26) | 2.7493 |
| **Delta (child − parent)** | | **+0.1649 (worse)** | −1 ep | **+0.9639 RMSE worse (best-vs-best)** | **−0.0852 (fits train better)** |

Sources: `tree/N0001_champion/N0011_h0010/result.json` (best_mae 23.458118438720703 @ep25),
`tree/N0001_champion/result.json` (best_mae 23.293153762817383 @ep26), TB `val/mae`, `val/rmse`,
`train/loss_epoch` per-epoch scalars (local reads; parent curve = canonical file
`events...1789812726...`, min 23.2932 @ep26 — NOT the stale v1 file `events...1789796237...`,
which ends 21.4589 and is historical record only).

## 2. Bar distances

- Pre-registered falsifier (anchor rule, live-parent semantics): support requires
  ≥0.30 below the live parent → bar = 23.2932 − 0.30 = **22.9932**.
- Child 23.4581 vs bar: **misses by +0.4649** (needs −0.4649 swing). Decisive miss.
- Child vs parent: **+0.1649 worse**, i.e. ~8× the same-seed paired-contrast precision
  (~0.02) — a real degradation, not harness noise (level noise ±1 MAE applies only to
  cross-seed absolute-level claims, not this paired contrast).
- Verdict: **contradicts** (ledger: contradicts w=1.0, H0010 conf 0.3200).

## 3. Epoch-by-epoch delta vs parent, val MAE (child − parent, ep1..32)

| ep | parent | child | Δ | ep | parent | child | Δ |
|---|---|---|---|---|---|---|---|
| 1 | 146.9976 | 183.1119 | +36.1143 | 17 | 23.8016 | 23.9086 | +0.1070 |
| 2 | 69.0238 | 74.1366 | +5.1128 | 18 | 23.7184 | 23.8444 | +0.1260 |
| 3 | 49.4376 | 49.6158 | +0.1782 | 19 | 23.6163 | 23.8540 | +0.2377 |
| 4 | 39.5353 | 39.6931 | +0.1578 | 20 | 23.5276 | 23.7066 | +0.1790 |
| 5 | 34.1235 | 34.4823 | +0.3588 | 21 | 23.5090 | 23.6123 | +0.1033 |
| 6 | 30.6235 | 31.0918 | +0.4683 | 22 | 23.4600 | 23.5419 | +0.0819 |
| 7 | 28.5269 | 28.6587 | +0.1318 | 23 | 23.3879 | 23.5074 | +0.1195 |
| 8 | 27.1416 | 27.0951 | −0.0465 | 24 | 23.3307 | 23.4874 | +0.1567 |
| 9 | 26.0856 | 25.9468 | −0.1388 | 25 | 23.3257 | 23.4581 | +0.1324 |
| 10 | 25.3461 | 25.3591 | +0.0130 | 26 | 23.2932 | 23.4654 | +0.1722 |
| 11 | 24.7792 | 24.8665 | +0.0873 | 27 | 23.3194 | 23.4818 | +0.1624 |
| 12 | 24.3675 | 24.5139 | +0.1464 | 28 | 23.3075 | 23.4887 | +0.1812 |
| 13 | 24.2335 | 24.3403 | +0.1068 | 29 | 23.3070 | 23.5097 | +0.2027 |
| 14 | 24.0380 | 24.1995 | +0.1615 | 30 | 23.3221 | 23.5334 | +0.2113 |
| 15 | 23.8607 | 24.0462 | +0.1855 | 31 | 23.3341 | 23.5533 | +0.2192 |
| 16 | 23.9168 | 24.0345 | +0.1177 | 32 | 23.3364 | 23.5739 | +0.2375 |

Reading: child worse in **30/32** epochs (better only ep8 −0.0465, ep9 −0.1388).
Mean Δ ep8–32 = +0.1305; mean Δ ep20–32 = +0.1661. Early transient (ep1 +36.11,
ep2 +5.11) washes out by ep3; from ep10 on every Δ is ≥0 except none — the gap is
persistent, narrow (+0.08…+0.24), and widens slightly into the tail (ep29–32 all
≥+0.20). Both curves' best epochs sit adjacent (25 vs 26); the child never leads at
any plateau epoch. No crossing, no late recovery — uniform small harm, not a
schedule artifact.

## 4. Pre-registered family comparison: exemplar-gating variants

| Node | Gate form | best MAE | Δ vs parent 23.2932 |
|---|---|---|---|
| N0009_h0015 (H0015) | per-exemplar **per-channel** MLP mask (256-dim vector per exemplar) | 23.6818 @ep23 | **+0.3886** |
| N0011_h0010 (H0010) | per-exemplar **scalar** gate (Linear(256,1), sigmoid) | 23.4581 @ep25 | **+0.1649** |

Both gating forms harm under the canonical protocol; the scalar gate harms
**2.36× less** (0.3886/0.1649 = 2.357) than the per-channel gate. Direction is
consistent across the family (gating exemplar tokens before the condenser never
helps: 0/2), while magnitude scales with gate capacity (1 scalar vs 256 channel
weights per exemplar). The clean unconfounded H0010 solo (+0.165) is therefore the
family's *best* exemplar-gating result and still a decisive contradict — the
failure is the gating mechanism itself, not the extra parameters of the N0009
variant. (Context: non-gating siblings under the same protocol help — N0008
22.9697 Δ−0.3235, N0010 22.8831 Δ−0.4101 — so the harness can and does register
gains; the gate family is specifically negative.)

## 5. Wall-clock

- Child: elapsed_s **1702.7**, 32/32 epochs, budget_hit **false**; headroom 1800 − 1702.7 = **97.3 s**.
- Parent canonical: elapsed_s 1699.4 — child cost essentially identical (+3.3 s; the
  single Linear(256,1) is negligible compute, as expected).
- Full 32-epoch run, no `_BudgetStop`, no timeout marking — complete observation.

## 6. Internal consistency (all check)

1. `result.json` best_mae 23.458118438720703 @ep25 == TB `val/mae` minimum (23.4581 @ep25) == `info.json` best_metric. ✓
2. `info.json`: status done, 32 epochs, tested_hypotheses [H0010], train_seconds 1702.7 == result elapsed_s. ✓
3. `config.toml`: `use_exemplar_gate=true`, seed 20260830, deterministic=true, augment=false — canonical protocol pins present. ✓
4. `model.py`: gate is `nn.Linear(256,1)`, zero weight init, bias +3.0 → sigmoid(3.0) = 0.9526,
   near-identity at init: the ep1 transient (+36.11) is init/optimization noise shared-shape
   with the parent's own ep1 (147.00), converged out by ep3 (+0.18). The persistent gap is learned, not an init shock. ✓
5. Overfit signature, not underfit: child final train loss 2.7493 < parent 2.8345 (−0.0852)
   while val MAE/RMSE both worse — the gate's capacity memorizes train, degrades val. ✓
6. RMSE agrees in direction: child best 91.6063 vs parent best 90.6424 (+0.9639); no MAE/RMSE conflict. ✓
7. Stale-file guard: the co-located v1 event file (ends 21.4589) was excluded; every parent
   number above comes from the canonical file (min 23.2932 @ep26 == `result.json`). ✓

## 7. Takeaways

- H0010 SOLO cleanly refuted: +0.1649 vs parent, bar 22.9932 missed by +0.4649 (contradicts w=1.0).
- Family result: scalar (+0.165) harms 2.36× less than per-channel (+0.389) — gating the exemplar stream before the condenser is net-harmful at any granularity tried.
- Run is complete and honest (32/32 ep, 1702.7 s, no budget hit); all artifacts cross-check — no re-run warranted.

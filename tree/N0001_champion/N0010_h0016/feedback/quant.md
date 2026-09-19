# Quantitative feedback — N0010_h0016 (H0016 SOLO, DAVE-lite verify-and-suppress)

- Node: `tree/N0001_champion/N0010_h0016`, H0016 solo (`use_verify_mask`), seed 20260830.
- Parent: canonical N0001_champion, best_mae **23.2932 @ep26** (protocol noaug+EMA+seeded+deterministic).
- Child: best_mae **22.8831 @ep32**, 32/32 epochs, budget_hit=false, elapsed_s 1751.4.
- Delta vs parent: **-0.4101**. Anchor-rule support line 22.9932 (parent − 0.30); child clears it by **0.1101**.
- Ledger (recorded fact, not re-verdict): `supports w=1.0`, H0016 conf 0.600 — the lab's second support, healthier margin than H0014's 0.024.
- Sources read: child `result.json`, `info.json`, `config.toml`, `model.py`, `idea.md`;
  child TB `run/latest/tb/t/events.out.tfevents.1789821393...31692.0` (val/mae, 32 pts);
  parent canonical TB `tree/N0001_champion/run/latest/tb/t/events.out.tfevents.1789812726...23038.0`
  (NOT the stale v1 file `...1789796237...`, whose 21.459 curve is historical record only);
  `memory/hypotheses.jsonl` H0016 rows; server best.pth headers via cac-server.
- Scope: numbers only. No verdict.

## 1. Final table

| Item | Child N0010_h0016 | Parent N0001_champion (canonical) |
|---|---|---|
| best_mae | 22.88313675 → **22.8831 @ep32** | 23.29315376 → **23.2932 @ep26** |
| best_epoch | 32 (best = last) | 26 |
| last val (ep32) | 22.883137 (= best) | 23.336433 (+0.0433 above its best) |
| epochs / n_epochs_done | 32 / 32 | 32 / 32 |
| budget_hit | false | false |
| elapsed_s | 1751.4 (elapsed field 1745.88; headroom 48.6 s under τ_max=1800) | 1699.4 |
| config_sha256 | `e8f99d7de3a34f5f` — matches local sha256 prefix of node config.toml | `679f37c26bc72103` — matches |
| model_sha256 | `2428f6e78ca06400` — matches local sha256 prefix of node model.py | `eaf2a8b93c280cba` — matches |
| best.pth cross-check (server) | epoch=32, best_mae=22.883136749267578 — **bit-exact vs result.json**; 125,383,769 B, sha256 `2318b326…` | epoch=26, best_mae=23.293153762817383 — exact vs result.json |
| config delta vs parent | exactly one added line: `use_verify_mask = true` (diff confirmed) | — |
| TB min check | TB val/mae min 22.883137 @step7295 (ep32) = result.json best | TB val/mae min 23.293154 @ep26 = result.json best |
| seed / deterministic / augment | 20260830 / true / false (both nodes identical) | same |

Exact delta: 22.88313675 − 23.29315376 = −0.41001701 → **−0.4101** (rounded-input convention).

## 2. Epoch-by-epoch val/mae vs parent (ep1..32; Δ = child − parent)

| ep | parent | child | Δ | ep | parent | child | Δ |
|---|---|---|---|---|---|---|---|
| 1 | 146.9976 | 147.2848 | +0.2872 | 17 | 23.8016 | 23.6681 | −0.1335 |
| 2 | 69.0238 | 67.8493 | −1.1745 | 18 | 23.7184 | 23.6231 | −0.0953 |
| 3 | 49.4376 | 49.0815 | −0.3560 | 19 | 23.6163 | 23.5209 | −0.0954 |
| 4 | 39.5353 | 39.7665 | +0.2312 | 20 | 23.5276 | 23.3665 | −0.1611 |
| 5 | 34.1235 | 34.2174 | +0.0940 | 21 | 23.5090 | 23.2575 | −0.2515 |
| 6 | 30.6235 | 30.7832 | +0.1597 | 22 | 23.4600 | 23.1708 | −0.2892 |
| 7 | 28.5269 | 28.6195 | +0.0926 | 23 | 23.3879 | 23.0954 | −0.2925 |
| 8 | 27.1416 | 27.2013 | +0.0597 | 24 | 23.3307 | 23.0270 | −0.3037 |
| 9 | 26.0856 | 26.1250 | +0.0394 | 25 | 23.3257 | 23.0213 | −0.3044 |
| 10 | 25.3461 | 25.4432 | +0.0971 | 26 | 23.2932 | 22.9917 | −0.3015 |
| 11 | 24.7792 | 25.0122 | +0.2329 | 27 | 23.3194 | 22.9766 | −0.3428 |
| 12 | 24.3675 | 24.6589 | +0.2915 | 28 | 23.3075 | 22.9482 | −0.3593 |
| 13 | 24.2335 | 24.4993 | +0.2658 | 29 | 23.3070 | 22.9319 | −0.3750 |
| 14 | 24.0380 | 24.2272 | +0.1893 | 30 | 23.3221 | 22.9075 | −0.4146 |
| 15 | 23.8607 | 24.0792 | +0.2185 | 31 | 23.3341 | 22.8905 | −0.4436 |
| 16 | 23.9168 | 23.8655 | −0.0513 | 32 | 23.3364 | 22.8831 | −0.4533 |

Sustained? Yes in the tail, no in the middle. Child leads 19/32 epochs, trails 13/32:
trails ep1 (+0.29) and ep4–ep15 uninterrupted (12 straight, worst +0.2915 @ep12);
leads ep2–ep3 (up to −1.1745 @ep2), then leads ep16–ep32 uninterrupted (17 straight),
margin widening monotonically from −0.0513 @ep16 to −0.4533 @ep32 with one ripple
(−0.1335 → −0.0953 @ep17–18). Cross-over mechanics: parent's curve breaks first —
parent upticks +0.0561 @ep16 (23.8607 → 23.9168) while child keeps descending; child
passes at ep16 and never gives it back. Child's own curve is strictly monotonically
decreasing across all 32 epochs (every epoch a new best); parent's is not (uptick
@ep16, then post-best drift +0.0262 @ep27 through +0.0433 @ep32).

Best@32 descending — quantified: child ep26→ep32: 22.991701 → 22.976624 (−0.015077) →
22.948185 (−0.028439) → 22.931915 (−0.016270) → 22.907524 (−0.024391) → 22.890499
(−0.017025) → 22.883137 (−0.007362). Total ep28→ep32 gain −0.065048 (mean −0.01626/ep);
final step −0.0074 is below the same-seed paired precision (~0.02), so the run ended
still descending but flattening — no plateau observed within budget.

## 3. Bar distances

- Vs parent: 22.8831 − 23.2932 = **−0.4101** (exact −0.41002). 1.37× the 0.30 falsifier step.
- Vs anchor-rule support line 22.9932 (= 23.29315376 − 0.30): 22.9932 − 22.8831 = child clears by **0.1101** (exact 0.11002). Margin is 4.6× H0014's 0.024.
- Vs literal text line 21.159 (idea.md DISPROVED-IF quotes the v1 anchor 21.459; superseded per AGENTS §6 falsifier-semantics rule): 22.8831 − 21.159 = **+1.7241 above** — misses the literal number by 1.72; recorded here for the record only, the live-parent line governs.

## 4. Wall-clock

- Child 1751.4 s vs parent 1699.4 s: **+52.0 s (+3.06%)** for the verify-mask forward (top-K + cosine gate) — small, same order as run-to-run jitter band; no budget concern (48.6 s headroom, budget_hit=false, full 32/32 epochs).

## 5. Internal consistency

- All green: TB minima = result.json bests on both nodes; best_epoch argmin correct (child ep32 = last val; parent ep26, last val +0.0433 above best); info.json best_metric/train_seconds/epochs match result.json on both nodes; checksums match file contents (child diff = one switch line only, pluggable rule holds); seed/deterministic/augment identical across the pair; no NaNs in either val/mae series; train/loss_epoch strictly fits on both (child 92.74 → 2.603; parent 91.30 → 2.834 — child fits train slightly better too).
- One error-profile wrinkle: MAE and RMSE diverge on the child tail — child RMSE bottoms 90.384 @ep23 then degrades to 91.074 @ep32 (+0.689) while MAE improves every epoch; parent RMSE @best-ep26 is 90.803, i.e. child MAE −0.41 coexists with child RMSE +0.27 vs parent-best-epoch. Same pattern class as a suppressor that trims many small overcounts while leaving/adding a few large misses.

- Child −0.4101 vs parent with a 17-epoch sustained, widening lead and a still-descending best@32; clears the live 22.9932 line by 0.1101 (4.6× H0014's margin) while missing the superseded literal 21.159 by 1.72.
- Crossover is dated: 12 straight trailing epochs (ep4–15) flip at ep16 on the parent's +0.056 uptick, then a monotonic tail; final-step gain −0.0074 is under paired precision, so the curve flattened but never plateaued.
- Fully consistent run (checksums, best.pth bit-exact, one-switch diff, 48.6 s budget headroom) with one MAE/RMSE divergence to carry into qual+causal: RMSE bottoms @ep23 then +0.689 while MAE falls throughout.

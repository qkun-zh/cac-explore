# Quantitative Feedback — N0021_dino_partialft

## reasoning

**Headline** (result.json): best MAE **20.438 @E25**, final 21.434, RMSE 83.055, 40/40 ep, 1455s,
params 23.11M. Parent N0010: best 21.531 @E26, final 22.607, RMSE 81.97.

| Metric | N0010 (frozen) | N0021 partial | Δ |
|---|---|---|---|
| best MAE | 21.531 | **20.438** | **−1.09 (−5.1%)** ✅ |
| final MAE | 22.607 | 21.434 | −1.17 ✅ |
| final RMSE | 81.97 | 83.06 | +1.09 ❌ |
| RMSE/MAE (final) | 3.63 | **3.88** ❌ | worse |
| RMSE/MAE (@best) | 3.53 (75.91/21.53) | 3.91 (79.84/20.44) ❌ | worse |

Ratio note: the quoted 83.06/20.44 = 4.06 mixes final RMSE with best MAE. Like-for-like is
still worse than parent either way (3.88 final, 3.91 at-best), so conclusion unchanged.

**Why RMSE worsened while MAE improved — trajectory evidence:**
- E16 first beats parent best (21.30); E18 hits **MAE 20.466 / RMSE 72.890 (ratio 3.56)** —
  the best-tail state of the whole run.
- E19–E24 regress (21.6–25.2), E25 recovers best MAE 20.438 but RMSE only 79.84.
- E26–E40: train loss 3.78→1.45 (−62%) while val MAE pins at 21.1–21.8 and val RMSE climbs
  79.8→83.1 (+4%). Classic tail-drift: continued tuning of blocks 10-11 fits high-count train
  outliers harder (MSE gradient domination) without moving median accuracy. The E18 state had
  parent-level calibration; 22 more epochs of low-LR backbone drift traded RMSE for nothing.
- Overfitting worse than parent: train loss ends 1.45 vs parent 7.8; val flat since E25.

**Partial FT vs full FT vs full freeze:** Full FT (N0021_dino_fullft) collapsed to 48.4 MAE
with E5 instability spike. Partial FT (blocks 10-11 + norm only, 0.1× lr) beat the frozen
champion by −5.1% and never destabilized (no OOM/NaN; worst epoch 25.2 @E14). Ordering:
partial FT > frozen > full FT. Unfreezing just the last 2 blocks gives real but modest gain.

**Wasted compute:** best @E25 of 40; last 15 epochs produced zero improvement (~735s, 51%).
Early-stop patience≈5 after E25 (or cap ~28ep as N0010 synthesis suggested) is justified.

## actionable_feedback

1. Keep partial-FT recipe (blocks 10-11, backbone_lr_mult=0.1) as new base substrate.
2. Binding constraint is now the outlier tail, NOT mean accuracy. Do not retry Huber /
   tail-reweight (failed levers); try checkpoint selection by val RMSE-constrained MAE
   (pick best MAE among epochs with RMSE ≤ running-min +10% → picks E18: −0.03 MAE,
   −7 RMSE) or test-time augmentation.
3. Two-phase schedule: unfreeze blocks 10-11 only for first ~20ep, re-freeze after — E18
   evidence suggests tail damage accrues in late epochs under cosine decay.
4. Cap epochs at 28 with patience=5; halve wall-clock at zero MAE cost.
5. Log per-image worst-20 val errors (high- vs low-count split) before choosing the next fix.

## hypothesis_updates

- hypothesis_id: H0030
- evidence_type: supports
- strength: 0.75
- reasoning: "IF only blocks 10-11+norm are unfrozen at 0.1×lr THEN gains materialize where
  full FT failed. Evidence: best 20.44 (−5.1% vs frozen champion), stable training, vs full-FT
  collapse 48.4 with instability. Falsification bar (≤18) NOT met — gain is modest, and RMSE
  tail degraded (3.88 vs 3.63), so support is partial-moderate, not decisive."
- ⚠️ Booking caveat: hypotheses.jsonl H0030 carries the R001 point-detection text, yet both
  full-FT-contradicts and this partial-FT-supports evidence target a *fine-tuning-scope*
  claim. Resolve the ID collision before booking (re-book under a fresh id, e.g. H0032).

# Causal Feedback — N0021_dino_partialft

## 1. Causal chain: intervention → effect → mechanism
Intervention: unfreeze exactly `blocks.10./blocks.11./norm.` (model.py:39-43) at lr 1e-4
(1e-3 × `backbone_lr_mult=0.1`), head at 1e-3. Effect: best MAE 20.438@E25 / 20.466@E18
(RMSE 72.89!) vs frozen parent 21.53 (−5.1%), training never unstable (worst epoch 25.2@E14,
no OOM/NaN, 40/40 done). Mechanism: the 10 frozen blocks act as a **fixed feature anchor**, so
the head sees a stationary input distribution and converges normally (train loss 8.64@E13 →
1.45@E40), while the 2 unfrozen top blocks — semantically closest to count/instance abstraction
— make a *bounded, slow* drift that adds dataset-specific detail the frozen features lacked.

## 2. Why blocks 10-11 only: timescale separation (τ_drift vs τ_fit)
Fine-tuning is stable only when feature drift per unit time < head re-fit rate. Full FT breaks
this multiplicatively: shifting patch_embed/pos_embed/early blocks perturbs inputs to *all*
subsequent blocks, which are themselves updating — feature velocity compounds with unfrozen
depth. Evidence (N0021_dino_fullft train.log): catastrophic **from E1** (MAE 48.71, loss
46.29 — before any cosine decay), E5 spike loss **93.89 / MAE 155.4** (large-count batch
gradient blowing up representations), then a dead plateau E7–E10: loss 35.34→35.23 (Δ=0.1 over
4 epochs), MAE pinned ~48.4. That flat-loss plateau is the signature of a moving-target
equilibrium, not slow convergence. Independent replication: N0022 (EBC + full FT) best 21.708,
final 27.40, stopped@E24 — full FT fails to beat champion across two different heads. Partial
FT restores τ ordering: 2-block drift at 1e-4 is slow/local; head at 1e-3 wins the race.

## 3. Why lr×0.1 specifically — and its falsifier
Claim: 0.1× places per-step backbone displacement below the noise scale of head gradients,
keeping taps 6+11 coherent for the softmax layer-gate. But **scope and lr are confounded**: the
failed sibling also used mult=0.1 yet collapsed — so 0.1 alone is insufficient; stability came
from scope×lr jointly. Falsification tests: (a) if lr is the binding constraint, fullft @
mult=0.01 should stabilize (<24 MAE by E8); if scope binds, it stays ≥25. (b) if 0.1 is merely
conservative (not special), mult=1.0 on blocks 10-11 should ALSO be stable — a cheap decisive
probe. Current evidence cannot separate "0.1 near-optimal" from "0.1 arbitrary among safe values".

## 4. Confounds I cannot rule out
1. **Dropout mismatch**: sibling config had dropout=0.15 vs champion 0.1 — the comparison is
   two-knob, not clean. 2. **Budget**: fullft planned 30ep, killed after E10 (result.json
   status "running"); though the E1–E10 trajectory rules out "needed more epochs" as primary
   cause. 3. **Single seeds**: ±0.3 MAE run variance; −1.09 gain is probably real, magnitude
   noisy. 4. **No warmup** in either run: the E5 spike could be an AMP cold-start artifact any
   warmup would fix, independent of drift. 5. Late-run RMSE creep (79.8→83.1 over E26–E40 while
   train loss fell 3.78→1.45) shows even 2-block FT causes slow tail overfit — the same
   mechanism as fullft, just dosed lower; "safe vs unsafe" may be a continuum, not a dichotomy.

## 5. Pre-registered predictions (if the timescale story is true)
- **mult=0.05** (blocks 10-11): drift dose halves → most gain vanishes: best MAE 20.8–21.6,
  converging toward frozen 21.53. If it MATCHES 20.44, gain is not drift-dose-dependent.
- **mult=0.2**: faster early descent but amplified tail damage: final RMSE ≥84, RMSE/MAE ≥3.9;
  best MAE within ±0.4 of 20.44. If it cleanly beats 20.44 with stable tail → 0.1 not special.
- **blocks 9-11 @0.1**: graceful scaling predicts 20.0–20.6 (block 9 also count-relevant);
  degradation >+1.0 MAE or any spike ⇒ sharp depth-threshold drift, refuting 3-block scope.
- Watch E16–E25 window: mechanism predicts best epochs occur there; a post-E30 best would
  contradict the "gain early, drift damages late" narrative.

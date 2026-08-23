# diagnostic.md — N0016_dino_seqcount

## 1. Failure trace
Symptoms: train CE 0.0307 (E1) → 0.0000 (E2+); val MAE pinned at 63.53888086582973 every epoch; RMSE 138.909 constant.

- **Targets are ~all zeros** (engine, code/engine/train.py:149): `F.adaptive_avg_pool2d(gt_d,(14,14)).flatten(1).round().clamp(0,63)`. GT density integrates to the count (Σdens = count, fsc147.py:48). **Mean**-pooling divides each 28×28 window's mass by 784: a pooled cell ≈ k/784 for k objects in that cell. A cell rounds to ≥1 only if it holds ≥392 count-mass — impossible when typical total count is tens. So `targets` ≡ 0 for virtually all images/positions.
- **Trivial copy task**: with ~100% of the 196 labels = token 0, the decoder just learns p(0)=1 at every position → CE→0 by E2 (E1 residual 0.0307 = fitting rare nonzero cells on thousand-count images).
- **Eval confirms all-zero output**: engine sums argmax over 196 steps (train.py:72). Predicting zero everywhere ⇒ pred=0 ∀images ⇒ MAE = mean(|gt−0|) = **mean val count = 63.53888086582973**, matching the log to full precision and bit-exact across epochs (weights change ⇒ identical predictions only if output is the degenerate constant). FSC147 mean ≈63/img (paper); champion MAE 21.53 « 63.5 consistent. RMSE 138.9 ≈ count std from heavy tail.
- Checked and CLEARED: pooling shape [B,1,S,S]→[B,1,14,14] correct (392/14=28 exact); `.flatten(1)` row-major matches decoder position order (no transpose bug); shift alignment correct (model.py:92: logits[i] sees [Start,t₀..t_{i−1}] via diagonal=1 mask, predicts targets[i] — no off-by-one copy path); eval path calls model.eval() + AR loop + sum(argmax) correctly.

## 2. Top-2 bugs
1. **Target scale bug** — code/engine/train.py:149. Mean-pool then round destroys counts by factor 784. Root cause of every symptom.
2. **Vacuous smoke gate** — smoke Synth data (train.py:26–46) uses blobs normalized to sum≈1 each (≤5 objects on 49×49 grid); same pool+round turns those into all-zero labels too, so `--smoke` passed green while learning nothing. No assertion existed on loss magnitude or prediction variance. Secondary static find: model.py:81 — prompt token is cat'ed then sliced off (`adapter(cat([prompt,tokens]))[:,1:]`); adapter is pointwise ⇒ exemplar prompt has ZERO effect (diverges from "chencoder verbatim" claim; not this failure's cause but fix in v2).

## 3. Minimal fix + v2 smoke asserts
Fix (one line, train.py:149) — rescale pooled map so Σtargets ≈ count, robust to any pooling geometry incl. non-integer smoke grids:
```python
pooled = F.adaptive_avg_pool2d(gt_d.float(), (Lg, Lg))
scale = gt_d.flatten(1).sum(1, None) / pooled.flatten(1).sum(1, None).clamp_min(1e-8)
targets = (pooled * scale[:, :, None, None]).flatten(1).round().clamp(0, K - 1).long()
```
(Equivalently `* (S//Lg)**2` for real data only.) Also wire prompt into memory properly (add or FiLM, don't slice).

V2 smoke MUST assert:
1. Initial CE ≈ ln(64)=4.16 (uniform head), NOT <0.5; after 2 ep loss > 0.001.
2. Target invariant per batch: `(targets.sum(1) - counts).abs().max() <= clamp_loss_tolerance`; assert fraction of nonzero tokens > 0.
3. Predictions vary: across-batch std of summed preds > 0 and ≠ any constant to machine precision (a bit-identical MAE repeat = instant red flag).
4. Unit test: zero-image input → near-uniform head logits (no premature collapse); after 50 steps on synthetic data with known per-cell counts ≥1, model beats all-zero baseline MAE.
5. Timing: 250 s/ep × 40 ep ≈ 2.8 h » τ_max 30 min ⇒ would timeout ~E7 even fixed; cut epochs or raise --timeout-min, consider vectorizing eval (batch AR decode).

Verdict: H0024 UNTESTED — this run measured the degenerate all-zero predictor, not SeqCount. Re-run v2 after fix before any refutation booking.

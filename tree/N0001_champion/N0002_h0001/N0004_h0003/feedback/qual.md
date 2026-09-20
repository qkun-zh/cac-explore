# N0004_h0003 — qualitative feedback (text-log analysis, AGENTS §9)

- **Node:** N0004_h0003 · **Hypothesis:** H0003 `use_padapt` (LOCA-style image-conditioned
  prototype adaptation) · **Parent:** N0002_h0001 (22.5641 @ep30) · **Composition:** SOLO.
- **Status:** `done`, 32/32 epochs, `budget_hit: false`, elapsed_s 1773.3 (< 1800 cap).
- **Artifacts read (server):** `tree/N0001_champion/N0002_h0001/N0004_h0003/result.json`,
  `info.json`, `config.toml`, `run/latest/best.pth`, TB scalars at
  `run/latest/tb/t/events.out.tfevents.1789885586.*`, and `chain_run.log`
  (tail + grep). Parent TB read from `N0002_h0001/run/latest/tb/t/`.

## 1. What the curves actually did

Both runs logged one scalar per epoch at identical TB steps (227, 455, … 7295 = 32 points;
identical step grid confirms the same 228-step epoch layout).

**Child (`use_padapt=true`) — `head.padapt` present, `model.py:162,171-175,233,289-290`:**

| epoch | 1 | 2 | 3 | 7 | 10 | 11 | 12 | **13** | 14 | 20 | 26 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val/mae | 152.78 | 69.22 | 49.692 | 28.893 | 25.987 | 25.688 | 25.579 | **25.566** | 25.592 | 26.018 | 26.13 | 26.167 |
| val/rmse | 175.78 | 120.85 | 112.99 | 99.875 | 96.815 | 96.35 | 96.142 | **96.099** | 96.138 | 96.611 | 96.75 | 96.80 |
| train/loss_epoch | 41.56 | 9.0033 | 8.3664 | 6.5575 | 5.6277 | 5.7303 | 5.5633 | 5.5819 | 5.5878 | 5.5859 | 5.5757 | 5.5889 |

- **Best = 25.566 @ep13** (`result.json:7-8` `best_mae`/`best_epoch`; `info.json` `best_metric`).
- After ep13 the val series is **monotonically rising**: 25.566 → 25.592 → 25.644 → … →
  25.976 → … → 26.167 @ep32. Val RMSE mirrors it exactly (96.099 → 96.80, a clean monotone
  rise). The test-loss surface is genuinely degrading, not MAE/RMSE disagreeing.

**Parent (`use_simprior`, no padapt):** val/mae 153.29 → 23.907 @ep13 → 22.564 @ep30 →
22.570 @ep32; val/rmse 174.39 → 89.24; train/loss_epoch 38.02 → 2.6183 (monotone).
Checkpoint confirms `epoch 30, best_mae 22.564146`, **0** `padapt` keys.

## 2. Overfit signature? No — it is underfit-plus-degradation

The pre-registered question was whether train loss kept falling while val worsened. **It did
not.** In the child, `train/loss_epoch` stops improving around ep12 (5.56) and sits in a flat
5.56–5.59 band through ep32 (final 5.5889) — i.e. training-set fit **stalls before/around the
ep13 val turn**, then val climbs for 19 straight epochs. So there is no diverging
train-down/val-up wedge. Instead the child:
1. carries a **strictly worse train loss at essentially every epoch from ep2** (9.0033 vs
   parent 8.6532; ep13 5.5819 vs 5.2261; ep32 5.5889 vs 2.6183), and
2. **never reaches the parent's level on val either** (child best 25.566 vs parent ep13 23.907).

This is the signature of a mechanism that both **caps training fit** and **deforms the val
decision**, not classic overfitting. Qualitatively it matches LOCA's named failure mode
(idea.md §6: unmodulated prototype adaptation pulls background statistics into the prototype):
the adapted prototypes send the Condenser keys/values to a region that is worse for held-out
images. The plateau in train loss is also consistent with the adapter's gradient gate
(`out` zero-init, idea.md §3) being only weakly effective as an optimization path.

## 3. Did the adapter branch cause early instability? No

- Epoch-1 val MAE is 152.78 vs parent 153.29; epoch-2 69.22 vs 70.05 — the child is actually a
  hair **better** for the first two epochs, then falls behind from ep3 (49.692 vs 48.788).
- No NaN / Inf / loss spike anywhere; `train/loss_epoch` descends smoothly 41.56 → 9.00 → 8.37.
  `lr-AdamW` bottoms at 3.41e-6 exactly as the cosine schedule specifies. Zero-init worked as
  designed: step-0 forward is numerically the parent's (idea.md §3).
- So the adapter did **not** blow up the run. It is a slow, persistent drag that only becomes
  visible once the parent would have kept improving (ep3 onward), with the curve tipping at ep13.

## 4. Checkpoint / use-evidence (is the refute informative?)

`run/latest/best.pth` is the ep13 best (`epoch 13, best_mae 25.566089`, 125.7 MB). Adapter
parameter norms in that checkpoint:

- `head.padapt.out.weight` Frobenius norm **2.4946**, bias norm 0.2905 — **not ≈ 0**.
- `head.padapt.attn.in_proj_weight` 9.199, `eproj.weight` 5.085, `fproj.weight` 4.581.

This satisfies idea.md §6's discriminator: an `out` norm ≈0 would mean the optimizer discarded
the adapter and the run uninformative. Here `out` moved decisively off zero, so the adapter was
**actively learned and used**, and it still lost — the refute is about adaptation capacity, not
about an ignored branch. (`result.json:12-13` records `config_sha256 769ddfa05d25dfa9`,
`model_sha256 93c221209869fe69`; the child config differs from the parent only by
`use_padapt = true`, parent `config_sha256 4d4c9baec459f657`.)

## 5. Comparison to N0002's clean descent

N0002 is a textbook clean descent: train loss falls monotonically end-to-end (38.02 → 2.62)
and val MAE falls monotonically to ep30 (22.564), with val/RMSE agreeing. N0004 diverges from
this trajectory almost immediately (worse train loss from ep2) and its val **reverses** while
the parent is still descending. The two runs share epoch 13 directly: parent 23.907, child
25.566 — the child is **1.66 MAE worse at the parent's own inflection region**, and the gap
widens to 3.60 at the end. This is not noise-level (±1 MAE per AGENTS §6): the child's *best*
epoch is 3.00 above the parent's *best*.

## 6. Warnings / benign items (no errors)

`grep -iE "WARNING|Error|Traceback|nan|inf|budget"` on `chain_run.log`: **no Error/Traceback,
no Inf, no `_BudgetStop`, no timeout.** The only hits are routine and identical to the parent's
run:
- `lightning ... Found 151 module(s) in eval mode at the start of training` — benign info
  (frozen backbone), fires on N0002 too.
- `torchmetrics ... compute method of metric MeanMetric was called before update` — benign.
- `fsc147.py:70 UserWarning: The given NumPy array is not writable` — benign, repeated per
  batch, self-suppressed after the first.
- `train/mae` is **nan for all 32 epochs — on both child and parent**. It is a logging artifact
  of that metric, not mechanism-specific or new; it does not affect the MAE used for the verdict.
- Termination is clean: `` Trainer.fit` stopped: `max_epochs=32` reached`, last line
  `Epoch 31/31 … val/mae: 26.167`. `run_smoke.log` ends `SMOKE_RESULT {'smoke': True}`.

## 7. Verdict

**REFUTED:** `use_padapt` was actively trained (checkpoint `out.weight` norm 2.4946, not 0)
yet its best val MAE is 25.566 @ep13 — 3.31 above the pre-registered 22.26 bar and 3.00 worse
than the parent's 22.5641 @ep30 — with val MAE rising monotonically 25.566 → 26.167 over
ep13→32 while train loss plateaued at ~5.58 (parent 2.6183); this is an underfit-and-degrade
refute, not an overfit signature and not early instability.

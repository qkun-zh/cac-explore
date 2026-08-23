# feedback/causal.md — N0012_dino_highres518

## reasoning
**The 26.03 vs 21.53 gap is NOT a clean refutation of high-res — the run was truncated.**
Evidence chain:
1. **Timeout, not convergence**: engine kills at `TAU_MAX_MIN=30min` (train.py:85). N0012 ran
   ~63 s/ep vs N0010's ~32 s/ep (+75% tokens × half batch ⇒ ~2× step cost). At E18 it hit the
   wall; status stayed "running"/timeout, epochs_done=18/40. N0010 got all 40 epochs.
2. **MAE was still descending when killed**: E16 29.39 → E17 27.65 → E18 26.03 (best each of
   last 3 epochs, loss 17.31→16.30 still dropping). No plateau, no overfit signal yet.
3. **Cosine schedule mismatch (primary causal suspect)**: cosine was built for 40 ep; at E18
   lr≈5.6e-4 (56% of peak) — the model never entered the low-lr refinement phase that produced
   the parent's late gains (N0010: 23.7@E23 → 21.53@E26 → stable, all post-E23 at lr<4e-4).
   Comparing N0012@E18/high-lr to N0010@E26/low-lr conflates schedule position with resolution.
4. **Pos-embed interpolation: NOT a problem**. 518px is DINOv2's *native* training resolution;
   dynamic_img_size=True yields exact 37×37 pos-embeds, zero interpolation error. This channel
   is exonerated.
5. **Batch size 8→4**: halves effective batch ⇒ 2× gradient noise; only the head/adapters
   train (frozen backbone) so sensitivity is moderate, but it compounds with the un-finished
   schedule. Confound, likely secondary.
6. **Token count itself**: sum-conserving density loss means +75% tokens doesn't bias counts,
   but gives the head a harder, higher-dim regression target needing more optimization steps —
   consistent with slower convergence per-epoch.
7. **RMSE criterion did fire**: 95.34/26.03 = 3.66× ≥ 3.63 bar — but at E18 with lr still high;
   outlier tail typically shrinks most during low-lr refinement (cf. N0010 RMSE curve).

**Verdict**: H0021 is *confounded-truncated*, not refuted. Current data weakly contradicts
(behind parent at equal-ish compute) but cannot distinguish "high-res hurts" from "high-res
needs its full schedule". What WOULD make resolution work: (a) raise timeout to ≥55 min OR cut
epochs to ~20 with cosine re-anchored to 20 so the low-lr phase actually happens; (b) restore
batch_size 8 via grad-accumulation×2 (memory allows: activations only, no param growth);
(c) optionally warmup+higher peak lr since more tokens dilute per-token gradient signal.

## actionable_feedback
- Do NOT book H0021 as refuted on this run; mark inconclusive/confounded in synthesis.
- Retry lever (cheap): rerun N0012 config with `TAU_MAX_MIN=60` + batch_size=8 via
  grad-accum 2, everything else identical — isolates schedule completion as the variable.
- Cheaper probe: epochs=22, cosine over 22, timeout 30 min — tests whether completed
  schedule at 518px closes the gap within budget.
- If retry still ≥23 after full schedule, then high-res is genuinely negative at this budget
  (2× train cost for no gain) and should be abandoned; pivot to H0022 tail-reweight.

## hypothesis_updates
- hyp_id: H0021 | evidence_type: contradicts | strength: 0.45 |
  reasoning: best 26.03 > 21.53 parent and RMSE/MAE 3.66≥3.63 fired the disproof criteria,
  BUT run truncated at E18/40 by timeout with MAE still improving and lr at ~56% of peak —
  the causal attribution (resolution bad) is confounded with (schedule incomplete). Weak
  contradicting evidence only; confidence should drop modestly, not collapse.
- hyp_id: H0017 | evidence_type: supports | strength: 0.35 |
  reasoning: multi-tap gated stack remains stable and trainable at 518px native res —
  monotone MAE improvement through E18, no divergence/OOM, gate+prompt machinery intact at
  1369 tokens; the substrate transfers across resolutions even though the run underperformed.

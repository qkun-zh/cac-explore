# feedback/causal_v2.md — N0017_dino_tailreweight (post-run sign-error re-analysis)

## Reasoning
SIGN AUDIT (confirmed): idea.md demanded UP-weighting rare high-count images ("reweight loss UP
toward the error-dominating dense samples"), but the engine (code/engine/train.py:290) executes
`w = 1/clamp(gt_c,min=1)**tail_exp` with tail_exp=+0.5 ⇒ w=count^-0.5: a count-1000 image gets
~1/32 the raw gradient weight of a count-1 image. N0017 ran the INVERTED experiment — targeted LR
cut on exactly the samples whose squared-error energy dominates the loss. Fingerprint corroborates:
MAE worse (22.19 best > 21.53) AND RMSE/ratio worse (3.80 vs 3.63) = starvation signature; idea.md's
own risk table says true up-weighting failure would show MAE worse + RMSE BETTER.
DOSE-RESPONSE PREDICTION (as of run time): removing tail gradient hurt ⇒ symmetric logic predicted
adding it should help, via one config flip (tail_exp=-0.5).
RESOLVED RETROACTIVELY: that exact sibling, N0019_dino_tailup (w∝count^+0.5, same mean-1 norm),
already ran same day: best MAE 23.361, RMSE 84.38, ratio 3.59 — WORSE than N0017 on MAE (+1.83 vs
parent) and failing even the "RMSE improves" fingerprint vs parent's ~82/21.53. Both directions of
the explicit knob lose to uniform ⇒ flat inverted-V optimum centered on uniform weighting.
Mechanism: the density-MSE base loss already scales ~count², so dense images naturally dominate
gradient allocation; up-tilt overshoots into overfitting them, down-tilt only partially rebalances
yet still taxes median-case fitting. Conclusion: tail errors are capability-limited (28×28 grid,
frozen features), not attention-limited — no loss-space reallocation can recover them.

## Actionable Feedback
- Do NOT run another sign-corrected sibling: N0019 already adjudicated the corrected sign and
  failed decisively. STATE.md correctly lists "tail-reweight ±" among failed levers; keep closed.
- Confound handling validated: booking H0026 as contradicted-w/-confound (w=0.65, not refuted)
  was right — the run tested tail-starvation, and the family verdict came from the pair
  (N0017+N0019), not from either alone.
- Process fixes stand (from synthesis): (a) numeric micro-example gate in idea.md (evaluate w at
  counts 20/200/2000 before coding — catchable at zero GPU cost); (b) direction-encoded config
  keys (`dense_downweight`/`tail_up`, never ambiguous `tail_reweight`); (c) stratified val MAE by
  true-count bucket mandatory for any loss-space node; (d) prefer dataset-level normalizer
  sqrt(c/c̄_train) over batch-mean normalization.
- If the outlier tail is ever re-attacked, go data-space (dense-sample oversampling in the sampler,
  decouples reallocation from per-batch noise) or capability-space (output resolution, N0023 line);
  loss-space is exhausted.

## Hypothesis Updates
- hyp_id: H0026 | evidence_type: causal re-audit | strength: 0.65 | verdict: contradicts (unchanged)
  reasoning: Kill bar stands for formula-as-written; v2 audit confirms inversion diagnosis, and
  N0019's independent failure removes any residual doubt that the harm came from the knob itself.
  Booked c=0.435 remains correct; no new event needed.
- hyp_id: H0022 (loss-shaping family) | evidence_type: causal, cross-node | strength: 0.85 |
  verdict: contradicts / CLOSE FAMILY
  reasoning: Both arms now empirically tested on the clean champion isolation — down-weight
  (N0017, 22.19) and up-weight (N0019, 23.36) — both lose to uniform (21.53). Uniform weighting is
  confirmed near-optimal for this recipe; residual tail requires capability, not reweighting.
  Ensure a closing evidence event against H0022 is booked if the gen-6 rebuild dropped it.

# feedback/causal.md — N0017_dino_tailreweight

## Reasoning
SIGN AUDIT (decisive): idea.md motivation demands UP-weighting rare high-count images
("reweight loss UP toward the samples whose errors dominate the metric"), but the executed
engine formula (code/engine/train.py:166) is w_i = count_i^-0.5 — a count-1000 image gets
~1/32 the raw weight of a count-1 image. N0017 therefore ran the INVERTED experiment: it
starved the dense tail of gradient instead of feeding it.

Causal chain consistent with every observed number:
1. Base-loss energy concentrates on dense images (density MSE and count L1 grow with count),
   so down-weighting them is a targeted LR cut on exactly the hardest cases.
2. Their residuals stay large → RMSE worsens out of proportion: final RMSE/MAE 3.80
   (85.79/22.60) vs parent 3.63; even at best epoch E30 the ratio is 3.73 (82.87/22.19).
3. Median-case accuracy degrades mildly (+0.66 best MAE, +3%) as reduced tail supervision
   slows adapter/head fitting; no instability (clean diagnostics, 23.11M, full 40ep).
Diagnostic signature: idea.md's own risk table predicts a TRUE up-weighting failure would
show MAE worse + RMSE BETTER. We observe BOTH worse — the fingerprint of the down-weight
direction, corroborating sign inversion rather than refuting up-weighting. Attribution is
otherwise clean: model.py identical to champion, single-knob delta, per-batch mean-1
normalization keeps gradient scale neutral (pure reallocation).

## Actionable Feedback
- H0026-as-written is CONTRADICTED (own kill bar: 22.187 > 21.53), but with a sign-error
  confound: the implemented formula contradicts the hypothesis' stated mechanism, so this
  result must NOT be booked as evidence against tail-upweighting.
- Recommend ONE GPU slot for an N00xx sibling with the sign corrected: w ∝ count^+0.5,
  mean-1 normalized. Zero code change needed — set `tail_exp=-0.5` in config.py (train.py:166
  then yields w = count^0.5). Justified despite 5 consecutive failures because:
  (a) dose–response logic from N0017 (removing tail gradient hurt → adding it is live);
  (b) it targets the dominant residual (RMSE/MAE 3.8) on the proven champion recipe;
  (c) cheapest possible test (~1275s, config-only, cleanest isolation in gen-4).
  Guardrails: early-stop rule armed (≥+1.5 worse than parent trajectory at ep16+); add
  stratified val MAE by true-count bucket (already specced in idea.md) so the mechanism is
  visible even if top-line MAE fails.
- Loss-shaping status: NOT exhausted — both failures were wrong-signed attempts at the tail
  (H0020 Huber caps gradients; N0017 inverse-sqrt starves it). The count^+0.5 sibling is the
  LAST loss-space shot; if it also fails, close the whole H0022 loss-shaping family and
  reallocate to data-space equivalents (dense-sample oversampling in the sampler decouples
  reallocation from per-batch weight noise) or to the live augreg merge (N0013 ahead).
- Implementation nit for the sibling: per-batch mean-1 normalization couples weight scale to
  batch composition (all-low-count batches inflate relative tail weight). Prefer a dataset-
  level fixed normalizer w = sqrt(c / c̄_train) to cut gradient variance.

## Hypothesis Updates
- hyp_id: H0026 | evidence_type: quantitative+causal | strength: 0.55 | verdict: contradicts
  reasoning: Kill criterion met for the formula AS WRITTEN (best 22.187 > 21.53; RMSE/MAE
  3.80 > 3.4 bar); but the formula implements the inverse of the claimed mechanism ("shifts
  gradient TO dense samples"), so contradiction applies only to tail-DOWN-weighting.
  Isolation otherwise clean: champion verbatim, single knob, no instability/timeout confound.
- hyp_id: H0022 (family) | evidence_type: qualitative+causal | strength: 0.35 | verdict: revise
  reasoning: Neither capping (H0020 refuted) nor down-weighting (N0017 contradicted) tested
  the stated mechanism; up-weighting remains untested and is the family's last mechanistically
  coherent arm. Book resolution after the sign-corrected sibling runs — either way closes it.
EOF_MARKER_NOT_NEEDED

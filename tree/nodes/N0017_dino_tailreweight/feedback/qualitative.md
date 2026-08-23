# qualitative feedback — N0017_dino_tailreweight

## reasoning
Result: best MAE 22.187 (@ep30) vs parent N0010 21.531; final RMSE/MAE ratio 85.79/22.60 = **3.80** vs
parent 3.63. Both MAE and tail-relative error WORSENED under `w_i = 1/sqrt(max(count_i,1))`.

**Root cause is an idea-level sign error, faithfully implemented.** idea.md's Motivation says "reweight
loss UP toward rare high-count images" (and inspiration_from_GOD §7 + H0022 intend exactly that), but the
specified formula DOWN-weights them: count=30 → w=0.183, count=500 → w=0.045 (~4× less gradient share).
Engine train.py:163-168 implements the spec verbatim, so the run tested the OPPOSITE of the stated intent.
Conceptual slip: classic class-balancing upweights rare CLASSES; here image sampling frequency is uniform
(1 per image), and count was wrongly used as a frequency proxy. Correct transplants: w ∝ count^tail_exp,
or inverse-frequency of COUNT BUCKETS (dense buckets are rare → get w>1).

Metric fingerprint corroborates the causal review: starving dense images further hurt median accuracy
AND raised the tail ratio — the gradient-share dial moved metrics in the direction opposite to hoped,
confirming the dial itself has leverage (not a null manipulation).

Code quality: model.py/config.py are champion-verbatim as specced (docstring clear, contract-compliant,
`loss_count_weight=1.0` explicitly pinned vs engine default 0.3 — good). Engine branch correct numerically
(per-image base loss unchanged, mean-1 normalization, AMP-safe). Deficiencies: (a) name `tail_reweight`
is direction-ambiguous — `dense_downweight`/`inverse_count_weight` would have forced the author to commit
to a direction; (b) no comment stating "w decreases with count"; (c) batch-mean normalization makes
weights depend on batch composition (minor noise source); (d) optional stratified eval (idea tunable)
wasn't implemented — it would have exposed the anti-tail weighting mid-run.

## actionable_feedback
1. NEVER launch on a prose-vs-formula pair without a worked micro-example: evaluate the formula at 2-3
   concrete counts (e.g. 20 / 200 / 2000) directly in idea.md next to the hypothesis. This contradiction
   was catchable in seconds pre-training at ZERO GPU cost — the redundancy check compared mechanisms but
   never executed the arithmetic. Add "numeric self-consistency check" to the Idea→Coding handoff gate.
2. Rename config key to encode direction (`dense_upweight` / `inverse_count_weight`) and add a one-line
   engine comment stating whether w grows or shrinks with gt_c.
3. If retrying the family: w = clamp(gt_c,1)^tail_exp (bucket-normalized), keep mean-1 normalization but
   consider epoch-level stats; implement stratified val MAE by true-count bucket to verify mechanism live.
4. Do not treat 22.19-vs-21.53 as pure noise refutation of "pro-tail helps": this run never applied the
   pro-tail direction. Book the sign flip as the primary finding.

## hypothesis_updates
- {hypothesis_id: H0026, evidence_type: contradicts, strength: 0.75,
  reasoning: "Falsification criterion fired cleanly (best MAE 22.187 > 21.53) under confound-free
  isolation (champion recipe verbatim, single change, identical params/epochs/duration). As literally
  written — w=1/sqrt(count) — the hypothesis is refuted: down-weighting dense images hurts."}
- {hypothesis_id: H0022, evidence_type: supports, strength: 0.55,
  reasoning: "Indirect, sign-flipped support: shifting gradient share AWAY from dense images worsened
  both MAE (+3.0%) and RMSE/MAE (3.63→3.80), matching tail-starvation fingerprint. The dial has real
  leverage in the predicted coordinate system; the untested pro-tail half remains plausible. Retry with
  corrected sign before any confirmation."}
- {hypothesis_id: NEW (propose create H0027-class), evidence_type: neutral, strength: 0.5,
  reasoning: "Proposal for Synthesis: IF pro-tail reweighting (w ∝ count^0.5, mean-1 normalized, plus
  stratified eval) IN champion recipe, THEN best MAE <21.5 AND RMSE/MAE <3.55, BECAUSE dense samples are
  rare and error-dominating so upweighting rebalances gradient allocation. DISPROVED IF best MAE >21.53.
  Motivated by N0017's anti-direction harm + fingerprint; one 21-min run falsifies."}

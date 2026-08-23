# synthesis.md — N0017_dino_tailreweight

## Verdict
**H0026-as-written REFUTED** (best MAE 22.187 > kill bar 21.53; RMSE/MAE 3.73@best, 3.80 final vs
<3.4 claim), **with a sign-error confound**: idea.md motivation demanded UP-weighting dense samples,
but the booked formula AND engine (train.py:166) implement DOWN-weighting (w=count^-0.5). The run
tested tail-starvation, not the intended pro-tail mechanism — contradiction applies only to the
formula-as-booked. **Intended up-weight direction remains UNTESTED** → sibling **N0019_dino_tailup**
(w∝count^+0.5, config-only flip, already registered/coded) carries it; its run adjudicates the whole
H0022 loss-shaping family either way. Metric fingerprint (MAE worse AND ratio worse) matches the
down-weight signature, corroborating the inversion diagnosis. No instability/OOM; cost = parent.

## Quality gate (7 dims, node/H0026-as-executed)
- mechanistic 0.30 — mechanism stated but implemented formula inverted the motivation (spec bug)
- scoped 0.65 — champion verbatim, single knob, mean-1 norm keeps gradient scale neutral
- predictive 0.40 — numeric bars set, but arithmetic never sanity-checked vs prose
- falsifiable 0.75 — kill bar fired cleanly and decisively on both claims
- novel 0.10 — transplanted class-balancing trick; novelty expressly not claimed
- transferable 0.20 — negative result scoped to wrong-signed variant
- actionable 0.25 — yields process fixes + sibling, no direct recipe gain
**quality = 0.38 · avail = 0.62 · score = 0.50**

## Dedup / contradictions resolved
- H0026 contradict strengths: 0.95 / 0.75 / 0.55 (quant/qual/causal). Booked ONE conservative
  consolidated event **w=0.65** (confound-aware midpoint; decisive miss but mechanism-inverted).
- H0022 feedback DIVERGED: quant→contradicts 0.70, qual→supports 0.55, causal→revise 0.35. All three
  reason from the same confounded run ⇒ **book NOTHING on H0022 from N0017**; defer to N0019
  (per causal reviewer: either outcome closes the family).
- Qualitative's proposed NEW hypothesis (H0027-class, pro-tail upweight) is subsumed by the already-
  registered sibling node N0019_dino_tailup — no duplicate bank entry; N0019's synthesis books it.

## Booking list (executed)
1. `create` H0026 (text/tags from idea.md; bank had no entry — Idea does not book).
2. `evidence` H0026 contradicts w=0.65, ts=2026-08-23T16:15:00+08:00, source_node=N0017_dino_tailreweight,
   sign-error confound noted, reviewer range 0.55–0.95. → H0026 c: 0.50→**0.435**.

## Handoffs for Lead (not applied here)
- tree.json: N0017 → status `"synthesized"`, quality 0.38, avail 0.62, score 0.50,
  tested_hypotheses ["H0026"] (best_metric/train_seconds already set).
- memory/failure_modes.md: add (a) prose-vs-formula numeric micro-example gate (evaluate w at counts
  20/200/2000 in idea.md before coding — this error was catchable at zero GPU cost);
  (b) config keys must encode direction (`dense_downweight`/`tail_up`, not ambiguous `tail_reweight`);
  (c) loss-level nodes MUST ship stratified val MAE by true-count bucket;
  (d) batch-mean weight normalization couples scale to batch composition — prefer fixed dataset-level
  normalizer sqrt(c/c̄_train) for N0019-class runs.
- STATE.md next-steps: queue N0019_dino_tailup next GPU slot (guardrails armed: early-stop ≥+1.5 worse
  than parent trajectory @ep16+; stratified bucket eval ON); if it fails too, close H0022 family and
  reallocate to data-space oversampling or resolution merges (N0014/N0015).
- Scheduling pattern: 2 consecutive nodes peaked @65–75% then flat ≥10ep — consider patience-5 early
  stop after augreg merge lands (~300s/run saved).

## tested_hypotheses
[H0026]

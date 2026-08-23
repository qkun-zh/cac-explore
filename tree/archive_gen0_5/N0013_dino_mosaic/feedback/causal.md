# feedback/causal.md — N0013_dino_mosaic

## reasoning
**Outcome**: best 22.404@E38 vs parent 21.531@E26 (+0.87, +4%); final 22.761 vs 22.607. Net
negative but small — plausibly within seed noise; NOT a decisive refutation of augmentation.
Causal chain:
1. **Phase inversion**: N0013 tracked AHEAD through E13-E20 (E13 24.38 vs 25.50; E18 23.59 vs
   24.32-best; E20 23.89 vs 24.96), then missed the parent's defining event — the low-lr
   refinement dip (21.53@E26 once lr fell below ~4e-4). N0013 sat at 24.74@E26 and only ground
   down to 22.4-22.9 across E33-40. Jitter+reg sped early fit but blunted fine-phase gains.
2. **Three-lever confound**: jitter(p=.5)+bbox(.15), dropout 0.1→0.2, wd ×5 moved together —
   the same compound-design flaw N0010's synthesis flagged. "Augmentation hurts" vs
   "over-regularization" is UNRESOLVED; only the bundle is priced, at +0.87.
3. **Over-regularization signals (moderate)**: delayed+shallower refinement (first sub-22.5 at
   E35 vs parent E26) and larger early oscillation (24.4→27.0→28.2 across E13-15). Counter-
   signal: final val nearly matches parent ⇒ at most MILD over-regularization. Note train-loss
   comparison is INVALID here: logged loss is computed on augmented inputs, so 8.26 vs 7.78
   says nothing about overfit.
4. **Tail untouched — mechanism explains why**: RMSE/MAE 3.62× (81.08/22.40) ≈ parent 3.63×.
   Mosaic-lite injects NO count variance: photometric jitter preserves counts; bbox jitter only
   perturbs the prompt↔density correspondence (label-noise style). The SeqCount+ tail benefit
   comes from synthetic HIGH-COUNT mass (paste+sum densities), which this proxy omits entirely.
5. **Early-stop rule misfire**: instantaneous same-epoch deltas crossed +1.5 at E21 (+1.71),
   E23 (+1.95), E26 (+3.21) — strict application would have killed a run that later reached
   22.40. Val oscillates ±1.5/epoch; the rule needs smoothing or it executes recoverable runs.
6. **Costs**: per-image Python jitter loop → 1410s vs 1275s (+11% wall-clock) for −0.87;
   batch-level photometric coin vs per-sample bbox coin is inconsistent granularity (minor).
7. **518px merge (N0014)**: cuts both ways. Pro: finer token grid raises the value of
   appearance invariance and enlarges the overfit surface augreg targets. Con: N0012 proved
   518px is schedule-starved (full 40ep + grad-accum needed), and N0013 shows augreg DELAYS
   convergence ~10ep — merged naively they compound into a guaranteed-truncated run.

## actionable_feedback
- Do NOT promote the augreg bundle to champion; net −0.87 with muddy attribution.
- N0014 merge: keep jitter, REVERT dropout→0.1 and wd→1e-4 — isolates augmentation at 518px
  and removes the suspected over-regularization drag. Hard precondition: re-anchored cosine +
  timeout ≥55min + grad-accum bs8 (N0012 lessons); otherwise skip the merge.
- Vectorize jitter (batched torchvision v2) and unify per-sample coin flips to reclaim ~135s.
- Booking hygiene: idea.md tags the augreg claim "H0022", but H0022 is already tail-reweight
  (booked 13:30) — synthesis must book the augreg resolution under a NEW id.
- Tighten early-stop rule before next dispatch: fire only on best-so-far gap ≥+1.5 held for
  2 consecutive epochs after E16.

## hypothesis_updates
- hyp_id: H0022 | evidence_type: supports | strength: 0.40 |
  reasoning: indirect support for tail-reweight motivation — a second variance-injection lever
  (photometric+bbox jitter) left RMSE/MAE at 3.62× vs parent 3.63×; passive regularization of
  frozen-feature inputs does not touch catastrophic outliers, so explicit gradient reweighting
  remains the only direct untested lever [causal].
- hyp_id: H0017 | evidence_type: supports | strength: 0.50 |
  reasoning: dual-tap gated stack trains stably under input-space jitter + 2×dropout/5×wd:
  monotone descent to 22.40, no OOM/divergence, gate+prompt machinery intact, +11% wall-clock
  only — substrate robust under augmentation, transfers to the N0014 518px merge [qualitative+causal].

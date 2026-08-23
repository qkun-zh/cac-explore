# feedback/quantitative.md — N0012_dino_highres518

## reasoning
Early-stopped at E18/40: best MAE 26.033 (E18, still falling: 28.72→27.65→26.03 over E11-18),
RMSE 95.34. Champion parent N0010 finished at 21.531 (best @E26). Same-epoch comparison is
decisive against H0021's mechanism: N0010 best by E15 was already 24.315 (its E18 snapshot
MAE 25.51), i.e. N0012@392px-parent-trajectory leads N0012@518px by ~1.2-1.7 MAE at matched
epochs despite N0012 taking 2x optimizer steps/epoch (batch 4 vs 8). Per-epoch wall-clock
~62s vs ~31s — resolution costs 2x compute for worse convergence.

H0021 claimed MAE ≤19.0 and RMSE/MAE <3.4 via +75% tokens. Observed RMSE/MAE =
95.34/26.03 = 3.66 — identical to parent's 3.63 outlier tail. The finer token grid did NOT
reduce catastrophic outliers; the small-object-miss mechanism prediction is contradicted so
far. Disproof bar "MAE >21.53" not yet formally met (run stopped early), but the node is
4.5 MAE behind champion with a flatter trajectory and only 22 epochs of lr schedule left;
reaching ≤19 is implausible.

Caveats before final refutation: (1) early stop at 45% of budget truncates the cosine-schedule
anneal that likely drove N0010's E26 dip; (2) batch_size 4 halves throughput AND changes
gradient noise — a known confound vs parent; (3) val curve is noisy (E05/E06 spiked to 36)
suggesting higher-variance optimization at 1369 tokens with lr=1e-3 possibly too hot for the
longer token sequence through the adapter.

Code/config audit: model.py is a verbatim N0010 clone (correct isolation); config matches spec
(518, bs4, 40ep, lr1e-3). No OOM, no instability; engine handled 37×37 density correctly.
Execution was clean — the negative result is informative, not a bug artifact.

## actionable_feedback
- Resolve H0021 as CONTRADICTED on current evidence: same-epoch deficit (~1.5 MAE), ratio 3.66,
  no tail improvement. Do not pursue >392px on this frozen stack without an optimization fix.
- If retrying high-res, confound-fix first: lr ↓ to 5e-4 (or warmup) + keep batch 8 via grad
  accumulation, and run ≥30ep so the schedule anneals. Otherwise the lever stays untested-clean.
- Cheaper direction per data: N0010 overfits after E26 (train loss keeps dropping, val flat) —
  capacity/data augmentation (N0013) attacks the actual bottleneck, resolution does not.
- Record wall-clock cost: 518px doubles epoch time for negative return — deprioritize scale-up
  levers under this budget.

## hypothesis_updates
- hyp_id: H0021, evidence_type: contradicts, strength: 0.85, reasoning: Best 26.03 @E18 trails
  parent's same-stage trajectory (best 24.32 @E15) and RMSE/MAE 3.66 ≈ parent's 3.63 — the
  finer token grid produced neither accuracy nor tail gains; MAE≤19 unreachable in remaining
  22 epochs given flat-to-noisy convergence.
- hyp_id: H0017, evidence_type: neutral, strength: 0.3, reasoning: Multi-layer gated taps ran
  without failure at 518 (no OOM/instability), but the early stop plus confounded batch-size
  change prevents attributing the deficit to or from the multi-layer substrate itself.
- hyp_id: H0019, evidence_type: neutral, strength: 0.2, reasoning: Already refuted by N0011;
  N0012's noisy high-token-count optimization weakly reinforces that scalar-gate architecture
  is not the binding constraint, but adds no direct evidence about per-token gating.

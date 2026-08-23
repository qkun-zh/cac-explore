# synthesis.md — N0012_dino_highres518

## Verdict
**Confounded-truncated, weak negative.** Run killed at E18/40 by TAU_MAX_MIN=30min timeout (~63s/ep vs parent's ~31s). Best MAE 26.03 (still monotonically improving: 27.65→26.03 over last 3 ep), RMSE 95.34, RMSE/MAE 3.66 ≥ 3.63 refutation bar. Trails champion N0010 (21.53 @E26/40) by +4.5, and trails the parent's matched-epoch trajectory by ~1.2–1.7 MAE despite 2× steps/epoch (batch 4 vs 8). Primary confound: cosine schedule built for 40ep — at E18 lr≈56% of peak, low-lr refinement phase (which produced N0010's E26 dip) never ran. Secondary: batch 4 halves throughput / changes gradient noise. NOT a clean refutation of resolution; do not book as fully refuted.

## Quality Gate (7 dims)
- mechanistic: PASS — single-lever clone isolates resolution; pos-embed exact at native 518 (exonerated).
- scoped: PASS — scope explicit (frozen DINOv2-S stack, FSC147, ≤32M).
- predictive: PARTIAL — MAE≤19 prediction clearly off-track; tail-reduction mechanism (ratio unchanged 3.66 vs 3.63) contradicted so far.
- falsifiable: PASS — pre-registered disproof criteria fired conditionally on truncation caveat.
- novel: PASS (ablation-novelty as declared) — cheapest scale-up lever test under budget.
- transferable: PARTIAL — lesson "resolution costs 2× wall-clock here; schedule completion matters" transfers to all high-res variants.
- actionable: PASS — concrete retry levers: full schedule (≥55min or 22ep re-anchored cosine), batch 8 via grad-accum×2, lr↓/warmup for longer token sequences.

## Deduplicated Updates
- H0021 contradicts w=0.45 (quantitative said 0.85, causal/qualitative argued truncated-run confound → resolved to moderate 0.45). Truncated-run confound noted explicitly; may still work with full schedule + lr tuning.
- H0017 supports w=0.35 (causal 0.35 support, qualitative neutral-support, quantitative neutral 0.3 → converged on modest support: stack trains stably at 1369 tokens, no OOM/divergence).
- Drop quantitative's H0019 note (already refuted by N0011; no new evidence).

## Booking List
1. evidence H0021 contradicts strength 0.45 — best 26.03>21.53, ratio 3.66≥3.63 fired, but E18/40 truncation with MAE falling & lr@56% peak confounds attribution [quantitative+causal].
2. evidence H0017 supports strength 0.35 — multi-tap gated stack stable/trainable at native 518px/1369 tokens through E18 [causal+qualitative].

## tested_hypotheses
["H0021", "H0017"]

## Scores
quality ≈ 0.35 (truncated run: clean execution but incomplete schedule limits evidential value) · availability: high (23.11M ≤32M, no OOM, 2× epoch cost) · score = quality·avail ≈ 0.35 · recommended tree status: "synthesized" with mae=26.03, rmse=95.34, epochs_done=18/40, verdict=confounded_truncated_weak_negative.

## Next-lever guidance
Retry only if cheap (22ep re-anchored cosine + bs8 grad-accum); else abandon >392px on this stack and pivot to N0013 augmentation / H0022 tail-reweight per STATE.md.

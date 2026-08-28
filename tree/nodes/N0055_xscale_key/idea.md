# Idea — N0055_xscale_key (parent: N0054_xscale_exemplar, frozen)

Adds a pluggable **XScale-Key** to the N0054 base (use_gca=True, use_ddca=False, use_xscale=True → 19.647).

## Change (structural, FROZEN, optimizer unchanged)
**XScale-Key (H0076, ~0.098M).** Exemplar-interface refinement gated by `use_xscale_key` inside
ExemplarEncoder (same pluggable precedent as XScale/use_ddca). On N0054's existing XScale fusion, an
ADDITIONAL coarse prototype `e_coarse = xk_proj(GAP(roi@3x3))` (B,K,256) is produced and supplied as a
**separate second exemplar key** to the Condenser's cross-attention: keys = [fine_prototypes ; coarse_prototypes]
(2K keys). e_mean for GCA stays on the fine prototypes only. Token/key dim unchanged; single-switch.

## Why (grounding)
- N0054's XScale (coarse summary FUSED into the pooled fine exemplar vector) is the validated win over
  GCA-only. The mechanism: coarse exemplar context improves cross-scale matching in the condenser.
- But fusion forces a SINGLE pre-mixed prototype. Supplying fine+coarse as TWO SEPARATE keys lets the
  condenser learn per-query which scale matters, a strictly richer hypothesis with the same ~0 params
  cost and no additional density/feature path changes.
- Keeps to proven interface: reads only h3 (backbone feature) + bboxes; outputs exemplar embeddings.
  No feature gating, no density bias, no optimizer/recipe change.

## Hypothesis
**H0076** IF [XScale-Key separate coarse exemplar keys] IN [frozen-N0054 base (GCA+XScale), use_ddca=False],
THEN [val MAE <19.647 (beat N0054)] BECAUSE [per-query fine-vs-coarse exemplar key weighting improves
cross-scale exemplar matching, lowering false-negative detections]. DISPROVED IF [val MAE ≥19.65].

## Gates
- R1 smoke: stub backbone, use_gca/use_xscale/use_xscale_key on; params ≤32M; density (B,1,96,96);
  loss finite & drops. Verify use_xscale_key=False removes the 2nd key (single-switch, = N0054).
- R2 30ep @384 frozen recipe. Compare vs N0054 (19.647). Early-stop if ep16+ ≥+1.5 worse than parent best.
- R3 pluggability: toggling use_xscale_key touches only ExemplarEncoder + key concat; no other coupling.

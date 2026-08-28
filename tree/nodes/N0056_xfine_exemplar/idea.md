# Idea — N0056_xfine_exemplar (parent: N0054_xscale_exemplar, frozen)

Adds a pluggable **XFine** to the N0054 base (use_gca=True, use_ddca=False, use_xscale=True → 19.647).
Motivated directly by the N0055 negative result: SEPARATE 2nd condenser keys (2K) degraded (+1.19),
so a different exemplar-interface direction is taken — enrich the SAME fused single prototype instead
of adding keys.

## Change (structural, FROZEN, optimizer unchanged)
**XFine (H0077, ~0.10M).** Exemplar-interface refinement gated by `use_xfine` inside ExemplarEncoder.
KEEPS N0054's exact winning XScale mechanism (coarse h3 ROI → GAP → proj → add into exemplar token).
Adds a SECOND additive scale-summary: from the FINER backbone feature h2 (1/8, 192ch), ROI-align the
same exemplar boxes at 7x7, GAP to a 192-dim summary, project to d_model, and ADD into the exemplar
token (identical fusion recipe as XScale). Still ONE prototype vector consumed by the Condenser (key
structure unchanged → no N0055 dilution). Reads only h2 (backbone feature) + bboxes.

## Why (grounding)
- N0055 proved the condenser wants a single fused prototype, not extra keys (attention dilution).
- N0054's fused-coarse-summary win is the validated density-interface lever. XFine applies the SAME
  proven additive-fusion recipe at the OTHER scale extreme (finer 1/8 vs existing 1/16), giving the
  condenser a sharper exemplar appearance view that the coarse-to-fine matching currently lacks.
- Mechanism-faithful "more of what worked": no key-structure change, no density bias, no feature gate.

## Hypothesis
**H0077** IF [XFine fused fine (h2) exemplar summary] IN [frozen-N0054 base (GCA+XScale), use_ddca=False],
THEN [val MAE <19.647 (beat N0054)] BECAUSE [a second, higher-resolution exemplar appearance summary
fused into the single prototype sharpens cross-scale exemplar matching and cuts false negatives].
DISPROVED IF [val MAE ≥19.65].

## Gates
- R1 smoke: stub backbone, use_gca/use_xscale/use_xfine on; params ≤32M; density (B,1,96,96);
  finite & drops. Verify use_xfine=False removes the fine branch (single-switch, = N0054).
- R2 30ep @384 frozen recipe. Compare vs N0054 (19.647). Early-stop if ep16+ ≥+1.5 worse than parent.
- R3 pluggability: toggling use_xfine touches only ExemplarEncoder (optional feat_fine= h2) + CountingHead
  pass-through; no coupling to other modules.

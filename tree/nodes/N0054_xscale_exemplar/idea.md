# Idea — N0054_xscale_exemplar (parent: N0052_pluggable_trunk, frozen)

Adds a pluggable **XScale (Multi-Scale Exemplar)** refinement to the pluggable base, with use_ddca=False
(per ablation) and use_gca=True (the only surviving aux). Base reproduces GCA-only (N0051 20.599).

## Change (structural, FROZEN, optimizer unchanged)
**XScale (H0075, ~0.098M).** Exemplar-side refinement gated by `use_xscale` inside ExemplarEncoder
(same pluggable precedent as use_ddca inside FineFuser). Each exemplar ROI is ALSO pooled at a second
coarse scale (xscale_size=3) alongside the base roi_size=7, GAP'd to a per-exemplar global summary
(B*K,in_dim), projected to d_model (xproj Linear), and ADDITIVELY fused into the attended exemplar token
BEFORE the Condenser. Token count/length unchanged (condenser interface preserved).

## Why (grounding)
- All prior failed additions (SALF, FILM, cross-attn feature path, bg-token, MoE, DDCA, RGA) fell into
  two camps: (a) add learned gating/modulation to the FEATURE path, or (b) add trainable bias to the
  DENSITY output. Both add optimization burden to an already-near-optimal condenser under 30ep.
- XScale is the FIRST to enrich the EXEMPLAR embedding itself (the one interface the condenser reads,
  explicitly allowed by §5.14), deterministically, with ~0 params, no feature gating, no density bias.
- It targets cross-scale invariance of exemplar appearance: a small exemplar and large exemplar of the
  same object class should yield similar query features; the coarse-scale global summary makes the
  token robust to object scale variation — a known FSC147 error source.
- Parent N0051 (GCA-only, no xscale) = 20.599 is the clean ablation control.

## Hypothesis
**H0075** IF [XScale multi-scale exemplar-summary fusion] IN [pluggable frozen-N0052 base, use_ddca=False,
use_gca=True], THEN [val MAE <20.599 (beat GCA-only)] BECAUSE [a deterministic coarse-scale exemplar
summary makes the condenser's exemplar query invariant to object scale, lowering false-negative
detections]. DISPROVED IF [val MAE ≥20.6 (no better than parent GCA-only)].

## Gates
- R1 smoke: stub backbone, use_gca=True/use_ddca=False/use_xscale=True builds; params ≤32M; density
  (B,1,96,96); loss finite & drops. Verify use_xscale=False removes xproj (single-switch).
- R2 30ep @384 frozen recipe. Compare vs N0051 (20.599). Early-stop if ep16+ ≥+1.5 worse than parent best 20.599.
- R3 pluggability: toggling use_xscale touches only ExemplarEncoder; no coupling to other components.

# Idea — N0052_pluggable_trunk (parent: N0036_gca_ddca, frozen)

Behavior-identical refactor to satisfy §5.14 pluggability. Bundles the strongly-coupled trunk
(FineFuser + ExemplarEncoder + Condenser + DensityDecoder) into ONE pluggable module (CountingHead)
with a clean interface (h2, h3, boxes) -> {density, fine, e_mean}. GCA and DDCA become independent
single-switch components gated by config flags. No component reads another's internals.

## Change (structural, FROZEN, optimizer unchanged)
**A. Trunk -> one CountingHead module** (H0070). Internals unchanged; only reorganized. Exposes
`fine` + `e_mean` as shared-interface outputs (not couplings).
**B. DDCA -> independent switch** (H0071). Config `use_ddca` toggles the dilated dw branch inside
FineFuser. Off = `refine(f)+f`; on = `+ctx(f)`. Same as N0051 ablation, now flag-gated.
**C. GCA -> independent switch** (H0072). Config `use_gca` toggles the aux count head (reads
GAP(fine)+e_mean, injects bias). Same as N0048, now flag-gated.

## Why
Future components bolt onto the frozen backbone via the two shared interfaces (backbone features,
exemplar embeddings). Any new component is single-switch ablatable by removing one config flag.
This is the new champion base enabling clean A/B on every downstream insertion.

## Hypothesis
**H0070** IF [trunk refactored into one pluggable CountingHead] IN [same frozen 384 recipe],
THEN [val MAE ≈ N0036 (20.49 ±0.3)] BECAUSE [only code reorganization, no math change].
DISPROVED IF [val MAE ≥21.0].

## Gates
- R1 smoke: stub backbone, use_gca/use_ddca all combinations build; param ≤32M; density (B,1,96,96);
  loss finite & drops.
- R2 30ep @384 frozen recipe (default flags: use_gca=True, use_ddca=True) -> expect ~20.49.
- R3 pluggability verified by construction: setting use_gca=False or use_ddca=False requires touching
  NO other code (config-only single-switch). Early-stop if ep16+ ≥+1.5 worse than N0036 20.49.

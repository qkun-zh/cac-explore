# Idea — N0051_gca_only (parent: N0036_gca_ddca, frozen)

A-alone ablation per R2 protocol (ablate B off). Replace the missing R2 cell: N0036 kept GCA+DDCA (A+B, 20.49); N0048 removed GCA kept DDCA (B-alone, 22.197). Missing: A-alone = keep GCA, remove DDCA.

## Change (structural, FROZEN, optimizer unchanged)
**Remove DDCA (H0073, -0.001M).** In FineFuser, drop the parallel `ctx = dw3x3_d2(f)` branch; forward becomes `refine(f)+f` instead of `refine(f)+f+ctx(f)`. Keep GCA head intact.

## Why this completes R2
- A+B (N0036) = 20.49
- B-alone (N0048_no_gca) = 22.197 → GCA worth ~1.7
- A-alone (this) → quantize DDCA: if A-alone ≈20.5 then DDCA≈0 (noise); if A-alone >20.9 then DDCA worth ~0.4+

## Hypothesis
**H0073** IF [remove DDCA, keep GCA] IN [frozen ConvNeXt-Tiny 384 recipe], THEN [val MAE ≈20.5] BECAUSE [zero-init ctx branch contributes little]. DISPROVED IF [val MAE ≥20.9 → DDCA non-trivial].

## Gates
- R1 30ep @384 frozen AdamW 1e-3 wd0.05 cosine bs16 AMP. Early-stop if ep16+ ≥+1.5 worse than N0036 20.49.
- Pluggability: DDCA is a single removable branch — toggling it off touches nothing else. Satisfies §5.14.

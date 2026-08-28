# Idea — N0053_rga_aux (parent: N0052_pluggable_trunk, frozen)

Adds a pluggable **RGA (Regionalized Count Auxiliary)** head to the pluggable base, with use_ddca=False
(per ablation: DDCA is harmful, +1.8). Base = N0052 GCA-only-equivalent (reproducible ~20.6).

## Change (structural, FROZEN, optimizer unchanged)
**RGA (H0074, +0.1M).** Independent aux head (`use_rga` flag) reading ONLY shared interfaces:
`GAP-per-region(fine)` + `e_mean` (exemplar embedding). Adaptive-pools fine into grid×grid regions
(default 4×4=16), per-region vector = [regional_GAP ‖ e_mean] → MLP → regional log-count n_rga.
Injects a regional bias `0.02·n_rga/region_area` (nearest-upsampled) into density — EXACTLY like GCA's
proven bias trick, but at regional granularity. Engine's per-pixel density MSE then supervises regional
structure. No inference-only change to the density head path; no feature modulation (avoids DDCA trap).

## Why (grounding)
- GCA (global log-count aux) is the ONLY component that survived ablation (worth ~1.6). RGA is its
  natural generalization: variance-stabilized (log) count supervision, spatially localized to the
  RMSE-dominated tail (RMSE 75-83 vs MAE ~20 => >500-object images dominate).
- Fully pluggable: reads only fine+e_mean; `use_rga` single-switch; +0.1M params; trains cheap (parallel aux).
- Prior spatial modulators (SALF/FILM/cross-attn/bg-token) all failed because they ADD learned gating
  to the feature path; RGA instead shapes gradients via output bias, per GCA's winning recipe.

## Hypothesis
**H0074** IF [RGA regional log-count aux bias] IN [pluggable frozen-N0052 base, use_ddca=False],
THEN [val MAE ≤20.0] BECAUSE [regional variance-stabilized count supervision reduces tail/RMSE error
that global GCA cannot localize]. DISPROVED IF [val MAE ≥20.7 (no better than N0051 20.599)].

## Gates
- R1 smoke: stub backbone, use_gca=True/use_ddca=False/use_rga=True builds; params ≤32M; density (B,1,96,96);
  loss finite & drops. Verify use_rga=False removes the module (single-switch).
- R2 30ep @384 frozen recipe. Compare vs N0051 (20.599) and N0052 (22.410). Early-stop if ep16+ ≥+1.5 worse than parent best 20.599.
- R3 pluggability by construction: toggling use_rga touches no other code.

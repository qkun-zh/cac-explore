# Idea — N0063_large_backbone (parent: N0054_xscale_exemplar, User-Guided backbone swap)

## Header
- **Parent**: N0054 (GCA+XScale) 19.647 / 31.32M.
- **Directive**: User-Guided swap to larger backbone to test if 32M cap is the bottleneck (User: "通过换上更大的backbone，来看看是不是参数量限制").
- **Deviation**: ALLOWS total >32M (max_params 60M) — diagnostic run to isolate backbone capacity vs head design. Logged as User-Guided deviation per AGENTS.md §1.

## Change (single switch `use_large_backbone`, frozen, no recipe change)
Replace HF `dinov3-convnext-tiny` (backbone ~27M, hs 96/192/384/768, hs_map 2,3 → 192@1/8 + 384@1/16) with **timm `convnext_small.in12k` (backbone ~49.5M, features [96,192,384,768] same dims, deeper)** via `timm.create_model(..., pretrained=True, features_only=True, out_indices=(1,2))`. Head (FineFuser 128, XScale, Condenser, GCA, DensityDecoder) untouched, still reads h2=192@1/8 + h3=384@1/16. Frozen (no grad), same /255 input (no extra norm) to isolate capacity, not preprocessing. `use_large_backbone=False` = exact champion.

- New backbone dims identical, so head param count unchanged (+0 head). Total 53.0M (49.5M backbone + 3.5M head) vs 31.32M → +21.7M.
- Single-switch for ablation; engine loss/recipe untouched.

## Why
If MAE drops with larger frozen backbone under identical head/recipe, the 32M cap is load-bearing and headroom exists in representation, not head design; if MAE stays ~19.6 or worsens, the bottleneck is head/training, not backbone capacity — closing the "larger backbone helps" hypothesis and confirming frozen-head saturation.

## Hypothesis
**H0091** IF [frozen backbone is swapped from dinov3-convnext-tiny (27M) to convnext_small.in12k (49.5M, same channel dims, deeper) IN frozen N0054 head (GCA+XScale, 30ep @384, no recipe change)] THEN [best val MAE < 19.647] BECAUSE [larger frozen representation supplies richer count-semantic features the pluggable head can exploit, and the 32M cap was the bottleneck]. DISPROVED IF best val MAE ≥ 19.647.

## Gates
- R1 smoke: total ~53M (>32 but allowed per deviation), head 3.5M, density (B,1,96,96) finite.
- R2 30ep: **CONFIRM** <19.45 → backbone capacity helps (deviation justified); **FAIL** ≥19.647 → H0091 refuted, param cap not bottleneck; **WEAK** 19.45-19.647 marginal.

# N0066_midft_intermediate — Mid-layer FT on intermediate readout (hs 2+3)

## Title
Fine-tune DINOv3-ConvNeXt-Tiny middle stages (1,2) at 0.1× LR with intermediate readout hs_map=(2,3) to test if mid-layer adaptation beats frozen N0054.

## Motivation & Intuition
N0054 (19.647, frozen) uses hs(2,3)=192/384 ch at 1/8+1/16. All 30ep failures since N0054 were head-only. Research_direction allows narrow FT (N0021: partial FT 20.44<21.53). Mid stages (1,2) encode count-relevant texture/part patterns; early stem (stage0) is general. Differential LR (backbone 1e-4 vs head 1e-3) prevents drift while adapting. This tests the user directive's core: intermediate output + mid FT vs frozen final.

## Architecture Spec
- **core_ideas**: selective mid-stage unfreeze (stages 1,2), intermediate hs_map (2,3), differential LR via param_groups, keep GCA+XScale head unchanged, frozen stem/stage0/stage3.
- **core_blocks**: Backbone(hf_model=facebook/dinov3-convnext-tiny, hs_map=(2,3), tune_stages=[1,2], backbone_lr=1e-4) → FineFuser(384,192) → ExemplarEncoder(in_dim=384) → Condenser → DensityDecoder → GCA.
- **network_structure**: img 384 → Backbone hs[2](192,48×48)+hs[3](384,24×24) → fuser→96×96 → exemplar on h3 → condenser → decoder → density 96×96 + GCA bias.
- **tunable_aspects**: tune_stages list, backbone_lr factor, hs_map (fixed 2,3 for this node).
- **invariants**: total params ≤32M, engine loss unchanged, single-switch vs N0054 is tune_stages+backbone_lr (head frozen otherwise), XScale+GCA retained.

## Proposed Hypotheses
- **H0093**: IF [backbone middle stages 1 and 2 fine-tuned at 0.1× head LR] IN [DINOv3-ConvNeXt-Tiny with intermediate readout hs_map (2,3) dims (192,384), GCA+XScale, FSC147 30ep @384] THEN [val MAE < 19.647 beats frozen N0054] BECAUSE [middle layers adapt count-relevant texture/shape while early stem stays general; differential LR prevents catastrophic drift]. DISPROVED IF [best val MAE >= 19.65].
- **H0094-part**: IF [final-layer readout hs_map (3,4) replaces intermediate] IN [same mid-FT regime] THEN [MAE worse than intermediate by >=0.5] BECAUSE [final stage is overly semantic/coarse (12×12 @384) collapsing instance separation]. DISPROVED IF [final MAE < intermediate MAE +0.5]. (Tested jointly with N0067.)

## Delta vs Parent
Parent N0054 frozen. Delta: Backbone tune_stages=[1,2] (was []), backbone_lr=1e-4 (new), param_groups() added, train() keeps tuned stages in train mode. hs_map stays (2,3), dims unchanged, head identical, single-switch head+backbone FT ablation.

## Novelty Statement
First mid-layer FT on ConvNeXt-Tiny in frozen-head lineage. Prior FT (N0021) was DINOv2 ViT top-blocks; this is ConvNeXt mid-stages with differential LR and intermediate-vs-final comparison. No prior node tests hs_map+FT coupling.

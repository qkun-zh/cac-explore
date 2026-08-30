# N0067_midft_final — Mid-late FT on final-layer readout (hs 3+4)

## Title
Fine-tune DINOv3-ConvNeXt-Tiny late stages (2,3) at 0.1× LR with final readout hs_map=(3,4) to test if final-layer output is worse than intermediate even when FT.

## Motivation & Intuition
Spatial fidelity vs semantics tradeoff: intermediate (48×48+24×24) retains instance texture; final (24×24+12×12, 768ch) is coarser and more semantic. Counting needs instance separation, not class semantics. Even with FT of its own stages (2,3), final should underperform intermediate (N0066). This is the user directive's second question: is intermediate output more suitable than final for counting?

## Architecture Spec
- **core_ideas**: final readout hs_map (3,4) dims (384,768), late stages 2,3 FT at 0.1×, same GCA+XScale head adapted to 768ch exemplar, differential LR.
- **core_blocks**: Backbone(hf_model=facebook/dinov3-convnext-tiny, hs_map=(3,4), tune_stages=[2,3], backbone_lr=1e-4) → FineFuser(768,384) → ExemplarEncoder(in_dim=768) → Condenser → DensityDecoder → GCA.
- **network_structure**: img 384 → Backbone hs[3](384,24×24)+hs[4](768,12×12) → fuser→96×96 → exemplar on h4 → condenser → decoder → density 96×96 + GCA bias.
- **tunable_aspects**: hs_map (3,4), tune_stages [2,3], backbone_lr.
- **invariants**: total params ≤32M (≈31.5M, +0.2M over N0054), same loss/engine, GCA+XScale retained, single-switch vs N0066 is hs_map+dims+stage shift.

## Proposed Hypotheses
- **H0094**: IF [final-layer readout hs_map (3,4) dims (384,768) with late stages 2 and 3 fine-tuned at 0.1× head LR] IN [DINOv3-ConvNeXt-Tiny, GCA+XScale, FSC147 30ep @384] THEN [val MAE worse than intermediate readout N0066 by >=0.5] BECAUSE [final stage is overly semantic and spatially coarse (12×12 @384) collapsing instance separation; intermediate retains instance-level texture and higher spatial fidelity]. DISPROVED IF [best val MAE < N0066_MAE +0.5 or final MAE <19.65].
- **H0093-counter**: Same as N0066 H0093 but tests if late-stage FT can still beat frozen intermediate — expected to fail, supporting single-slider mid-layer law.

## Delta vs Parent
Parent N0054. Delta: hs_map (2,3)→(3,4), backbone_dims (192,384)→(384,768), tune_stages []→[2,3], backbone_lr 1e-4, param_groups(). Head adapters (fuser ch, exemplar in_dim) scaled to 768ch.

## Novelty Statement
First final-layer readout test in this lineage. Directly contrasts intermediate vs final under matched FT protocol. No prior node evaluated hs_map (3,4) or (4,*).

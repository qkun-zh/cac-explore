# feedback/quantitative.md — N0004_effnet_pyrmatch

## reasoning
Best val MAE 40.37 / RMSE 119.68 @20ep/535s, total 3.65M (6× smaller than N0002/N0003). Beats N0002 (42.05) by only 4% — below H0006's 8% bar; loses to N0003 (34.26). Convergence plateaued ~E14 (40.9→40.37 over last 6 epochs), unlike ViT/ConvNeXt nodes which still descended — small-backbone capacity limit, exactly what H0007 probed: <35 @10ep is DISPROVED (40.37 @20ep).

## actionable_feedback
- Do not scale this branch further at b0 capacity; if retried, use efficientnet_b3/b4 or convnext_small within budget.
- Gate maps were the interesting signal but engine saves no images — a child could dump gate statistics to result.json diagnostics.
- Multi-scale gating gave no decisive edge over single-scale DINOv2 → backbone feature quality dominates mechanism choice at this scale.

## hypothesis_updates
- H0006: neutral, strength 0.25. No same-backbone single-scale ablation ran; cross-node comparison confounded by backbone size (40.37 vs 42.05 with 6× fewer params is mildly positive but <8% bar).
- H0007: contradicts, strength 0.85. ≤12M EffNet reached only 40.37 (>35 even @20ep) — backbone scale IS a bottleneck for matching-style heads on FSC147.

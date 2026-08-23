# feedback/quantitative.md — N0008_dino_xattn

## reasoning
Best val MAE 46.56 @30ep/1063s — WORST DINOv2 node; H0015 (<=25.5) disproved in this configuration. Trajectory is diagnostic: E1-E12 oscillated 176-308 MAE with loss stuck ~60-87, then monotone recovery E13-E30 (159→46.6, still falling at cutoff). This is an optimization failure signature, not a mechanism ceiling: the same decoder reached 34.26 on ConvNeXt features at lr=1e-3, but DINOv2 token magnitudes + norm_first attention at lr=1e-3 without warmup destabilize early training; by the time it stabilized, only ~17 effective epochs remained.

## actionable_feedback
- Retry as N0009 with lr=2.5e-4, dec_layers=1, K=4 (smaller decoder = easier optimization), epochs=30.
- If a second instability appears, add explicit warmup to engine (out of scope for node code).
- Do NOT abandon the merge: recovery slope suggests mechanism works once past the unstable basin.

## hypothesis_updates
- H0015: contradicts, strength 0.55 — AS CONFIGURED. Optimization confound explicitly noted; mechanism-level question remains open.

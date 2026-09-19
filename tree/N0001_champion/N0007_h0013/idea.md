# N0007_h0013 — evolution of N0001_champion

## Hypothesis Set
The executor agent tests the following 1 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0013** — IF we replace the fixed bilinear upsampling that brings the coarse backbone readout to the fused resolution with a learned depth-to-space subpixel block inside the FineFuser IN the FineFuser of the CountingHead THEN final val MAE is at least 0.30 lower than with bilinear upsampling BECAUSE a fixed bilinear kernel low-pass-filters the coarse stage and smears each density head support, while a learned per-channel subpixel upsample preserves the detail that the count sum integrates and tightens localization around density peaks without changing the feature interface or adding meaningful parameters. DISPROVED IF final val MAE is not at least 0.30 lower than with bilinear upsampling
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
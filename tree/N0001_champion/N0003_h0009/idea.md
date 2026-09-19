# N0003_h0009 — evolution of N0001_champion

## Hypothesis Set
The executor agent tests the following 1 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0009** — IF we pool the fine feature map globally and emit one scalar gating coefficient that reweights the exemplar tokens per feature channel IN the Condenser of the CountingHead THEN final val MAE improves by at least 0.30 over the unconditional condenser BECAUSE a per-exemplar channel-gain lets the decoder express count-dependent attention that a fixed affine readout cannot, steering the condenser toward scale-consistent exemplar weights. DISPROVED IF val MAE is not at least 0.30 lower than the unconditional condenser
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
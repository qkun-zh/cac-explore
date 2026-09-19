# N0009_h0015 — evolution of N0001_champion

## Hypothesis Set
The executor agent tests the following 1 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0015** — IF we add a small exemplar-conditioned gating MLP that maps each exemplar embedding to its own per-channel weight vector and multiplies that vector into the channels of that exemplar's token before the token enters the condenser cross-attention IN the CountingHead of the champion counter THEN final val MAE is at least 0.30 lower than the champion without the gate BECAUSE every exemplar token is a pooled ROI whose channels mix object appearance with that box's background and illumination, and one fixed or image-global gate cannot choose a different channel subspace for each exemplar, so a per-exemplar channel mask lets the condenser matching keys carry the object-bearing channels of that specific prototype and down-weight the context channels, which reduces both missed matches for unusual exemplars and false matches on textured background. DISPROVED IF final val MAE is not at least 0.30 lower than the champion N0001 seed of 21.459, that is, final val MAE is above 21.159
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
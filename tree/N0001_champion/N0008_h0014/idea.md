# N0008_h0014 — evolution of N0001_champion

## Hypothesis Set
The executor agent tests the following 1 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0014** — IF we bolt a similarity-aware enhancement block onto the fused fine feature map that projects the fine map and the exemplar tokens into a shared score space, softmax-normalizes each query-exemplar similarity map jointly across the exemplar axis and the spatial axis, and adds a similarity-weighted mix of value-projected exemplar tokens back onto every spatial location as a residual before the map enters the condenser IN the CountingHead of the champion counter THEN final val MAE is at least 0.30 lower than the champion without the enhancement BECAUSE the condenser gives every location the same exemplar attention distribution, so background patches that weakly match any prototype still pass feature evidence to the density head, while a dual-normalized similarity map selects which prototype matches at each location and sharpens the locations where that prototype matches, injecting the match decision into the feature the decoder sums and leaving the residual features untouched where no prototype matches, which cuts false-positive density on clutter. DISPROVED IF final val MAE is not at least 0.30 lower than the champion N0001 seed of 21.459, that is, final val MAE is above 21.159
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
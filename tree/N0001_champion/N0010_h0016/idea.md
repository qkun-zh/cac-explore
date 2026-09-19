# N0010_h0016 — evolution of N0001_champion

## Hypothesis Set
The executor agent tests the following 1 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0016** — IF we add a verify-and-suppress stage that pools the fused fine feature at the top-scoring peaks of the preliminary density map, scores each pooled peak by cosine similarity against an exemplar prototype pooled in the same fine-feature space, and multiplies the density map by a learned-threshold verification gate before the loss IN the CountingHead of the champion counter THEN final val MAE is at least 0.30 lower than the champion without the verifier BECAUSE the decoder is trained only to deposit mass at salient regions, so distractor objects and textured background that match nothing in the exemplars still receive spurious mass that inflates count error, while a verification gate whose offset is learned through the density loss suppresses mass exactly at peaks whose appearance disagrees with the exemplar prototype and leaves verified peaks intact, removing false positives without touching the frozen backbone. DISPROVED IF final val MAE is not at least 0.30 lower than the champion N0001 seed of 21.459, that is, final val MAE is above 21.159
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
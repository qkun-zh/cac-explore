# N0034_h0031 — evolution of N0029_tf_champion

## Hypothesis Set
The executor agent tests the following 1 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0031** — IF a training-free TFCounter baseline with SAM-vit_b box prompts and fusion_type=mean/fusion_ratio=0.5 on FSC147 val under the 12GB VRAM gate (vit_h forbidden), IN the N0029_tf_champion training-free tree, THEN val MAE is at or below 45.0 (a usable external anchor that beats the DEI-only vitl16 41.062 by not being strictly worse than 45), BECAUSE TFCounter is a published training-free SAM-prompt counter that should land in the same ballpark as local CountingDINO ablations while serving as an independent method family reference on the same val split, DISPROVED IF val MAE exceeds 45.0 after the full val pass prints MAE:..., or if the run fails to produce a parseable MAE line under vit_b weights.
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
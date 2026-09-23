# N0035_h0032 — evolution of N0029_tf_champion

## Hypothesis Set
The executor agent tests the following 1 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0032** — IF a training-free TF-CAC baseline with SAM-vit_b box prompts plus DINOv2-vits14 features on FSC147 val under the 12GB VRAM gate (vit_h forbidden), IN the N0029_tf_champion training-free tree, THEN val MAE is at or below 45.0 (same external-anchor bar as TFCounter on the identical split), BECAUSE TF-CAC combines promptable segmentation with frozen DINOv2 density estimation and the paper reports test MAE 12.26 under box prompts, so a local val run with the light vit_b backbone should still be a competitive training-free anchor rather than a failure above 45, DISPROVED IF val MAE exceeds 45.0 after the full val pass, or if DINOv2-vits14 hub load fails offline, or if the run OOMs under the 12GB gate.
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
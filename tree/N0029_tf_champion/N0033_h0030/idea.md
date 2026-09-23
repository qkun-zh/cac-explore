# N0033_h0030 — evolution of N0029_tf_champion

## Hypothesis Set
The executor agent tests the following 1 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0030** — IF a training-free CountingDINO ablation that enables cosine_similarity=true on top of the byte-identical proven full-flag dinov3_vits16 configuration (divide_et_impera/filter_background/ellipse_normalization/ellipse_kernel_cleaning all true, every other ROI-norm/minmax/resize flag unchanged), IN the N0029_tf_champion training-free tree on FSC147 val, THEN val MAE is at or below the seed best 39.696 (cosine path does not regress the proven config), BECAUSE the paper uses cosine-normalized frozen DINO features for matching while the seed full-flag run left cosine_similarity at default-off and still reached 39.696, so enabling the paper-faithful cosine path should either preserve or improve that score rather than push MAE above the proven baseline. DISPROVED IF val MAE exceeds 39.696 after enabling cosine_similarity on the otherwise byte-identical full-flag vits16 config, or if the cosine flag is a silent no-op producing a bit-identical results.csv row to the seed.
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
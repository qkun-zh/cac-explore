# N0036_h0033 — evolution of N0029_tf_champion

## Hypothesis Set
The executor agent tests the following 1 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0033** — IF a training-free CountingDINO full-flag run on dinov3_vitb16 (divide_et_impera/filter_background/ellipse_normalization/ellipse_kernel_cleaning all true, cosine off to match seed proven path, every other ROI-norm/minmax/resize flag byte-identical to the seed) lands between the local proven full-flag scores on FSC147 val with bar at or below 39.696, IN the N0029_tf_champion training-free tree, THEN val MAE is at or below 39.696 (mid backbone does not regress below the proven seed config), BECAUSE vitb16 sits between vits16=39.696 and vitl16=32.531 in capacity so a flag-complete bit-budget-similar run should stay at least as good as the smaller seed while testing whether backbone scale alone moves the full-flag score. DISPROVED IF val MAE exceeds 39.696 after the flag-complete vitb16 run, or if the run OOMs under the 12GB inference gate, or if flags silently differ from the seed proven path.
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
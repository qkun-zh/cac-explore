# N0004_h0010 — evolution of N0001_champion

## Hypothesis Set
The executor agent tests the following 2 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0010** — IF we compute a per-exemplar reliability gate as sigmoid of a linear read-out of each exemplar token and rescale that exemplar embedding before it enters the condenser cross-attention IN the CountingHead of the champion counter THEN final val MAE is at least 0.30 lower than the champion without any exemplar gating BECAUSE a gate that varies between exemplars of the same image lets the condenser down-weight a weakly-localized prototype whose ROI features are dominated by background or occlusion, so attention concentrates on trustworthy prototypes and count error on mixed-quality exemplar scenes falls, while a fixed or image-global gate cannot make this per-exemplar choice. DISPROVED IF final val MAE is not at least 0.30 lower than the champion with the gate removed
   → current conf 0.500 — one test event

2. **H0009** — IF we pool the fine feature map globally and emit one scalar gating coefficient that reweights the exemplar tokens per feature channel IN the Condenser of the CountingHead THEN final val MAE improves by at least 0.30 over the unconditional condenser BECAUSE a per-exemplar channel-gain lets the decoder express count-dependent attention that a fixed affine readout cannot, steering the condenser toward scale-consistent exemplar weights. DISPROVED IF val MAE is not at least 0.30 lower than the unconditional condenser
   → current conf 0.400 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
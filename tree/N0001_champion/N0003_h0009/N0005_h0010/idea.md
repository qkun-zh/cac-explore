# N0005_h0010 — evolution of N0003_h0009

## Hypothesis Set
The executor agent tests the following 2 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0010** — IF we compute a per-exemplar reliability gate as sigmoid of a linear read-out of each exemplar token and rescale that exemplar embedding before it enters the condenser cross-attention IN the CountingHead of the champion counter THEN final val MAE is at least 0.30 lower than the champion without any exemplar gating BECAUSE a gate that varies between exemplars of the same image lets the condenser down-weight a weakly-localized prototype whose ROI features are dominated by background or occlusion, so attention concentrates on trustworthy prototypes and count error on mixed-quality exemplar scenes falls, while a fixed or image-global gate cannot make this per-exemplar choice. DISPROVED IF final val MAE is not at least 0.30 lower than the champion with the gate removed
   → current conf 0.500 — one test event

2. **H0011** — IF we keep the same 33,024-parameter exemplar channel gate that the gated child tests but sever its conditioning by feeding the gate a fixed frozen random vector of the same shape instead of the pooled fine features IN the CountingHead of the gated champion THEN final val MAE is at least 0.30 worse than the content-conditioned gate result of 22.608 BECAUSE an equal-capacity gate whose input carries no information can only absorb into the condenser projections as a constant rescale, so if it reproduces the conditioned gate score then the gate benefit is inert capacity not feature conditioning, and the two runs separate mechanism from padding. DISPROVED IF final val MAE with the frozen-random gate is within 0.30 of the content-conditioned gate result of 22.608
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
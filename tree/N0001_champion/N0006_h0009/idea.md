# N0006_h0009 — evolution of N0001_champion

## Hypothesis Set
The executor agent tests the following 3 hypotheses. Test each exactly ONCE,
in one job, minimizing wall-clock. Run the training, observe the val metrics, then
record the outcome in the ledger and in feedback/notes.md.
1. **H0009** — IF we pool the fine feature map globally and emit one scalar gating coefficient that reweights the exemplar tokens per feature channel IN the Condenser of the CountingHead THEN final val MAE improves by at least 0.30 over the unconditional condenser BECAUSE a per-exemplar channel-gain lets the decoder express count-dependent attention that a fixed affine readout cannot, steering the condenser toward scale-consistent exemplar weights. DISPROVED IF val MAE is not at least 0.30 lower than the unconditional condenser
   → current conf 0.320 — one test event

2. **H0011** — IF we keep the same 33,024-parameter exemplar channel gate that the gated child tests but sever its conditioning by feeding the gate a fixed frozen random vector of the same shape instead of the pooled fine features IN the CountingHead of the gated champion THEN final val MAE is at least 0.30 worse than the content-conditioned gate result of 22.608 BECAUSE an equal-capacity gate whose input carries no information can only absorb into the condenser projections as a constant rescale, so if it reproduces the conditioned gate score then the gate benefit is inert capacity not feature conditioning, and the two runs separate mechanism from padding. DISPROVED IF final val MAE with the frozen-random gate is within 0.30 of the content-conditioned gate result of 22.608
   → current conf 0.500 — one test event

3. **H0012** — IF we condition the condenser cross-attention temperature on the image global count prior computed by a small MLP over the pooled fine feature and the mean exemplar embedding so that dense scenes attend broadly over prototypes and sparse scenes attend sharply IN the Condenser of the CountingHead THEN final val MAE is at least 0.40 lower than the fixed-temperature condenser BECAUSE a single attention temperature cannot serve both crowd regimes, sharp attention overweights the one best-matching prototype and undercounts dense clusters while flat attention spreads mass over irrelevant prototypes and overcounts sparse scenes, so a count-scaled temperature reduces error on both tails that the fixed condenser leaves uncorrected. DISPROVED IF final val MAE is not at least 0.40 lower than the champion with the fixed-temperature condenser
   → current conf 0.500 — one test event

## Procedure
1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.
2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.
3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.
4. When done: record evidence in the ledger + write feedback/notes.md + result.json.
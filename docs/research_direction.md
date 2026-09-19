# Research Direction (mission · locked per hard rule)

**Mission**: ≤32M total params · same-parameter-class SOTA MAE on FSC147 **test**.

**Standing regime (user directive, 2026-08-30)**: backbone frozen, head-only
training. Innovation lives in the pluggable head — components bolt onto the
frozen intermediate features hs(2,3) + exemplar embedding only (single-switch
ablatable; see AGENTS.md hard rule §5.14 equivalent). Backbone release /
unfreeze is proven net-harmful (H0005 refuted) and out of scope.

**Benchmark protocol**: FSC147 VarV2 384px. val MAE drives selection; test is
reported once per node and compared only within the same parameter class.
Early-stop bar at ep16+: same-epoch ≥ +1.5 worse than parent best.

**Evolution budget**: per node, wall-clock budget τ_max = 1800s hard ceiling;
≤3 coding fix retries; K_SYNTH = 2 new hypotheses per node.

**Selection constants** (fixed by repo AGENTS.md): λ_acc=0.85, λ_parent=0.60,
η=0.20, K_HYPO=2, conf boundaries 0.75 / 0.25. Changes require a journal entry.

## Frontier (from N0001 synthesis: open questions worth spending evolution on)
1. **Resolution / multires** (392/518; recorded, never proven) — candidate for a
   first open-branch hypothesis.
2. **Training-dynamics variants** — SWA, tail_reweight, dual_res_eval, longer
   epochs, EMA; config-off but never actually tested.
3. **New pluggable head parts** — anything attachable above frozen features +
   exemplar embedding that does NOT duplicate the refuted families (DDCA-dilated,
   RGA/extra spatial summaries, aggregation swaps, final-layer readout, unfreeze).
4. **Condenser / decoder depth scaling** under the 32M budget.

## Settled (will not re-litigate)
GCA+, XScale+, frozen hs(2,3) readout, cross-attn condenser kept.
DDCA−, RGA/extras−, unfreeze−, final-layer readout− excluded.
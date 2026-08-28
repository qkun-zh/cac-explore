# Failure analysis 0827 v2
N0036 GCA+DDCA 20.49 is local optimum. All plugins degraded:
- FILM, cross-attn, exemplar attn, PPC all +2-3
- bg token E1 34.6 vs N0036 E1 ? but best 29.5 still +9
Hypothesis: Condenser is already optimal, any extra gating adds optimization burden under 30ep + frozen backbone.
Next: test DDCA alone (N0048) and iterative (N0045) which are minimal.

--- append 2026-08-28 (locked) ---
Exemplar-interface enrichment is EXHAUSTED at N0054 (GCA + single fused coarse XScale).
- N0055 separate 2nd coarse key (2K condenser keys): 20.835 (+1.19) — attention dilution.
- N0056 fused fine h2 exemplar summary into same prototype: 24.313 ES@17 (+3.06) — extra scale entropy.
Both degraded under 30ep + frozen backbone, mirroring the earlier density-side/modulator negatives.
Root cause: the optimizer ~already saturated the head capacity budget at N0054; ANY added exemplar
info (keys OR extra scale) only adds fitting burden without a trainable-density-path payoff.
N0054 (19.647) = sharp local optimum of the exemplar-embedding interface = the frozen deliverable.

--- append 2026-08-28 (locked, fully triangulated) ---
N0057 condenser REPLACEMENT (cosine-sim matcher, no learned MHA/FFN) = 21.076 (+1.43 vs N0054).
The matcher converged initially (peaked E15) but never had N0054's late drop (E18-27 → 19.65).
This proves the cross-attn condenser's capacity is genuinely load-bearing for late-training
exemplar matching refinement, not just a wasteful overfit.

FULL VERDICT (exhaustive pluggable ablation on frozen ConvNeXt Tiny, 30ep @384):
- Density-side: DDCA, RGA, SALF, FILM, cross-attn, MoE, bg-token — all degrade (~1-3 worse)
- Exemplar-side: XScale-Key (separate 2nd key, +1.19), XFine (fused h2, +3.06 ES) — degrade
- Condenser: cosine-sim matcher REPLACEMENT — degrades (+1.43)
- EXHAUSTED: N0054 GCA+XScale = the frozen pluggable optimum at 19.647

The head is a tightly-matched system under frozen 30ep: neither the condenser, the exemplar
interface, nor the density path can be additively improved or simplified without regression.
N0054 is the deliverable.

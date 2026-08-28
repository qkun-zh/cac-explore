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

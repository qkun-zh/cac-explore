# Failure analysis 0827 v2
N0036 GCA+DDCA 20.49 is local optimum. All plugins degraded:
- FILM, cross-attn, exemplar attn, PPC all +2-3
- bg token E1 34.6 vs N0036 E1 ? but best 29.5 still +9
Hypothesis: Condenser is already optimal, any extra gating adds optimization burden under 30ep + frozen backbone.
Next: test DDCA alone (N0048) and iterative (N0045) which are minimal.

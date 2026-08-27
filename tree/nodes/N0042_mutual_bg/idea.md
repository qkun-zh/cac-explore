# Idea — N0042_mutual_bg (parent: N0036_gca_ddca, frozen)
Mutual-aware early interaction (MAFEA) + background token (TBD) + 4D similarity spirit (SSD). Instead of Condenser-only late fusion, add zero-init mutual cross-attention before FineFuser.

## Change (structural, FROZEN, optimizer unchanged)
**Mutual-bg + zero-init cross (H0062, +0.25M).** h3↔exemplar mutual attn with 1 learnable bg token, zero-init residual. Condenser still active. GCA+DDCA kept.

## Hypothesis
**H0062** IF mutual-bg THEN MAE ≤19.7 BECAUSE early target-aware features reduce confusion. DISPROVED IF ≥20.5.

# Idea — N0039_exemplar_attn (parent: N0036_gca_ddca, frozen)
Best so far: N0036 GCA+DDCA = 20.49. Key gap: exemplars only interact via Condenser at 96x96. Add exemplar→density cross-attention at decoder output to spatially modulate density predictions.

## Change (structural, FROZEN, optimizer unchanged)
**Exemplar-aware spatial attention (H0059, +0.2M).** After Decoder0, compute exemplar query (B,K,d_out) × density key (B,Hf*Wf,d_out) → spatial attention map (B,1,Hf,Wf). Apply sigmoid gate to dens0. Same GCA+DDCA base from N0036.

## Hypothesis
**H0059** IF exemplar spatial attention THEN MAE ≤19.8 BECAUSE direct spatial guidance from exemplars. DISPROVED IF ≥20.5.

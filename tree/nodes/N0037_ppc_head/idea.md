# Idea — N0037_ppc_head (parent: N0026_res_sweep, frozen)
Maths lens: point process factorization. GT density = N·p(x) where p integrates to 1. MSE entangles scale⊗shape (Fisher non-diagonal). Factorize: shape branch predicts p(x) (sum=1) via Hellinger, count branch predicts N via log.

## Change (structural, FROZEN optimizer unchanged)
**PPC-Head (H0053, +0.06M).** Keep FineFuser+Condenser frozen, replace Decoder 192->1 with:
- shape head: Cond 64 -> Conv 64->32 -> Conv 32->1 -> softplus -> normalize sum=1 -> p_hat
- count head: GAP(fine 128)+e_mean256 -> MLP 384->64->1 -> softplus N_hat
- density = N_hat * p_hat
Loss remains engine MSE+Hellinger? Engine computes MSE vs gt_d, but we can make p_hat normalized so MSE on density still works; p normalization ensures Campbell consistency. Trainable ~0.06M. Total 28.80M <32M.

## Hypothesis
**H0053** IF factorized N·p head THEN val MAE ≤19.3 BECAUSE orthogonal shape/scale removes heteroscedastic bias. DISPROVED IF MAE ≥20.1.

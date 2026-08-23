# T0024 — Execute N0011_dino_pertok_gate_huber (full training)

Node: N0011_dino_pertok_gate_huber
Parent: N0010_dino_multilayer_long (MAE 21.53)
Hypotheses: H0019 (per-token gate), H0020 (Huber loss)

## Changes from parent
- Scalar layer gate logits → per-token gate MLP (768→64→2, ~0.03M params)
- MSE loss → Huber(delta=5.0)
- count_w reverts to 0.3 (from 1.0)

## Config
- input_size=392, epochs=40, lr=1e-3, count_w=0.3
- loss_function='huber', huber_delta=5.0, gate_mlp_hidden=64
- Expected ~23.16M params

## Criteria
- val MAE < 21.53 (beat parent)
- RMSE/MAE < 3.6 (reduce outlier tail)
- H0019 confirmed if MAE improves >=1.0
- H0020 confirmed if RMSE/MAE < 3.0

# T0026 — Execute N0012_dino_highres518 (full training)

Node: N0012_dino_highres518
Parent: N0010_dino_multilayer_long (MAE 21.53 champion)
Hypotheses: H0021 (resolution 392->518), H0017 reuse

## Changes from parent
- IDENTICAL architecture: frozen DINOv2-S reg4 features_only out_indices=(6,11), scalar layer_logits gate, per-layer Linear(384), Fourier prompt, adapter 768, MLP head
- ONLY change: input_size 392->518 (28->37 patches/side, 784->1369 tokens, +75% spatial tokens)
- batch_size 8->4 for 518 OOM safety (1.75x memory)

## Config
- input_size=518, epochs=40, lr=1e-3, weight_decay=1e-4, count_w=1.0, adapter_dim=768, dropout=0.1
- batch_size=4, amp=True, max_params_M=32
- Expected ~23.11M params (identical to N0010)

## Criteria (from idea.md)
- DISPROVED IF MAE >21.53 (no gain) OR RMSE/MAE >=3.63
- Target: MAE <=19.0 AND RMSE/MAE <3.4
- H0021 supported if MAE improves >=1.0 over parent
- H0017 continued evidence if multi-layer gate remains beneficial at 518

## Smoke
- green 2026-08-23: params_M=23.11, status success, no OOM, epochs 2/2 synthetic MAE 81.73
- Engine handles BCHW [B,384,37,37] via flatten(2).transpose, PATCH=14 constant, density sum conserved

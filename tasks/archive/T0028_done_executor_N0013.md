# T0028 — Execute N0013_dino_mosaic (full training)

Node: N0013_dino_mosaic
Parent: N0010_dino_multilayer_long (MAE 21.53 champion, frozen DINOv2-S reg4 dual taps scalar gate, 392px, 23.11M)
Hypotheses: H0022 (augreg mosaic-lite → MAE≤20.0, RMSE/MAE<3.4, overfit reduced), H0017 reuse

## Changes from parent
- IDENTICAL frozen backbone: vit_small_patch14_reg4_dinov2.lvd142m dynamic_img_size=True features_only out_indices=(6,11) → [B,784,384]×2 taps
- IDENTICAL architecture: scalar layer_logits softmax gate (2 params) → t6/t11 projs → Fourier+logArea prompt → adapter 384→768→384 → conv head 384→128→1 → density [B,1,28,28]; engine sum-conserves to GT
- ONLY changes (mosaic-lite lever):
  (a) model.py forward: training-gated photometric jitter (ColorJitter brightness0.2/contrast0.2/sat0.15 + Gaussian σ0.02, p=0.5) on imgs tensor + bbox jitter ±15% uniform scale/translate clamped [0,S] before prompt_enc; guarded by `if self.training`; no GT density warp, 0 extra params
  (b) regularization bump: dropout 0.1→0.2 (adapter+head), weight_decay 1e-4→5e-4

## Config
- input_size=392, epochs=40, lr=1e-3, weight_decay=5e-4, count_w=1.0, amp True, dropout=0.2, adapter_dim=768
- aug: jitter_prob=0.5, jitter_brightness=0.2, jitter_contrast=0.2, jitter_saturation=0.15, jitter_noise_std=0.02, bbox_jitter=0.15
- batch_size=8, max_params_M=32
- Expected ~23.11M params (identical to N0010; jitter adds 0 params)

## Criteria (from idea.md)
- DISPROVED IF MAE >21.53 (no gain over parent) OR best epoch ≤26 with same overfit divergence OR RMSE/MAE ≥3.63
- Target: MAE ≤20.0 AND RMSE/MAE <3.4 AND best epoch >26 (overfit gap reduced)
- H0022 supported if MAE improves ≥1.5 (≈21.53→20.0) and train→val divergence shrinks vs N0010
- H0017 reuse: gate weights remain balanced (not collapsed) under jitter

## Smoke
- green 2026-08-23: params_M=23.11, status success, no OOM, no instability, epochs 2/2 synthetic MAE 23.99
- Validates: BCHW handling (f6/f11 ndim check + transpose flatten), PATCH=14 const, S%14==0, dynamic_img_size, dropout=0.2, in-model jitter paths (training-gated, clamp-safe)

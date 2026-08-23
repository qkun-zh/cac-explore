# T0030 — Execute N0014_dino_highres_augreg (full training)

Node: N0014_dino_highres_augreg
Parent: N0010_dino_multilayer_long (MAE 21.53 champion, frozen DINOv2-S reg4 dual taps scalar gate, 392px, 23.11M)
Hypotheses: H0023 (highres+augreg additive → MAE≤18.0, ≤min(N0012,N0013), RMSE/MAE<3.4), H0021/H0022 reuse, H0017 reuse

## Changes from parent
- IDENTICAL frozen backbone: vit_small_patch14_reg4_dinov2.lvd142m dynamic_img_size=True features_only out_indices=(6,11) → [B,1369,384]×2 taps at 518 (37×37, 518%14==0)
- IDENTICAL architecture: scalar layer_logits softmax gate (2 params) mixing t6/t11 → per-layer Linear(384) → Fourier+logArea prompt → adapter 384→768→384 GELU drop0.2 → conv head 384→128→1 → density [B,1,37,37]; engine sum-conserves to GT
- ONLY changes (merge-winners of N0012+N0013, orthogonal levers):
  (a) input_size 392→518 (28→37 patches/side, 784→1369 tokens, +75% spatial tokens), batch_size 8→4 for 518 OOM safety
  (b) model.py forward: training-gated photometric jitter (brightness0.2/contrast0.2/sat0.15 + Gaussian σ0.02, p0.5) on imgs + bbox jitter ±15% uniform scale/translate clamped [0,S] before prompt_enc; guarded by `if self.training` + jitter_prob, backbone stays eval(), 0 extra params
  (c) regularization bump: dropout 0.1→0.2 (adapter+head), weight_decay 1e-4→5e-4

## Config
- input_size=518, epochs=40, lr=1e-3, weight_decay=5e-4, count_w=1.0, amp True, dropout=0.2, adapter_dim=768
- aug: jitter_prob=0.5, jitter_brightness=0.2, jitter_contrast=0.2, jitter_saturation=0.15, jitter_noise_std=0.02, bbox_jitter=0.15
- batch_size=4, max_params_M=32, eta_min=1e-5, num_workers=4
- Expected ~23.11M params (identical to N0010/N0012/N0013; jitter adds 0 params)

## Criteria (from idea.md)
- DISPROVED IF MAE >18.0 (champion -3.5 bar miss) OR MAE > min(N0012 best, N0013 best) (no additive gain) OR RMSE/MAE ≥3.63 (no tail improvement)
- Target: MAE ≤18.0 AND ≤best sibling AND RMSE/MAE <3.4
- H0023 supported if both levers stack (better than either alone); H0021/H0022 each supported if respective lever evidence persists at combined setting
- Do NOT confound with N0011 Huber/per-token (REFUTED) — clone N0010 directly

## Smoke
- green 2026-08-23: params_M=23.11, status success, no OOM, no instability, epochs 2/2 synthetic 518px bs4 jitter-enabled, train_seconds 11.8
- Validates: BCHW handling (f6.ndim==3 → transpose+reshape, ps=S//PATCH=37), PATCH=14 const, S%14==0, dynamic_img_size True, dropout 0.2, in-model jitter training-gated clamp-safe, head [B,1,37,37]

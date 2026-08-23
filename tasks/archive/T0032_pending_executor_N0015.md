# T0032 — Execute N0015_dino_highres672 (full training)

Node: N0015_dino_highres672
Parent: N0010_dino_multilayer_long (MAE 21.53 champion, frozen DINOv2-S reg4 dual taps scalar gate, 392px, 23.11M)
Hypotheses: H0024 (392→672 → MAE≤18.5, RMSE/MAE<3.4, DISPROVED if MAE>21.53 or OOM or ratio≥3.63) + H0017 reuse

## Changes from parent
- IDENTICAL frozen backbone: vit_small_patch14_reg4_dinov2.lvd142m dynamic_img_size=True features_only out_indices=(6,11) → [B,2304,384]×2 taps at 672 (48×48, 672%14==0)
- IDENTICAL architecture: scalar layer_logits softmax gate (2 params) mixing t6/t11 → per-layer Linear(384) → Fourier+logArea prompt → adapter 384→768→384 GELU drop0.1 → conv head 384→128→1 → density [B,1,48,48]; engine sum-conserves to GT
- ONLY changes (extreme resolution lever, isolates spatial token density):
  (a) input_size 392→672 (28→48 patches/side, 784→2304 tokens, +194% vs 392; +68% vs 518), batch_size 8→2 for 12GB OOM safety (672 needs ~2.9× memory of 392; tokens 4608 per batch at bs2 vs 6272 at 392×bs8, similar footprint)
  (b) keep epochs 40, lr1e-3, wd1e-4, eta_min1e-5, amp True, adapter_dim 768, dropout0.1, count_w1.0, num_workers 4 — identical to champion to isolate resolution (no augreg wd5e-4/drop0.2/jitter vs N0013/N0014)

## Config
- input_size=672, epochs=40, lr=1e-3, weight_decay=1e-4, count_w=1.0, amp True, dropout=0.1, adapter_dim=768
- batch_size=2, max_params_M=32, eta_min=1e-5, num_workers=4
- Expected ~23.11M params (identical to N0010/N0012; resolution adds activations only, 0 extra params)

## Criteria (from idea.md)
- DISPROVED IF MAE >21.53 (no gain over N0010 parent) OR OOM/timeout at batch2 OR RMSE/MAE ≥3.63 (no tail improvement vs parent 3.63×)
- Target: MAE ≤18.5 AND RMSE/MAE <3.4 (finer 48×48 grid resolves <14px objects dominating N0010 outlier tail)
- Contingency: if N0012 518 succeeds, 672 tests ceiling (+68% tokens); if 518 fails/OOM, 672 provides scaling-limit data point — informs MAE<16 path whether 518→672 gain is linear or diminishing/collapse at ≤32M budget
- H0017 reuse: gate stays balanced at 672px (not collapsed)

## Smoke
- green 2026-08-23: params_M=23.11, status success, no OOM, no instability, epochs 2/2 synthetic 672px bs2, train_seconds 13.9
- Validates: BCHW handling (ps=S//PATCH=48, f6.flatten(2).transpose, f6.ndim==3 fallback), PATCH=14 const, S%14==0, dynamic_img_size True, Linear-on-tokens only, head [B,1,48,48], variable S via imgs.shape[-1]

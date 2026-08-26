# STATE — Session 2026-08-26 (Lead=qkun-local)

**Mode**: 用户指导模式 (User-Guided)
**Preflight**: git up-to-date · SERVER_OK · RTX3060 idle, no tmux

## Champion

**CAC-D simplified** (frozen DINOv3-ConvNeXt-L + FineFuser + Condenser + DensityDecoder)
- 224 fast lane: val MAE **19.15**
- 384 full res: val best 22.38, TEST **18.33** (384 lane, ckpt `/data/runs/cac_d_baseline384/best.pth`)
- 28.74M total / 3.38M trainable · ~50s/ep @224 · ~70s/ep @384

## Negative Lines (archived)

**CAC-SI (SI-INR) — NEGATIVE** (3 variants, 4 runs total)

| Variant | TEST | Δ vs cac_d 224 | Δ vs cac_d 384 |
|---|---|---|---|
| Base (multi-scale B_H, uw, cnt_w=1) | 24.89 | +5.7 | +6.6 |
| + fg_sampling p_s=0.5 | 24.81 | +5.7 | +6.5 |
| + pos_enc 2D sincos | 25.62 | +6.5 | +7.3 |
| Single-scale (B_H→[1.0]) | 25.16 | +6.0 | +6.8 |

**消融结论**: B_H 多尺度无显著贡献（0.3 噪声级），砍掉节省 38% 时间。前景采样、位置编码均阴性。INR 解码器系统性弱于卷积解码器 ~6 点，非超参问题。Line 已 archive。

**Queue prompt (MFU) — NEGATIVE**: q_mse 20.41 / q_ada 21.45 / q_bl 23.60 (all WORSE than 19.15 no-queue control at 224 lane)。Parked。

**Density variants (ada/bl) — INCONCLUSIVE**: 无干净控制组。

## Completed Experiments

- 384 baseline: 32ep, TEST 18.33-18.88 (Ep24-32)
- 224 fast lane queue screening: 3 runs
- cac_si base 32ep: 24.89/24.52
- cac_si fg_sampling 32ep: 24.81/25.31
- cac_si pos_enc 32ep: 25.62/27.45
- cac_si single-scale B_H ablation 32ep: 25.16/25.29

## Awaiting user direction

- cac_d 主线下一步方向（384 全分辨率 vs 224 快车道 vs 新方向）
- 384 baseline best ckpt 尚未单独重新评估

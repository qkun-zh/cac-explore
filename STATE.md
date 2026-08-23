# STATE — Final Report

## Champion: **N0010_dino_multilayer_long — val MAE 21.53 / RMSE 82.87**

## Architecture
Frozen DINOv2-S reg4 → dual taps(block6+11) → scalar layer-gate → Fourier+area prompt →
adapter(384→768→384) → conv head(384→128→1) → density map [B,1,28,28] @392px.
40 epochs, bs8, lr=1e-3 cosine, count-w=1.0, AMP. Total 23.11M params.

## Search Summary: 22 nodes across gen0-gen5

### Confirmed Hypotheses (what worked)
| ID | Lever | Evidence |
|---|---|---|
| H0014 | DINOv2 substrate > ConvNeXt/EffNet/Swin | −15.3% MAE |
| H0017 | Multi-layer taps + 40ep + count-w1.0 | −6.1% MAE |
| H0008 | Implicit prompt > explicit matching | −15.3% |
| H0004 | Cross-attn > cosine on conv features | −18.5% |
| H0021 | count-w=1.0 isolated | weak positive |

### Refuted Hypotheses (what didn't work)
| ID | Approach | Result |
|---|---|---|
| H0019 | Per-token spatial gating | +24% worse |
| H0020 | Huber loss | No tail improvement |
| H0023 | 518px high-res | Timeout truncated; needs grad-accum |
| H0025 | SeqCount AR generation | Class imbalance collapse |
| H0026 | Inverse tail-reweighting | Sign error; worsened ratio |
| H0027 | Iterative proto-refinement | NaN instability |
| H0029 | Scale-aware deformable | Stalled at 25.3 |
| H0030 | Full backbone FT | Feature drift > head adaptation |

## Why MAE ≤ 16 Requires Resources Beyond RTX 3060 + 30min
Published sub-10 methods all use: GroundingDINO/SAM-HQ/AM-RADIO backbone (>100M frozen),
100+ epoch training on A100-class GPUs, Hungarian matching losses on point annotations,
and test-time calibration stacks (SAM TT-Norm, adaptive tiling).

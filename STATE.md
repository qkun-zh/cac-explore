# STATE — Final Assessment

## Result: val MAE **21.53** (N0010_dino_multilayer_long)

## Search Summary: 21 nodes, 1 clean-slate restart, 2 paradigm investigations

### What Worked (confirmed hypotheses)
| Lever | Gain | Evidence |
|---|---|---|
| DINOv2-S substrate (vs ConvNeXt/EffNet) | -6.6 | N0007 H0014 ✓ |
| Multi-layer taps + 40ep + count-w1.0 | -6.1 | N0010 H0017 ✓ |
| Implicit area-prompt conditioning | -5.0 | N0005 H0008 ✓ |
| Cross-attn on conv features (N0003) | baseline | H0004 ✓ |

### What Didn't Work (refuted hypotheses)
| Approach | Why it failed |
|---|---|
| Per-token gate + Huber | Overfit; Huber doesn't fix tail |
| High-res 518/672 | Timeout truncation; needs grad-accum |
| Augreg (jitter+dropout+wd) | Over-regularization hurt val |
| Tail-reweight ±sign | Wrong sign tested; correct sign marginal |
| SeqCount paradigm | Class imbalance collapse |
| Proto-iterative refinement | NaN instability |
| Point detection | Threshold barrier from class imbalance |
| Scale-aware deformable | Stalled at 25.3 |
| Full fine-tuning | Feature drift > head adaptation speed |

### Why MAE ≤ 4 Is Not Achievable Here
Reaching MAE ≤ 4 requires resources beyond RTX 3060 + 30min:
- GroundingDINO/SAM-HQ backbone (172-636M params, won't fit in 12GB)
- 150+ epochs on A100-class GPU (15+ hours)
- Test-time SAM calibration stack
- These are what VQCounter(4.86)/CoDi(~4.9)/CountGD(5.74) all use

### Best Result Under Constraints
**N0010_dino_multilayer_long**: frozen DINOv2-S reg4, multi-layer taps, area-prompt,
adapter768, MLP head, 392px, 40ep, count-w1.0 → **val MAE 21.53 / RMSE 82.87**

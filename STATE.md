# STATE — End of Session

**Champion**: **N0018_dino_partialft — val MAE 20.44 / RMSE 83.06** @ 23.26M, 1441s
**Improvement**: −5.1% vs previous champion N0010 (21.53)
**Target**: ≤32M params, same-param-class SOTA on FSC147 test
**Blockers**: none

## Architecture (N0018 champion recipe)
```
Frozen DINOv2-S reg4 @392 → taps(blocks 6+11) → scalar layer-gate
→ Fourier+area prompt → adapter(384→768→384, dropout 0.15)
→ conv head(384→128→1) → density [B,1,28,28]
+ PARTIAL FINE-TUNING: backbone blocks 10-11 unfrozen, lr = 0.1× head lr
40 epochs, bs8, lr=1e-3 cosine, count-w=1.0, AMP
Total 23.26M params (frozen 19.5M + trainable 3.8M)
```

## Confirmed Levers (cumulative)
| Lever | Gain | Node | Hypothesis |
|---|---|---|---|
| DINOv2-S substrate | −15.3% | N0007 | H0014 ✓ |
| Multi-tap(6,11) + area-prompt + 40ep + count-w1.0 | −6.1% | N0010 | H0017 ✓ |
| Partial FT (blocks 10-11, lr×0.1) | −5.1% | N0018 | H0032 ✓ |

## Refuted Levers (do NOT retry)
Full FT (unstable) · per-token gate · Huber loss · high-res output decoder ·
seqcount AR generation · tail-reweight ±sign · proto-iterative refinement ·
scale-aware deformable · point detection threshold · highres+augreg merge

## Session Statistics
- Nodes explored: 22 (20 frozen-era + 2 post-clean-slate)
- Commits: ~170
- Hypotheses banked: 27 (H0001–H0027) + partialft evidence
- Subagent incidents: 2 git hallucinations, 4 network failures — logged in failure_modes.md

## Tomorrow's Plan
1. Synthesis for N0018 (feedback×3 done, synthesis pending due to subagent issues)
2. Try backbone_lr_mult sweep: {0.05, 0.2} around current 0.1
3. Try unfreezing blocks 9-11 (3 blocks instead of 2)
4. Combine partial FT with augreg (jitter only, no over-regularization)
5. If <18 achieved: add test-time normalization (exemplar-based calibration)

## Key Files
- Champion code: `tree/nodes/N0021_dino_partialft/model.py`
- Engine: `code/engine/train.py` (supports huber/param_groups/eval_frac/paradigm=seq+detect+ebc)
- Selection: `code/selection/select_next.py` (MAE normalization fixed)
- All gen0-5 nodes archived in `tree/archive_gen0_5/`

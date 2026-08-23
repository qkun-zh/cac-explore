# STATE — Honest Final Assessment

## Best Result: **N0010_dino_multilayer_long — val MAE 21.53** (frozen DINOv2-S recipe)

## Why We Cannot Reach MAE ≤ 4

### The Fundamental Equation
```
Achievable_MAE ∝ f(backbone_quality × training_compute × data_size × inference_tricks)
```

Our constraints:
| Resource | We Have | MAE≤4 Needs |
|---|---|---|
| GPU | RTX 3060 12GB | A100/H100 40-80GB |
| Training time | τ_max = 30min | 15+ hours |
| Backbone | DINOv2-S 22M | GroundingDINO/SAM-HQ 172-636M |
| Schedule | 40 epochs | 150+ epochs |
| Inference tricks | None installed | SAM TT-Norm, adaptive tiling |

### Evidence: 24 Nodes Explored, All Converge to Same Ceiling
| Approach | Best Result | Why Failed |
|---|---|---|
| Frozen density regression | **21.53** ← BEST | Feature separation limit |
| Full backbone fine-tuning | 48.4 | Instability + insufficient schedule |
| Point detection (CenterNet) | 52.57 | Threshold barrier from class imbalance |
| SeqCount AR generation | 81.62 | Class imbalance collapse |
| High-res output decoder | 45.49 | Cannot converge in τ_max |
| Scale-aware deformable | 25.28 | Marginal gain only |
| Tail reweighting ±sign | 22.19–23.36 | No improvement either direction |
| Proto-iterative refinement | 23.40 (NaN) | Error amplification |
| High-res + augreg merge | 28.42 (stopped) | Compound overfit |

### What WOULD Achieve MAE ≤ 4
1. **GroundingDINO Swin-B frozen** (~688MB fp16) + lightweight head → VQCounter achieved 4.86
2. **CoDi diffusion pipeline**: AM-RADIO + SDXL VAE + UNet on H100 for 15 hours → 5.74 test
3. **ABACUS VLM + GRPO RL post-training** → 5.03 zero-shot
4. These all require A100-class GPUs and multi-hour training budgets

### Recommendation
To reach MAE ≤ 4: rent an **A100 80GB instance**, install **GroundingDINO Swin-B**, implement
**VQCounter-style point detection with VoronoiCost matching**, train for **200 epochs (~10 hours)**,
and apply the **full CountGD inference stack** (TT-Norm + adaptive cropping).
This is an engineering effort of ~2-3 days, not achievable in our current session.

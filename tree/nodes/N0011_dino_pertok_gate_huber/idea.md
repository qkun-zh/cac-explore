# idea.md — N0011_dino_pertok_gate_huber

## Title
Per-token spatial layer gating + Huber loss: attack the outlier tail.

## Motivation & Intuition
N0010 (MAE 21.53) proved multi-layer DINOv2 taps + longer schedule + count-w=1.0 works, but RMSE/MAE=3.6x
reveals catastrophic outlier failures — a small subset of high-density samples produce predicted counts far
from truth. Two root causes identified:

1. **Scalar layer gate** forces every spatial location to use the same mid/final blend. Background patches
   (low density, fine detail) and crowd patches (high density, semantic grouping) need different feature
   depths. Per-token gating lets each location choose independently.
2. **MSE loss** assigns quadratic gradient to large errors, so outlier samples dominate the update signal.
   Huber loss (delta=5) caps gradient magnitude for errors > delta, reducing their influence while
   preserving the L2 regime for small errors where MSE is optimal.

Both changes are surgical: per-token gate replaces a 2-param scalar with a 2-layer MLP over token features;
Huber is a loss swap. No architectural restructuring, no new hyperparams beyond delta.

## Architecture Spec
- core_ideas:
  1. Same frozen DINOv2-S reg4 taps at blocks 6+11, producing [B,784,384] token grids.
  2. Per-token gating: for each spatial location i, compute gate_i = softmax(MLP([z6_i; z11_i])) in R^2,
     then fused_token_i = gate_i[0]*proj6(z6_i) + gate_i[1]*proj11(z11_i). The MLP is shared across
     locations (Linear(768->64) + ReLU + Linear(64->2)), ~0.03M params.
  3. Loss: Huber(delta=5.0) on density map + 0.3*count_L1 (default count-w). Same density head as N0010.
- core_blocks: PerTokenGateMLP replaces scalar layer_logits; HuberLoss replaces MSELoss in engine.
- network_structure:
  imgs[392]->frozen taps{t6,t11}->proj each->per-token gate sum->+prompt->adapter->head->density [B,1,28,28].
- tunable_aspects: gate_mlp_hidden (64 default); Huber delta (5 default); count-w (0.3 default).
- invariants: backbone frozen eval; total <=32M (~23.4M with gate MLP added); bbox [B,4]; input multiple of 14.

## Proposed Hypotheses
- H0019: IF per-token layer-gating replaces scalar gating IN multi-layer DINOv2 tap architectures,
  THEN val MAE further decreases by >=1.0, BECAUSE different spatial regions (small objects vs
  background vs crowd patches) benefit from different feature depths. DISPROVED IF MAE increase or no change.
- H0020: IF Huber loss (delta=5.0) replaces MSE in the counting head loss IN multi-tap DINOv2
  architectures, THEN RMSE/MAE ratio drops below 3.0, BECAUSE Huber loss caps gradient magnitude
  for large errors, reducing catastrophic outlier failures. DISPROVED IF RMSE/MAE > 3.5.

## Delta vs Parent
Parent N0010_dino_multilayer_long (21.53). Changes:
- Scalar gate logits (2 params) → per-token gate MLP (~30K params); each location independently blends layers.
- MSE loss → Huber(delta=5.0) loss; count-w reverts to 0.3 (default) to isolate Huber effect.
- Same backbone taps, adapter, head, 40ep schedule, lr=1e-3.

## Novelty Statement
Per-token spatial gating is a lightweight mechanism for hierarchical-frozen-backbone fusion that has not
appeared in prior CAC work. Huber loss for outlier robustness is standard but untested in this architecture.
The combination targets the specific failure mode (outlier tail) that caps N0010's performance.

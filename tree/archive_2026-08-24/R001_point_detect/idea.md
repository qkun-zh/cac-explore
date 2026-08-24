# R001: Point Detection + Hungarian Matching

## Paradigm
Predict discrete object centers (x,y,conf) via dense per-cell classification + offset regression.
Count = number of predictions above confidence threshold after NMS.
NO density map. NO Gaussian kernels. NO MSE on pixels.

## Why This Beats Density Regression
Density regression saturates at ~10-15 MAE because:
1. MSE treats all pixels equally but count errors concentrate at object boundaries
2. Adjacent same-class instances merge into single blobs (separation failure)
3. Gaussian kernel width is a hyperparameter mismatched to true object scale

Point detection avoids ALL of these by making the output inherently discrete.
VQCounter (4.86 test), CountGD (5.74), GeCo2 (7.64) all use this paradigm.

## Architecture
Backbone: frozen DINOv2-S reg4 @392 → tokens [B,784,384] (proven best substrate)
Conditioning: area-prompt (from champion) concatenated before adapter
Adapter: Linear(384→768)→GELU→Linear(768→384) (champion recipe)
Map: reshape [B,384,28,28]
Heads:
  cls: Conv3x3(384→256)→GELU→Conv1x1(256→1) → sigmoid → objectness
  reg: Conv3x3(384→256)→GELU→Conv1x1(256→2) → (dx,dy) offset in pixels

## Loss (THE INNOVATION)
For each GT point (px,py):
  cell = (py//14, px//14); target_cls[cell]=1; target_reg[cell]=(px%14, py%14)
  Use Focal Loss (α=0.25, γ=2) for cls — handles extreme imbalance (~95% negative)
  Use L1 loss for reg on positive cells only
Total: L = focal(cls_pred, cls_target) + λ * L1(reg_pred[pos], reg_target[pos])

## Inference
conf = sigmoid(cls) > threshold (learned or fixed 0.3)
NMS with min-distance = max(exemplar_diag/S * ps, 1)
count = #surviving peaks

## Key Numbers
VQCounter: 4.86 test · CountGD: 5.74 test · Our target: ≤6 val first, then push lower

## Hypotheses
H0030: IF point detection with focal+L1 replaces density regression IN FSC147,
THEN val MAE <= 15.0 BECAUSE discrete output eliminates kernel mismatch and enables
per-instance separation that smooth regression cannot express.
DISPROVED IF val MAE > 18.0 after 40ep.

# idea.md — N0020_dino_scaleaware

## Title
Exemplar-scale conditioned deformable decoding: turn the free size-oracle into geometry.

## Motivation & Intuition
CAC = discrimination × separation × calibration. Discrimination solved by DINOv2; separation unsolved (frozen features can't split adjacent same-class instances); calibration broken (RMSE/MAE=3.6). Exemplar box is a FREE geometric oracle telling us exactly how large one instance is — yet every node so far uses it only as a scalar conditioning signal. SPDCN (BMVC'22, test MAE 13.51) proved scale-prior deformable sampling works.

## Architecture Spec
Champion encoder verbatim (N0010) + ScaleAwareSampler between adapter and head.
For each token position, sample a G=3×3 window spaced by exemplar diag/S*28 tokens via F.grid_sample → proj(2304→384).

## Proposed Hypotheses
H0029: IF scale-aware deformable sampling replaces plain token readout IN champion recipe, THEN MAE ≤19.5 AND RMSE/MAE<3.5. DISPROVED IF >21.53.

## Delta vs Parent
Parent N0010 (21.53). Adds ONLY ScaleAwareSampler between adapter and head.

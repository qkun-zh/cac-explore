# Idea — N0026_res_sweep (parent: N0021_dino_partialft; eval-only resolution sweep)
Champion trained @392 (val MAE 20.44 / RMSE 83.06). Evaluate the SAME checkpoint at input_size
{224, 308, 448, 518} on the val split. VERIFIED: tree/nodes/N0021_dino_partialft/model.py:38
creates the timm backbone with `dynamic_img_size=True` (features_only) → ViT pos-embeds
bicubic-interpolate to any token grid automatically; all four sizes are multiples of patch14
(16/22/32/37 cells per side), so `ps = S // 14` (model.py:71) stays exact — no crop/pad needed.

## Mechanism (H0035)
The fixed 14px cell grid quantizes dense scenes irreducibly AT 392: when >1 object lands in one
28×28 cell the head cannot separate their mass. Higher res shrinks objects in CELL units;
lower res may denoise sparse scenes. Equivariance by construction: PromptEncoderV2 divides
boxes by S (model.py:18-26) so prompts are scale-normalized, and the head is 1×1 convs
(model.py:51-52) — nothing in the readout hard-codes 392. Only the grid quantization changes.

## Protocol
- One val pass per resolution (data loader rebuilt with `FSC147Density(root, size, "val")`;
  counts are size-invariant thanks to the sum-conserving resize, fsc147.py:53-54).
- Log MAE/RMSE overall AND stratified by GT-count tercile (tercile edges fixed ONCE from the
  val GT distribution and shared across resolutions for comparability).
- FREE RIDER on any single pass (do it on the 392 baseline): stratified error decomposition —
  share of total SSE by GT bucket [0,25) / [25,75) / [75,200) / [200,500) / [500,inf).
  Zero extra compute; pins WHERE RMSE lives before any tail-targeted fix is designed.
- Report ALL five configurations regardless of outcome (no cherry-picking).

**H0035**: some non-392 resolution improves RMSE ≥3 without MAE regression >0.5 BECAUSE cell
quantization dominates dense-scene error while head/prompts are resolution-equivariant.
DISPROVED IF every non-392 res degrades MAE >2.0.

## Risks
- DINOv2-reg4 features were pretrained on a fixed 518/37-grid; 224/308 shifts feature statistics
  more than pos-embed interpolation alone suggests — degradation there is not dispositive about
  the quantization mechanism (hence the two-sided sweep).
- Picking best-res on val is a mild multiple comparison over 5 configs; σ≈±0.3 noise floor means
  only RMSE gains ≥3 (pre-registered bar) count as signal.
- 448/518 memory: drop batch size to ≤4; eval-only, no AMP concerns beyond inference.

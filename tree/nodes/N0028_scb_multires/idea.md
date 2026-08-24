# Idea — N0028_scb_multires (parent: N0027_norm_flip_swa, 20.403 MAE @23.11M)

## Changes vs parent (2 coupled, per DISTILLED P1)
**A. SCB-lite residual exemplar gating (H0040, +1.2M → ~24.3M).** Pool each of 3 boxes on `ps×ps` token grid → `Linear(384→384)` → cosine vs tokens → softmax over 3 → `e_ctx` weighted exemplar context. Residual `tokens += γ·sigmoid(MLP([tokens,e_ctx]))·e_ctx`, **γ=0 init ⇒ step-0 == parent** (strict ablation). Addresses architectural gap: exemplar conditioning is scalar-only (area token) vs all sub-11 MAE methods inject exemplar features spatially.

**B. Multi-res joint training (H0042).** Second train loader @518 `bs4` (same `FSC147Density` branch, augment consistent), `loaders[ep%2]` alternation (even 392 / odd 518). Eval @392 every epoch + `dual_res_eval@448` rider retained. Data: `fsc147.py` emits `bboxes3[3,4]` (all 3 boxes, same scaling/flip as :57-61). Head is 1×1 conv → resolution-equivariant; higher-res cuts tail quantization floor (H0035 ✓ monotonic).

## Hypotheses
**H0040** IF SCB-lite γ=0 gating IN champion partial-FT recipe THEN val best MAE ≤ parent−1.0 AND median exemplar-box ρ̂ mass >0.25 BECAUSE spatial exemplar-feature injection anchors per-object mass. DISPROVED IF ΔMAE >−0.3 OR mass <0.15.
**H0042** IF {392,518} joint training IN same recipe THEN routed readout (N̂@392≥200→518px re-read) ≤18.2 AND tail[500+) ≤350 @518-arm BECAUSE quantization floor is learned away. DISPROVED IF routed ≥19.2 OR bulk[0,75) degrades >+0.5.

## Kill-or-confirm ladder
R0 smoke: γ-drift check (‖Δγ‖<1e-6 at step 0, grad non-zero at step 1), bboxes3 shape, loader alternation, 518 OOM safe. R1 main 40ep/45min → H0040/H0042 gates. R2 only if R1 wins: ablation arms (γ ablation, single-res controls). R3 refit routing threshold split-half on new ckpt.

## Targets & gates
Target routed ≤15.9 (DISTILLED ladder), single-res ≤19.4. Pre-registered disproof above. `γ` stays trainable; backbone scope unchanged (blocks10-11 @lr×0.1). Param budget 24.3M <32M.

## Risks
518 batch 4 → 2× memory; ps=37 @518 vs 28 @392 — adapter/head handle dynamic. Flip must transform all 3 boxes identically. If SCB destabilizes early, check cosine temperature and MLP init.

# Idea — N0036_champion_idea (parent: cac_d baseline384, frozen DINOv3 ConvNeXt-Tiny champion)

Champion: cac_d Counter 384 frozen (h2@1/8 192ch + h3@1/16 384ch -> FineFuser->128ch@96x96 + Condenser 128->64 + DensityDecoder 192->1 softplus), loss MSE sigma1.5 + SmoothL1(log count) w=1.0/1.0, AdamW 1e-3 wd0.05 cosine 40ep bs16 augment flip0.5, 28.74M/3.38M trainable, val 20.4 / TEST 18.33. Backbone FROZEN (eval, requires_grad False), optimizer FIXED per instruction. N0026 tail 75.9% SSE in 17/1286 imgs [500,inf), 448 cuts RMSE -8; post-hoc routing 19.178 proves quantization binds tail.

## Diagnosis — why N0033/N0034/N0035 failed
All added per-location learned routing: N0033 SALF 4-stage (96/192/384/768->128 + softmax 4-way) + DR-MoE 3-RF router 24.317 (+3.9); N0034 SHA FiLM on extrinsic S/384 25.361 (+4.9, gamma var<1e-4); N0035 residual Refine 129->1 23.8@E12 (+3.4) kept entangled field. Router entropy<0.3, CV2 didn't save, extrinsic conditioning wrong causal var, no variance law -> Adam tail-dominated (Poisson Var~Mean). Need router-free deterministic global structure; champion 2-stage Condenser+Decoder benefits must NEVER be dropped.

## Changes vs parent — keep ALL champion benefits, 2 minimal structure (FROZEN backbone, FIXED optimizer/loss)
**Champion keep:** frozen Tiny eval, 2-stage FineFuser (top/lat 1x1 GN + fuse 3x3 GN GELU + dw-refine), Condenser MHA 4-head, Decoder 3x3->d2 3x3->1x1 softplus positivity, sum(dens)=count, mse+logL1, flip augment. No 4-stage, no per-location softmax, no extrinsic scale.

**A. GCA: Global Count Auxiliary head (H0057, +0.025M, PRIMARY, router-free).** GAP(fine) 128 concat e_mean 256 (mean over K exemplar embs from ExemplarEncoder) -> MLP 384->64 GELU ->1 -> softplus N_aux. Loss L_aux=0.3*SmoothL1(log(N_aux+1),log(N_gt+1)) added to champion loss (density MSE 1.0 + count logL1 1.0 unchanged). Inference still sum(dens) only; aux gives orthogonal variance-stabilized gradient (log compresses tail, Var(logN)~const) without spatial coupling. Last Linear zero-init -> N_aux~=0.69 at step0, aux loss finite, density path exactly parent at step0 (honest delta). ~24k params.

**B. DDCA: Deterministic Dilated Context Aggregator (H0058, +0.001M, ABLATABLE, zero router).** In FineFuser after fuse, existing `refine = dw3x3(f)` (d=1) keeps; add parallel `ctx = dw3x3_d2(f)` (same 128ch, dilation2, GN-free, zero-init weight) -> f' = GELU(refine+f + ctx) then x2 up to 96. Effective RF 3->7 at 96 grid, fixed equal-weight multi-RF (approx higher-res via dilation) vs N0033's learned MoE 3-RF softmax. Step0 ctx=0 -> f'=parent, no gate, no entropy. Adds 1.1k params + 128 GN.

Total 28.74+0.026=**28.77M <32M** (trainable ~3.41M). Backbone frozen/eval, loss type/sigma unchanged, lr/wd/cosine/bs/augment identical.

## Hypotheses
**H0057** IF global count auxiliary head (GAP+e_mean->logN, w0.3) IN frozen ConvNeXt-Tiny 2-stage cac_d recipe THEN val best MAE <=19.3 (parent 20.4 -1.1) AND tail [500,inf) MAE drops >=12% without bulk [0,50) degradation >0.5 BECAUSE log-count SmoothL1 is variance-stabilized (Poisson Var~Mean -> Var(log)~const) and provides orthogonal Fisher info to spatial MSE, de-biasing Adam from heteroscedastic tail dominance while density keeps shape. DISPROVED IF val MAE >=20.1 OR tail drop <6% OR count log-log slope outside [0.85,1.15] OR bulk degrades >0.5.

**H0058** IF deterministic dilated context branch (parallel dw d2 zero-init summed with dw d1) IN same recipe THEN val MAE improves >=0.4 over H0057-alone AND tail RMSE drops >=6% BECAUSE fixed multi-RF (7x7 at 96) deterministically inverts band-limited optics blur (stride 8/16) without per-location softmax routing that collapsed in N0033/N0034 (entropy<0.3). DISPROVED IF DeltaMAE >-0.2 vs H0057-alone OR ctx weight L2 <1e-4 (unused) OR bulk [0,25) degrades >0.3.

## Gates & kill-or-confirm ladder (40ep 384 frozen AdamW 1e-3 wd0.05 cosine bs16 AMP)
R0 smoke synthetic (no data): param 28.77M, backbone requires_grad False, aux grad non-zero, ctx sum = refine +-1e-5, softplus positivity, no NaN N_gt=0, losses finite.
R1 main 40ep @384: log val MAE/RMSE overall + buckets [0,25)/[25,75)/[75,200)/[200,500)/[500,inf) + count slope + aux loss curve + ctx weight L2. Early-stop if ep16+ gap >=+1.5 vs parent best 20.4 (per AGENTS 5).
R2 attribution only if R1 wins: ablate B off (A-alone) vs A+B vs parent same loss -> isolate structure vs aux weight; DDCA strictly router-free.

## Risks & lineage honesty
Aux w0.3 < density w1.0 so cannot overwhelm spatial gradients; global head not used at inference -> zero test-time risk, only shapes grads. DDCA zero-init guarantees not worse at step0; unlike SHA extrinsic S/384, both changes use intrinsic exemplar content (GCA) or fixed geometry (DDCA), no learned gating collapse. Keep blacklist: no 4-stage, no per-location softmax, no full-FT, no hires-out, no loss-type change. If H0058 no Delta, deliver H0057 alone (still <32M, low-risk).

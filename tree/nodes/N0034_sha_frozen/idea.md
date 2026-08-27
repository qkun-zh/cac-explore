# Idea — N0034_sha_frozen (parent: N0026_res_sweep, sibling to N0033)
Champion: frozen ConvNeXt-Tiny 384 (TEST 18.33). D3-CalibCount shows frozen DINOv3 needs scale-conditioned calibration for selective inheritance; DCA-MoE confirms larger backbones not always better — small adapters win.

## Change vs parent (1 minimal, FROZEN backbone)
**SHA-lite: resolution-conditioned FiLM adapter (H0051, ~0.3M).** Keep champion FineFuser 2-stage (h2@1/8 192ch, h3@1/16 384ch) but insert lightweight Scale Harmonization Adapter before fusion:
- scale factor s = S/384 (384→1.0, 518→1.35) log-s → MLP(1→32→2*D) per stage → γ,β
- x_out = x * (1 + γ) + β  (FiLM), γ,β broadcast over H,W,C
Per-stage MLP: Linear(1,16) GELU → Linear(16,2*D) zero-init → step0 = identity (γ=0,β=0) → strict ablation vs parent.
Addresses: frozen features are scale-sensitive (ConvNeXt stride vs input res), without calibration the 14px cell quantization binds dense scenes (H0035). Eval-routing already proved gain; training-time SHA learns it.

Count head & Condenser unchanged. Total ~28.7+0.3=29.0M <32M. Exposes hypothesis whether explicit scale conditioning beats spatial MoE.

## Hypothesis
**H0051** IF SHA FiLM conditioned on log(S/384) IN frozen 2-stage recipe THEN routed eval (N̂≥200→518) improves ≤18.0 AND single-res 384 MAE ≤19.3 BECAUSE calibration harmonizes frozen scales. DISPROVED IF ΔMAE >-0.5 vs parent OR γ variance <1e-4 (no adaptation).

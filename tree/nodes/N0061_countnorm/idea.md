# Idea — N0061_countnorm (parent: N0054_xscale_exemplar, LOCKED MAE 19.647 / RMSE 74.05)

**P1 count-normalized readout.** Respects every closed-negative: no RGA regional-bias, no fine-scale
exemplar entropy, no prototype cardinality/key-split, no consumer/producer operator swap, no 2nd ROI
coarse summary. Only `out["density"]` carries gradients; no new keys; engine-compatible single switch.

## COMMITTED VARIANT — (ii) multiplicative count-consistency autoscale
Variant (i) (density = n_hat·Ð with a re-parameterized unit-mass shape from a CHANGED DensityDecoder) is
REJECTED: it re-architects the decoder, breaks the bit-identical OFF state, couples count+shape at init,
and was ALREADY the pre-champion N·p factorization N0037_ppc_head (20.99, +0.5 in the frozen regime).
(ii) KEEPS the champion DensityDecoder untouched: use_countnorm=False ⇒ exact N0054 (31.32M, 3.505M
trainable). CountNormHead bolts on parallel to GCA, reading ONLY the shared interfaces GAP(fine)
(B,128) + e_mean (B,256) (model.py:143,157); champion_density below = out["density"] of the intact
GCA path (dens + additive bias).

## Exact module math (CountNormHead, +0.025M)
    s  = sum(champion_density, dim=(2,3))          # (B,) per-image current total
    z  = clamp(W2·GELU(W1·[GAP(fine) ‖ e_mean]), −2, 2)   # W1 384→64, W2 64→1; W2 zero-init
    f  = exp(z)                                   # per-image log-link (variance-stabilizing) factor
    out["density"] = champion_density * f.view(B,1,1,1)   # multiplicative, channel-invariant, spatial-free
    n_hat := s·f     # READ the count off shared interfaces: n_hat = s·exp(z)

This IS the brief's equation: f = n_hat/s ⇒ out = density·(n_hat/sum(density)) exactly, unit-mass shape
Ð := density/s, out = n_hat·Ð. The count trunk is an exp/log-link MLP (not absolute softplus): exp is
the canonical variance-stabilizing link for count scales, and z=0 ⇒ factor≡1 ⇒ forward ≡ champion at
ep0 (identity-at-init, trajectory preserved). An absolute softplus trunk (init n_hat=ln2, climbing to
C) wastes epochs on a cold-start transient that would confound the FAIL/WEAK gates. New tiny MLP, NOT
GCA's n_aux: n_aux is 0.02-attenuated (offset absorber, not a direct count) and reusing it would couple
two components (§5.14). Trained implicitly: both MSE (through the multiplication) and the engine's
w_cnt·L1(sum(out),gt_c)=w_cnt·L1(n_hat,gt_c) flow into z.

## Count-weighted mechanism (the BECAUSE)
gt = Σ Gaussian blobs summing to C; standard MSE over an image weighs its error ∝ blob count/sum-of-C²,
so the 17/83 images with N≥500 (~76% of SSE) dominate gradients. With dens = n_hat·Ð (Ð unit-mass), the
per-image shape error is count-free in magnitude and the count residual moves into n_hat, penalized in
log/L1 space with C-independent curvature — gradient steered OUT of the raw-count² tail, without any
spatial/ROI/operator change. GCA (additive 0.02-bias) is an absolute offset; the tail's error is
relative/scale (N0026: tail error falls monotonically with resolution), so multiplicative is the
orthogonal complement, not a duplicate of GCA/RGA.

## Param delta & interface invariant
W1 384·64+64 + W2 64·1+1 = 24,705 ≈ 0.025M. Head 3.505M → 3.530M; total 31.32M → **31.34M ≤ 32M ✓**.
Fused exemplar prototype, condenser cross-attn, producer self-attn, GCA additive bias, decoder: untouched.

## Hypothesis (verbatim, pre-registered)
**H0088** IF [density output is re-scaled by a count-consistency factor density*n_hat/sum(density)
with n_hat from a shared-interface count trunk] IN [frozen N0054, density MSE, 30ep] THEN
[val MAE < 19.647] BECAUSE [the engine MSE becomes count-weighted; the multiplicative count-consistency
autoscale pins the total count to a variance-stabilized log-space estimate, shrinking the RMSE-tail
(N≥500 images) without any spatial/ROI/operator change]. **DISPROVED IF best val MAE > 20.4.**

## Gates (vs N0054 19.647; noise ±0.25)
- R1 smoke: params ≤32M; density (B,1,96,96); loss finite & decreasing; use_countnorm=False restores
  exact champion; use_countnorm=True has factor≡1 at ep0.
- R2 CONFIRM (H0088 supported, new champion) if best val MAE **< 19.45**; run 2nd seed if first best < 19.40.
- R3 WEAK-KEEP (tie ≈19.6; count-consistency saturated/informative) if **19.45 ≤ best ≤ 20.0**.
- R4 FAIL (refutes H0088) if best **> 20.0**. Early-stop: ep16+ same-epoch ≥ +1.5 worse than parent.
# N0009_h0008 — use_hires post-decoder detail residual (child of N0002_h0001)

Parent: N0002_h0001 (live parent, EMA val MAE 22.5641 @ep30). Solo single-switch child.
Regime: frozen backbone, head-only; decoder in_ch untouched (192); canonical protocol,
seed 20260830, 32ep/1800s, EMA eval. Bar: confirm iff final EMA val MAE ≤ 22.26
(≥0.30 below 22.5641) AND gt>500 dense-image mean |Δ| below the parent 511.7 baseline.

## 0. Booked hypothesis (runner-parseable — exactly one such line in this file)

1. **H0008** — IF high-resolution decoder detail residual via use_hires IN a solo single-switch child of live parent N0002_h0001 trained under the canonical protocol at seed 20260830 for 32ep/1800s with the module constructed AFTER all parent modules with append-only RNG and decoder in_ch untouched at 15k added params, THEN final EMA val MAE falls at least 0.30 below live parent 22.5641 to 22.26 or lower AND gt>500 dense-image mean |Δ| falls below the parent 511.7 baseline, BECAUSE a zero-init post-decoder detail pass over joint density logits plus frozen h2-derived fine context restores small-peak sharpness that the single 96x96 cond path smooths away without touching the confirmed similarity readout and without sharing any projection so simprior gradients stay isolated, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol at seed 20260830 AND gt>500 dense-image mean |Δ| is not reduced below the parent 511.7 baseline.

## 1. Survey note (local/research/fallback_verify_next.md Direction 3 family)

- Ref A — SANet scale-aggregation + DME (ECCV 2018): transposed-conv decoder to full
  input resolution recovers fine density detail; DME alone −26.1 MAE vs MCNN on
  ShanghaiTech-A. Lesson: resolution at the density interface is first-order for dense
  scenes. Transplant here as a small post-decoder residual (no DME rebuild, no
  encoder retrain, MSE loss untouched) to stay in budget.
- Ref B — FeatUp model-agnostic upsampling (ICLR 2024): learnable joint-bilateral
  upsamplers guided by the high-res image restore edge-aligned detail without changing
  feature semantics. Lesson: guide the detail lift with high-res structure (here the
  frozen h2 octave) instead of hallucinating peaks from the blurred 96-grid alone.
- Ref C — DAVE detect-and-verify (CVPR 2024) + DM-Count blur diagnosis (NeurIPS 2020):
  Gaussian-smoothed GT plus pixel-independent loss yields blurry maps; distribution
  matching localizes dense regions better. DAVE's verify ablation is second-order
  (+0.50 val), and our own verify-gate death (N0006, see §2) confirms gating is spent.
  Lesson: the complementary half is a resolution residual that gives the decoder
  separable humps to integrate — architecture, not a new loss, supplies sharpness.

## 2. Three-lens reasoning

- Math lens. The density term supervises a 96x96 field (384 input at 1/4 res) under
  MSE with σ=1.5 Gaussian GT. Neighboring dots in dense blocks merge into one
  coarse-grid hump; MSE punishes a sharp-but-offset peak more than a smooth blob, so
  the optimizer settles on under-counted smears — exactly the measured tail (gt>500
  mean |Δ| 511.7, severe under-count; oracle global affine ≤0.34 MAE proves no global
  recalibration headroom). A zero-init additive high-frequency correction on the
  pre-softplus logits changes this local geometry without moving global scale: at
  step 0 the residual is identically zero (output ≡ parent), and every learned
  increment is a signed per-cell peak split/fill the single cond pass cannot express,
  while the GCA count anchor and loss stay untouched.
- Lineage lens. Every tested branch since the confirmed simprior readout (H0001,
  23.293 → 22.5641) operated pre-decoder on the similarity/matching side, and all
  four similarity-family deaths share one signature — interference with the
  confirmed readout or its gradient path: (a) H0003 use_padapt (N0004, +3.00):
  unmodulated cross-attention over 2304 h2 tokens corrupted Condenser K/V;
  (b) H0004 use_simbank (N0005, timeout-null 48.12): second full-grid qproj plus
  75-key max einsum blew the τ_max budget with high-variance gradients;
  (c) H0006 use_h2pool (N0007, +25.28 catastrophic): reusing simprior.qproj let a
  garbage-gradient stream freeze shared q/k at half parent norms and stall temp —
  the shared-projection ban; (d) H0007 use_simcal (N0008, +5.07 then NaN wall):
  undetached S recompute plus std-Jacobian path blew up. The gate side died too
  (H0005 use_verify, N0006, +1.02 used-and-harmful). The decoder interface itself —
  decoder in_ch still 192, head block never modified — is the one load-bearing
  surface with zero tested interventions: the remaining headroom. This proposal
  obeys the ban by construction (own parameters only, frozen-h2 read-only context,
  detached inputs, no S recompute, no shared projection), so its clean null reads
  as mechanism rejection, never as readout corruption.
- Low-cost / counter-intuitive lens. Spending the smallest budget on the output side
  rather than the matching side: ≈15.4k params (0.05% of the 31.35M node, total
  ≈31.37M ≤ 32M), ≈0.1 GFLOPs at 96x96 (no 192-grid tensors, no einsum, no second
  qproj — the three budget killers of the dead branches), learned entirely inside
  τ_max parity. Counter-intuitive because it deliberately throws away the richest
  signal (the similarity volume S) and uses only two impoverished streams — bare
  logits plus a 24ch h2 guide — on the theory that the bottleneck is rendering
  sharpness, not match evidence. Clean-null criterion is crisp: head norm ≈ 0 with
  train ≈ parent ⇒ resolution was not binding; engaged-but-worse ⇒ decoder-capacity
  rejection; either way the simprior readout (q/k norms, temp trajectory) must match
  the parent, else the run is a harness failure, not a verdict.

## 3. Targeted change spec (ONE change: use_hires, ≈15.4k, post-decoder only)

- Switch: `use_hires = false` default in config.toml (parent-identical when off).
  Module `HiResDetail` constructed AFTER all parent modules (append-only RNG last);
  flag-off path skips it and reproduces the parent forward byte-for-byte.
- Inputs (both stop-gradient detached at routing): pre-softplus decoder logits
  `L` (B,1,96,96) — the decoder forward is split into `logits = head(block(x))`
  then `softplus`, with identical op order so flag-off output is bit-identical —
  and frozen `h2` (B,192,48,48) read-only via `.detach()` (backbone already frozen;
  detach additionally blocks any second gradient path through the guide).
- Forward: `g = Guide(h2sg)` → bilinear ×2 to 96x96 (B,24,96,96);
  `j = cat([L.detach(), g], 1)` (25ch) → `Ref1(25→24, 3x3, GN, GELU)` →
  `Ref2(24→24, 3x3, GN, GELU)` → zero-init `Head(24→1, 1x1)` → residual `r`
  (zeros at init); output `dens = softplus(L + r)`. GCA path (GAP(fine)+e_mean)
  untouched; decoder in_ch stays 192; only `out["density"]` feeds the loss.
- Param accounting: Guide 192→24 1x1 (4632) + GN (48) = 4680; Ref1 25→24 3x3
  (5424) + GN (48) = 5472; Ref2 24→24 3x3 (5208) + GN (48) = 5256; Head 25.
  Total 15,433 ≈ 15k card. No transposed conv (bilinear only — no checkerboard,
  no extra params vs the 20–40k Direction-3 draft, same separable-peak motive).
- Falsifier: the booked line in §0 (dual bar 22.26 global + 511.7 dense-tail,
  same-seed canonical). Diagnostics at verdict: head/out norms (null vs engaged),
  simprior q/k norms + temp vs parent (isolation check), dense-image count probe
  direction (recall, not scale shift).

# N0008_h0007 — causal feedback (H0007 `use_simcal`)

- **Node:** N0008_h0007 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Result:** 27.6371 @ep09, 32/32ep stepped, `budget_hit=false`, elapsed 1797.8 s · **Verdict: REFUTE** (misses 22.26 bar by +5.38, then diverges to NaN ep10–32).
- **Scope:** this file attributes the failure to a mechanism and rules out harness. Numbers verdict lives in `quant.md`; trajectory shape in `qual.md`. No confidence math, no bookings (synthesis owns those).

## 1. Checkpoint probe (server `run/latest/best.pth`, epoch 9, best_mae 27.6371)

Probed via `ssh cac-server` (`/data/miniconda/envs/cac/bin/python`, `torch.load` CPU, `state_dict` L2 norms — same method as N0007 causal §1, so cross-node comparison is apples-to-apples):

| Param | N0008 (27.64 @ep9) | N0002 parent (22.56 @ep30) | N0007 (47.84 @ep32) |
|---|---|---|---|
| `head.simcal.out.weight` (64,2,1,1) | **0.2743** (init 0.0) | — | — |
| `head.simcal.out.bias` (64,) | 0.0518 (init 0.0) | — | — |
| `head.simcal.tau` | **0.8888** (init 1.0) | — | — |
| `head.simprior.qproj.weight` | **4.7036** | **5.7314** | 3.5155 |
| `head.simprior.kproj.weight` | **4.9405** | **7.3547** | 3.5428 |
| `head.simprior.temp` | **0.0255** (init 0.07) | **0.0393** | 0.0547 |
| `head.simprior.out.weight` | 0.2187 | 0.5874 | 0.3611 |
| `head.simprior.out.bias` | 0.0518 | 0.0730 | 0.3315 |
| `head.decoder.head.weight` | 0.9536 | 1.2862 | 2.5853 |
| `head.cond.out.weight` | 4.1774 | 3.2677 | 5.6698 |

Per-channel detail: `simcal.out` ch0 0.1604 / ch1 0.2225 (calibrated-margin channel leads); `simprior.out` ch0 0.1538 / ch1 0.1555 (both equal-small, vs parent 0.4158/0.4150 equal-large). `simcal.out.bias` and `simprior.out.bias` are **bit-identical** (`torch.equal` True, maxabsdiff 0.0); `simcal.out.weight` vs `simprior.out.weight` cos 0.9127 (maxabsdiff 0.044).

## 2. Structural fact: the second gradient path is real (code-read, not inferred)

`model.py:180-187`: under `use_simcal`, forward recomputes `q = norm(simprior.qproj(fine))`, `p = norm(simprior.kproj(e))`, `S = einsum(q,p)` and adds `simcal(S)` into `cond_map`. There is **no `.detach()`** on `S`, `q`, `p`, or the simprior projections. Autograd therefore flows from `simcal.out` backward through the `mu/sd` reductions, the SNR-softmax, and the `Z` normalization into the **shared `simprior.qproj/kproj`** — a second gradient path into the confirmed H0001 readout, exactly the risk N0007 synthesis §4.2 flagged ("autograd from `simcal.out` through recomputed `S`"). Read-only input tensors (`fine`, `e` never gated) do not block this: the path is through the projection *parameters*, not the inputs.

Two corollaries confirmed by the probe: (a) the identical biases are *expected*, not a bug — both `out` biases add into the same `cond_map`, so `dL/db` (summed shared upstream over B,H,W) is bit-identical every step from the same zero init with the same optimizer state; it independently proves **both branches were engaged** (bias moved 0.0→0.0518). (b) The 0.91 weight-cos is expected redundancy — `dL/dW = upstream · evᵀ` shares the upstream, so weights track each other to the extent the evidence pairs (`c_mag`≈`cons`, `c_cal`≈shape-variant of `top1`) correlate.

## 3. Training-curve causality (TB series, quant §2 / qual §1 summarized for attribution)

- Zero-init held at step 0 (ep1–5 child tracks parent within ±0.6), then **learned separation** ep6–9 (gap +0.5→+0.75→+0.95→+1.35, widening every epoch) — divergence was learned through the new path, not an init break.
- Pre-collapse distress: child `val/rmse` bottoms ep6 (95.48) then **rises** ep6→9 (+1.98) while `val/mae` still falls — tail-first mass misallocation before the blow-up (same RMSE/MAE-decoupling family as the sibling, here as prelude not garnish).
- Hard wall: `train/loss` NaN from step 2249 (inside ep10); all epoch tags (`val/mae`, `val/rmse`, `train/loss_epoch`) NaN ep10–32. Train+val synchronized ⇒ forward/loss numerics, not an eval glitch. Parent under the identical harness is finite all 32 epochs.

## 4. Mechanism verdict (attribute, then rule out)

- **Primary: second-gradient-path interference + numerical instability of the calibration backward pass — H0007 refuted as built.** The probe shows the predicted distortion signature: shared `qproj` −18% (4.70 vs 5.73) and `kproj` −33% (4.94 vs 7.35) suppressed off the working point; confirmed-readout trust projection starved to 37% (0.22 vs 0.59); `simprior.temp` over-sharpened ÷1.54 past the parent (0.0255 vs 0.0393, ÷2.75 from init) toward N0006-collapse territory (0.0214) — the opposite sign from N0007's stall (0.055), i.e. distortion not starvation. Meanwhile the new branch engaged off its clean null (`out` 0→0.27, `tau` 1.0→0.89, calibrated channel leading), so this is engaged-and-harmful, not ignored. The backward pass through `S.std()` carries `1/sd` and `1/sd²` Jacobians; combined with a ÷2.75-sharpened softmax (`1/temp` Jacobian gain) and double-counted gradients into the same projections, per-cell flat-`S` regions produce explosive updates → NaN at step 2249. The calibration hypothesis ("anisotropic-baseline removal + SNR re-weighting adds counting signal beyond raw simprior") is therefore refuted in this implementation: the module moved every diagnostic the wrong way (q/k down, trust down, temp past the working point, RMSE up pre-collapse, MAE +5.07, then non-numeric).
- **Family consequence for synthesis: N0007 §4.2's conditional fires, with a distinct signature.** N0007 predicted: "if N0008 also suppresses q/k or stalls temp, the family expands from 'second similarity bank' to 'any second gradient path into simprior projections'." q/k ARE suppressed (−18%/−33%, well above N0007's frozen ~3.5 floor but well below the 5.7/7.35 working point), so the expansion holds — but the temp sign differs (over-sharpen 0.0255 vs stall 0.0547) and the terminal mode differs (NaN wall at ep10 vs finite stall 47.84). Record as: second-path interference confirmed in two flavors — N0007 starvation-via-garbage-keys, N0008 distortion-plus-instability-via-calibration-backward. Both implicate the shared-projection pattern, neither implicates h2 specifics.
- **Ruled out — clean null (calibration ignored):** `simcal.out` 0.27, `tau` moved −0.11, biases off zero, calibrated channel dominant. The optimizer used the branch; it hurt.
- **Ruled out — N0007-style frozen starvation as the whole story:** q/k at 4.70/4.94 are 34–40% above N0007's 3.52/3.54 floor; temp went sharper (0.0255), not stalled (0.0547); `simprior.out.bias` stayed small (0.05) vs N0007's 0.33 drift. Different attractor.
- **Ruled out — decoder/cond growth as prime mover:** `decoder.head` 0.95 UNDER parent (1.29, vs N0007's compensatory 2.59); `cond.out` 4.18 mildly above parent (3.27, vs N0007's 5.67). No compensatory doubling — features broke before the readout could compensate.
- **Ruled out — harness / protocol / bookkeeping:** server `sha256sum` of `config.toml` → `87ed8f2fbf33c621…` and `model.py` → `a2488efa51ec0f15…` match `result.json` `config_sha256`/`model_sha256` first-16 exactly (local shas byte-identical to server); server config has `seed = 20260830`, `augment = false`, `use_simcal = true` on `use_simprior = true` (solo switch); 32/32 epochs stepped, `budget_hit=false`, `elapsed_s` 1797.8 < 1800, status `done`. Same-seed paired contrast vs 22.5641 is valid; the +5.07-then-NaN is model, not harness. (Minor: no server-side `run/latest/result.json` — record lives at node root; sync artifact only.)

## 5. Causal ordering for synthesis

`recomputed-S without detach opens second path into simprior.qproj/kproj (§2) → calibration-backward (1/sd Jacobians) + sharpening temp (1/temp gain) double-drive the shared projections → q/k pulled off working point (−18%/−33%), trust projection starved (37%), temp over-sharpened past it (0.0255) → ep6–9 MAE gap widens while RMSE already rises → gradient explosion at step 2249 → synchronized train+val NaN ep10–32.` Zero-init guaranteed step-0 equality but, as in N0007, could not protect the shared parameters once training started.

## 6. Redirect (no bookings — synthesis scope)

- Do NOT retry any variant that backprops through `simprior.qproj/kproj` (recomputed-`S`, shared-query, or statistics-over-live-`S` without gradient blockade). N0007's proposed gotcha line ("reuse of similarity evidence is recompute-detached or statistics-only with gradients blocked") covers this node retroactively: SimCal was statistics-only in the forward but not in the backward.
- The narrow surviving question — whether detached statistics (`S.detach()`, own temperature, no path into q/k) calibrate without destabilizing — is a synthesis decision with a new falsifier, not taken here. Any such revisit must pre-register q/k-norm + both-temp diagnostics (this file's Table §1 is the template) and a NaN-collapse stop rule, since the failure mode here is non-finite, not just worse-finite.

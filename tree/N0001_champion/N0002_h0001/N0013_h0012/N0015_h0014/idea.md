# H0014 idea — energy-peakiness-gated per-cell discount of diffuse background texture (`use_peakcal`)

Parent N0013_h0012 (H0012 cellcal CONFIRMED) · SOLO single-switch card (`use_peakcal`, default false; build-time assert it requires `use_simprior`) · frozen backbone, head-only training · decoder in_ch untouched (192 = 128 + 64) · protocol v2 (augment=true via `--set augment=true`, 32ep/1800s, EMA eval) + seed FIXED 20260830 · triple bars (conjunctive): THEN final EMA val MAE <= 20.5861 (0.30 below live 20.8861) AND gt>500 dense-tail mean |del| below 421.13 AND gt<50 sparse slice at or below 5.45 — per §6 bars semantics these numbers re-instantiate from the live v2 parent at verdict time; the hypothesis text itself is never edited.

## §0 Booked-hypothesis placeholder (Lead: pass verbatim to `discovery hypo --new`)

IF energy-peakiness-gated per-cell discount (use_peakcal) IN the live N0013_h0012 head under v2 protocol seed 20260830, THEN final EMA val MAE falls at least 0.30 below 20.8861 AND gt>500 dense-tail mean |del| falls below 421.13 AND gt<50 sparse slice falls at or below 5.45, BECAUSE the fixed-shape diffuse field f = ReLU(mhat-1) * flatness(peak) fires exactly on flat elevated texture that cellcal amplified while vanishing on peaked object centers, and the bounded gain in [0.7, 1] with learned strength w < 0 discounts precisely the background-texture mass, DISPROVED IF final EMA val MAE is not at least 0.30 below the live v2 parent or dense-tail mean |del| is not below the live dense baseline or sparse gt<50 slice exceeds 5.45 or w_final stays near zero or non-negative or any dense-block prediction drops more than 3 percent vs parent.

## §1 Survey note (web-verified 2026-09-21, no closed-door derivation)

Claim grounded: background texture carries diffuse density energy that must be discounted without touching peaked object energy — and the H0013 verdict (pos top1 mean -0.12, matcher blind) bans every match-involving correction, leaving the energy side as the only admissible substrate.

- L2HCount (Xu et al., arXiv:2503.12935 — verified: abs page resolves to "L2HCount: Generalizing Crowd Counting from Low to High Crowd Density via Density Simulation"; Dual-Density Memory Encoding Module with LDCM/HDCM learns low- vs high-density patterns separately because perspective makes diffuse and peaked regimes coexist in one image): precedent that diffuse-vs-peaked separation is load-bearing. Our card is its frozen-head per-cell analog: a peakiness gate instead of a memory bank.
- CREAM (Xu et al., Image and Vision Computing 161, 2025, doi:10.1016/j.imavis.2025.105632 — verified: DOI resolves via ACM/researchr to "CREAM: Few-shot Object Counting with Cross REfinement and Adaptive density Map"; Cross Refinement module weakens background information in the query image; code github.com/CBalance/CREAM): precedent that query-side background suppression is separable from matching. Our card is its energy-side counterpart: discount from local peakiness statistics, no background prototypes, no second match.
- CRNet (Liu et al., IEEE TIP 29, 2020, doi:10.1109/TIP.2020.2994410 — verified: DOI resolves to "Crowd Counting Via Cross-stage Refinement Networks"; progressive refinement of predicted density maps against background clutters with hierarchical density priors; code github.com/lytgftyf/Crowd-Counting-via-Cross-stage-Refinement-Networks): precedent that predicted-density errors concentrate on crowd/background-ambiguous areas and are corrected from density-side priors. Our card applies the correction pre-decoder as a bounded gain rather than post-hoc stages.
- RAZ-Net (Liu et al., CVPR 2019, doi:10.1109/CVPR.2019.00131 — verified: DOI resolves to "Recurrent Attentive Zooming for Joint Crowd Counting and Precise Localization"; learned density deviates from true person density, ambiguous regions re-inspected at resolution): precedent that flat/ambiguous density must be treated differently from peaked confident density. Our peakiness ratio is the single-pass static version of that distinction.
- Deliberately NOT cited: temperature/entropy/matching literature (all banned by the H0013 verdict for this card), divisor schedules, distributional statistics. This card computes no K-softmax, no entropy, no per-image distributions — one 3x3 avgpool and pointwise ops on detached energy.

## §2 Exact math + attach points (parent `tree/N0001_champion/N0002_h0001/N0013_h0012/model.py`)

Parent facts: backbone frozen (`requires_grad_(False)` model.py:35, `train()` keeps net eval :45-48); `CountingHead.__init__` (:147-161) reads non-module flags `use_simprior` (:159) and `use_cellcal` (:161); `CountingHead.forward(h2, h3, bboxes_in)` (:163-180) computes `fine` (:165, (B,128,96,96)), `e` (:168), `cond_map` (B,64,96,96) (:170), SimPrior residual branch (:173-177), decoder concat `torch.cat([fine, cond_map], 1)` (:178, 128+64=192). `SimPrior.forward` (:217-232): `q`/`p` own projections (:219-220), `S` cosine volume (:221, (B,K,96,96)), K-softmax `W` with `self.temp.clamp_min(1e-3)` (:222), `top1` (:223), `cons` (:224), `ev = cat([top1, cons])` (:225, (B,2,96,96)), cellcal gain (:226-231), `return self.out(ev)` (:232). `CellCal` owns one zero-init scalar (:235-246). `Counter.__init__` (:249-272) attaches SimPrior (:266-267) then CellCal (:272) AFTER every parent module — the append-only RNG pattern (AGENTS.md rule 13) this card copies. Config: seed 20260830 (config.toml:9), `use_simprior=true` (:27), `use_cellcal=true` (:28), `augment=false` live (:32; v2 runs add `--set augment=true`).

PeakCal (`use_peakcal`, ONE zero-init scalar `w_p`, D=0.3 and sigma=0.25 FIXED constants, never learned), all shapes for S=384, batch B:

```
# --- own diffuse-texture field from detached fine energy (B,128,96,96); decoupled from cellcal branch ---
m    = fine.detach().mean(dim=1, keepdim=True).clamp_min(0)          # (B,1,96,96)
mbar = m.mean(dim=(2, 3), keepdim=True)                              # (B,1,1,1)
mhat = (m / (mbar + 1e-6)).clamp(0, 10)                              # mean ~1 by construction
loc  = F.avg_pool2d(mhat, 3, stride=1, padding=1)                    # (B,1,96,96) local mean
peak = (mhat / (loc + 1e-4)).clamp(0, 5)                             # >1 peaked center, ~=1 diffuse flat
elev = (mhat - 1).clamp(0, 5)                                        # elevated-only (ReLU via clamp)
flat = torch.exp(-((peak - 1).square()) / 0.125).nan_to_num(0.0, 1.0, 1.0)  # sigma=0.25 fixed -> 2s^2=0.125; ~1 iff peak~=1
f    = (elev * flat).clamp(0, 5)                                     # (B,1,96,96) pure-data diffuse field
arg  = (w_p * f).clamp(-3, 3)
gain = torch.exp(arg).clamp(1 - D, 1).nan_to_num(1.0, 1.0, 0.7)      # D=0.3 FIXED; NaN falls back to identity
ev   = ev * gain                                                     # broadcast over the 2 evidence channels
out  = self.out(ev)                                                  # existing zero-init projection; (B,64,96,96)
```

Attach (minimal-diff DECISION — two trailing kwargs on `SimPrior.forward`, no refactor of its internals): signature becomes `forward(self, fine, e, use_cellcal=False, w_c=None, use_peakcal=False, w_p=None)`; body inserts the peakcal block after the cellcal block (:231) before `return self.out(ev)` (:232). Defaults keep every existing call (:175/:177) valid and parent-path byte-identical. `CountingHead.forward` flag-on path (nested inside the `use_simprior` branch; `fine` already in scope at :165): pass `use_peakcal=self.use_peakcal, w_p=self.peakcal.w_p` through alongside the existing cellcal kwargs. `CountingHead.__init__` reads `self.use_peakcal = _get(cfg, "use_peakcal", False)` (non-module flag, no RNG — same pattern as :159/:161). `Counter.__init__` constructs `self.head.peakcal = PeakCal()` LAST, after the CellCal attach (:272); `w_p` zero-init draws no RNG. PeakCal holds NO normalization layers, so train/eval agree exactly.

Temp-pin hygiene (6th collapse instance): when `use_peakcal` is on, the Coding Agent sets `self.simprior.temp.requires_grad_(False)` at construction AND the W line (:222) uses `self.temp.detach().clamp_min(1e-3)`. Detach is value-identical, so the step-0 identity proof below is unaffected; the free-temp collapse path under joint perturbation is structurally closed.

Step-0 byte-identity proof (for the Coding Agent — must hold before smoke): (a) `w_p` init exactly 0.0 ⇒ `arg` exact zeros ⇒ `exp(0)==1.0` exactly ⇒ `clamp(1.0, 0.7, 1.0)==1.0` ⇒ `ev * gain == ev` bit-for-bit (fp16 identity holds: x*1.0==x) ⇒ `self.out(evp) == self.out(ev)` ⇒ flag-on-at-init forward equals the parent; temp detach changes no values; (b) with `use_peakcal=false` (default) no new line executes — forward is the parent's byte-for-byte; (c) def-use: `self.head.peakcal` defined in `Counter.__init__` before any forward; `fine` in scope at :165; Condenser path untouched; (d) no shared state: `w_p` lives under `self.head.peakcal` only, never aliasing `simprior.*`, `cellcal.*`, or `cond.*`; the `m/mhat/peak` field is recomputed from `fine.detach()` and never reuses (or writes) cellcal locals; (e) smoke must assert: flag-off == parent AND flag-on-at-init == flag-off (max abs diff 0). Params +1 scalar, far under the head budget.

Finite-guard inventory (EVERY division/exp/norm listed — the H0013 NaN lesson): (1) `mbar + 1e-6` denominator (copied from parent :229); (2) `loc + 1e-4` denominator in peak; (3) `mhat` clamped [0,10], `peak` clamped [0,5], `elev` clamped [0,5], `f` clamped [0,5] — no unbounded field ever reaches a mult/exp; (4) flatness `exp` of a non-positive argument (in (0,1] by construction; worst case `exp(-128)` underflows to clean 0.0, never NaN/Inf) wrapped in `nan_to_num`; (5) `w_p * f` clamped [-3,3] before `exp` (output in [0.05, 20.1], no overflow in fp16/fp32) and gain clamped [0.7, 1.0] then `nan_to_num` with nan→1.0 (any pathology defaults to no-discount identity); (6) temp path keeps parent `clamp_min(1e-3)` plus detach; (7) NO new normalization layers, NO softmax, NO log — the only reduction is the parameter-free `avg_pool2d`, which is exactly deterministic train/eval.

Structural-twin pre-emption: NOT H0013 `negsup` (that card built background prototypes + a second match volume + K-max; this card has no prototypes, no projections, no einsum, no match of any kind — a pure function of detached fine energy, so match-blindness cannot touch it); NOT H0011 `msq` (no queries, no attention, no feature path — a bounded gain on confirmed evidence, so the recall-destruction signature is structurally unavailable); NOT an F5 density-logit refine (intervention is pre-decoder on `ev`, decoder in_ch stays 192); NOT §11 exemplar gating (`e` is read-only cosine input inside the simprior match and is never rescaled, gated, or written).

## §3 Why-not (H0013 match-blindness + H0009 gate lesson + H0011/H0013 collapse + §11)

- H0013 match-blindness honored by construction: the field uses `fine.detach()` ONLY — `S`, `top1`, `W`, `cons`, `e`, and `h3` never appear. Pos top1 mean -0.12 is irrelevant because no threshold, margin, or gate reads any match quantity; the diffuse/peaked distinction lives entirely in energy texture statistics.
- NOT an H0009-style dead gate: `w_p` scales an ALREADY-COMPUTED diffuse field `f`, exactly the sanctioned pattern — `f` is fully computed in the forward pass from frozen features regardless of `w_p`, and `w_p`=0 is the identity. Gradient is nonzero from the first backward pass (dL/d`w_p` = sum(dL/d`ev` · `ev` · gain' · `f`)), so `w_p` escapes zero at once if discounting helps; the avgpool path carries no parameters and must open nothing before learning. Learning sets only the STRENGTH of a fixed-shape field, never opens or closes a representation path.
- H0011/H0013 collapse excluded by two independent bounds: peaked centers carry `flat`→0 hence `f`≈0 and keep gain≈1 for ANY `w_p` (selectivity is structural, not learned), and even fully-active cells saturate at gain 0.7 (max 30% local discount — a crater toward zero is arithmetically unavailable).
- §11 compliance (explicit): §11 bars pre-condenser exemplar gating in ANY granularity. This card never gates, rescales, adapts, or writes `e` — the exemplar path through the Condenser is byte-untouched, and peakcal holds no exemplar parameters. The intervention is an energy-side gain on `ev` — a similarity-side/decoder-side live direction per §11 ("query/similarity-side and decoder-side — unproven, re-book fresh"), not a settled one.

## §4 Predictions + failure modes + 4 reads

Verdict context: live parent N0013_h0012 v2 20.8861 @ep32; dense gt>500 mean |del| 421.13; sparse gt<50 5.771 vs v2-baseline 5.451 (+0.32 guard miss); H0013 refuted (best.pth 23.266, dense 511.67, inverted selectivity, NaN wall ep23 — model path under AMP, data path exonerated).

- P1 (sparse correction): gt<50 slice 5.771 → ≤5.45 — the diffuse field fires on the flat elevated texture cellcal amplified (w_c=+0.1658), discounting exactly the false mass.
- P2 (dense neutrality): dense-tail holds below 421.13 with dense-block predictions (3425/3428/3433) intact — peaked centers carry f≈0 structurally, distinguishing from H0011-style collapse.
- P3 (overall): final EMA val MAE ≤ 20.5861. All three bars conjunctive.
- F1 (quiet null): `w_p_final` within ±1e-2 of zero ⇒ field never engaged ⇒ REFUTE even if a bar flickers.
- F2 (dense collapse): any dense-block prediction drops >3% vs parent (3425/3428/3433) or dense-mean gain < 0.9 ⇒ H0011/H0013 signature ⇒ REFUTE.
- F3 (sparse unchanged): sparse gt<50 stays above 5.45 despite `w_p` off-zero ⇒ discount misfires (sigma scale or peak/flat misdiagnosis of the texture driver) ⇒ REFUTE with diagnosis; do not re-book energy variants silently.

Mechanism reads at verdict (Lead hook harness):
1. `w_p_final` ≤ -1e-2, decisively off-zero with CORRECT sign (diffuse discounted, peaked kept).
2. Sparse gt<50 slice ≤ 5.45 (the +0.32 eaten — the entire point).
3. Dense-block recall preserved: each of 3425/3428/3433 within 97% of the parent prediction AND dense-tail below the live dense baseline (no collapse toward zero).
4. BOTH MAE bars: final EMA val MAE ≥0.30 below the live v2 parent AND the dense-tail bar jointly.

## §5 Risks + same-seed v2 pair protocol + futility

- Risk: sigma=0.25 mis-scaled (true texture flatness wider/narrower than ±0.25 in peak units) — bounded by F3 as the refutation path with full diagnostic value; sigma stays a fixed constant under phase-1 cost discipline, never tuned post-hoc.
- Risk: `w_p` dead (diffuse field uncorrelated with residual) — cheap F1 null with full diagnostic value (reads 2–4 still valid on best.pth).
- Risk: avgpool border artifacts at 96×96 edges — one-cell rim, negligible mass; no guard needed beyond the clamps.
- Risk: temp freeze removes a parent degree of freedom (parent temp trajectory learned) — accepted deliberately: the 6th collapse instance proved free temp collapses under any joint perturbation, and pinning is value-identical at step 0.
- Risk: extra forward ops (one avgpool + pointwise on 96×96) — negligible against the backbone; no RNG draws, no buffers, hence no train/eval skew.
- Protocol: SAME-SEED v2 pair only — child (`use_peakcal=true` over the parent flags, `--set augment=true`) vs live N0013_h0012 v2 (`--set augment=true`), seed 20260830 FIXED, 32ep/1800s, EMA eval, `final EMA val MAE` + `gt>500 dense-tail mean |del|` + `gt<50 sparse slice`. NEVER cross-protocol: pre-v2 (augment=false) numbers are historical record only. Verdict binds to live v2 numbers at run time (§6 semantics); booked 20.8861/421.13/5.45 and any re-instantiated live values are both stated in the evidence note. `repro_run --set seed=` is FORBIDDEN without explicit user approval.
- Futility (rule 16): launch carries `--set futility_bar=20.5861`; gates ep16 WARN-only / ep24 HALT; feasibility must be visible by ep24 — the discount is a one-scalar correction whose signal (`w_p` leaving zero with negative sign in early-epoch hooks) appears in the first epochs if the mechanism is real.

# H0017 idea — peak-gated additive mass placement (use_peakmass), SOLO child of N0015_h0014

- Parent: `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014` (LIVE, v2 seed 20260830, 32ep/1800s EMA eval, val MAE 19.3431 @ep32).
- Switch: SOLO `use_peakmass`, default false. Exactly one testable change from the parent.
- Regime: frozen backbone, head-only training. Decoder (`DensityDecoder`, `model.py:128-140`) untouched — `in_ch` stays `D + cond_dim`; no parent module shape/init touched.
- Protocol: v2 + fixed seed 20260830 (`config.toml:9`), same-seed paired contrast per §6 (precision ~0.02; level noise ~±1 MAE never compared across seeds).
- Triple conjunctive bars (bind live N0015 under current protocol): THEN final EMA val MAE at least 0.30 below 19.3431 (<= 19.0431) AND dense-tail (gt > 500 mean |del|) below 330.81 AND sparse (gt < 50) at or below 5.45. Launch carries `--set futility_bar=19.0431` (rule 16).
- §6 fidelity: falsifier below is numeric and testable; no hedging in §0.

## §0 One-liner for `discovery hypo --new`

IF use_peakmass peak-gated additive placement IN the N0015_h0014 head THEN final EMA val MAE drops at least 0.30 below 19.3431 with dense-tail below 330.81 and sparse at or below 5.45 BECAUSE sharp-peak-gated additive mass places count mass only at spatially precise energy centers that survive the pixel-MSE veto and recover dense-block recall without similarity evidence DISPROVED IF final EMA val MAE is not at least 0.30 below 19.3431 or dense-tail is not below 330.81 or sparse exceeds 5.45.

## §1 Survey — peak / local-maxima-guided placement and localization-aware counting (verified, resolves live)

All three arXiv IDs below were fetched and resolve (abstract + submission history confirmed 2026-09-21). No temperature/entropy/matching literature is cited.

1. CoDi — exemplar-conditioned diffusion low-shot counter producing narrow-peak density maps counted by NMS local maxima (arXiv:2512.20153, Sustar et al., submitted 23 Dec 2025, v2 16 Jul 2026). Core transferable observation: §IV-G shows blanket density blobs (LOCA-style wide Gaussians) yield inaccurate NMS peaks and counts, while narrow expressive peaks count unambiguously; object centers are detected as local maxima in 3x3 neighborhoods above tau_max = 0.1. This is the direct precedent for gating mass on sharp local maxima rather than diffuse energy.
2. DAVE — detect-and-verify paradigm for low-shot counting (arXiv:2404.16622, Pelhan et al., CVPR 2024). First stage generates a HIGH-RECALL candidate set from density estimation, second stage verifies and rejects outliers, jointly raising recall and precision (~20% MAE over top density counters). Precedent for the two-step logic used here: propose generously from a recall signal, then admit mass only through a precision gate — here the precision gate is peakness, applied inline rather than as a second network.
3. P2PNet — purely point-based counting/localization framework (arXiv:2107.12746, Song et al., ICCV 2021 Oral). Discards density-blob and pseudo-box intermediates; directly predicts point proposals with one-to-one Hungarian matching and a density-normalized AP metric. Precedent for the claim that spatially precise point-like placement beats blob mass on localization-sensitive objectives — the pixel-MSE term in our loss is exactly such a localization-sensitive objective.

What the survey decides: none of these attach a peak-gated additive residual inside a frozen-backbone counting head; the mechanism below imports only the peak-precision insight, implemented as a zero-init additive path so the parent is the exact step-0 control.

## §2 Mechanism — exact math, diagnosis, attach points, step-0 proof, guards, temp-pin

### 2.1 Program bind (the card's raison d'etre) with numbers

- Pixel-MSE vetoes imprecise mass: energy blobs place mass slightly-wrong, so MSE rises more than the L1 count-anchor falls. Spatial precision in this lineage comes only from exemplar similarity, which is dead on collapse images (top1 mean -0.12).
- Gains therefore multiply dead evidence (the H0010/H0012 limit), AND blanket additive blobs get MSE-vetoed (H0016: enmass futility-HALTED ep24, verdict 22.19 at +2.85 over parent pace, dense tail +155 over parent).
- The remaining opening: mass placed ONLY at sharp peaks is MSE-safe because it is spatially precise, and it is energy-derived so it is similarity-independent. That is the single disjoint-support regime neither veto covers.

### 2.2 Exact math (`use_peakmass`)

All ops at the 1/4-res grid (B,1,96,96), recomputed decoupled from `fine.detach()` (guarded ops, same pattern as CellCal at `model.py:235-237`):

```
m      = fine.detach().mean(dim=1, keepdim=True).clamp_min(0)   # (B,1,96,96)
mbar   = m.mean(dim=(2,3), keepdim=True)                        # (B,1,1,1)
mhat   = m / (mbar + 1e-6)                                      # mean 1.0 by construction
loc    = F.avg_pool2d(mhat, 3, stride=1, padding=1)             # local mean
peak   = (mhat / (loc + 1e-4)).clamp(0, 5)                      # >1 center, ~1 flat
peakness = F.relu(peak - 1)                                     # zero on flat texture INCLUDING flat-elevated diffuse background; >0 only on sharp centers
placed = PeakMassProj(mhat * peakness)                          # Conv2d(1,64,1) weight+bias ZERO-INIT, +128 params (64 w + 64 b)
cond_map = cond_map + placed                                    # pre-decoder, additive
```

- `e` untouched; `fine` read via detach only (no gradient path into the trunk, no rescaling of `fine`).
- No match/prototypes/queries/attention/temperature anywhere in the new path.
- `cond_map += placed` lands pre-decoder in `CountingHead.forward` (`model.py:165-188`), same residual interface as the SimPrior attach (`model.py:173-185`).

### 2.3 Pre-emptions (not a retried refutation)

- Why not enmass retried: enmass placed mass proportional to `mhat` EVERYWHERE including diffuse texture — that support is exactly the MSE-vetoed regime (gradient died: proj norm 0.0536 after 24ep, mean weight -0.00079 ~ 0). PeakMass places mass ONLY where `peakness > 0` (sharp centers) — disjoint support from the vetoed regime, different actuator input (`mhat*peakness` vs `mhat`), different claim (precision-gated placement vs blanket), different falsifier (§4 F1/F2).
- Why not peakcal retried: PeakCal (H0014, `model.py:269-281`, math at `model.py:240-251`) is a DISCOUNT — a gain clamped to (0.7, 1] with scalar `w_p ~ 0` null. PeakMass is PLACEMENT — additive with expected `mean weight > 0`. Opposite sign, opposite claim, opposite weight-read falsifier.

### 2.4 Attach points (file:line refs against live parent N0015 model.py)

- `CountingHead.__init__` (`model.py:146-163`): add non-module flag `self.use_peakmass = _get(cfg, "use_peakmass", False)` after the `use_peakcal` line (`model.py:163`).
- `CountingHead.forward` (`model.py:165-188`): after the SimPrior/CellCal/PeakCal residual block (`model.py:175-185`), add `if self.use_peakmass: cond_map = cond_map + self.peakmass(fine)` — pre-decoder, decoder call at `model.py:186` unchanged.
- `Counter.__init__` (`model.py:285-316`): construct `self.head.peakmass = PeakMass()` LAST, after the PeakCal attach (`model.py:312`), append-only per rule 13. New `PeakMass` class sits after `PeakCal` (`model.py:282`).
- `config.toml`: add `use_peakmass = false` default next to `use_peakcal = true` (`config.toml:29`); child flips true.
- Temp-pin hygiene REQUIRED: PeakMass owns no temperature and reads neither `SimPrior.temp` nor similarity; the existing pin block (`model.py:313-316`, `temp.requires_grad_(False)` when peakcal is on) is preserved verbatim and extended — while `use_peakmass` is on, `simprior.temp` stays pinned (no free-temp path reopens through the new module).

### 2.5 Step-0 proof

At init `PeakMassProj.weight = 0` and `bias = 0` (explicit `nn.init.zeros_`), so `placed == 0` everywhere regardless of `mhat*peakness`; `cond_map` is bit-identical to the parent's, the decoder input is unchanged, and forward is the parent's byte-for-byte. Zero-init draws no RNG, and LAST construction keeps every parent init draw at its exact RNG position (rule 13). `use_peakmass = false` skips the branch entirely.

### 2.6 Guards (finite everywhere)

All divisions clamped with eps (`mbar + 1e-6`, `loc + 1e-4`); `peak` clamped to (0, 5); `peakness` is a ReLU (non-negative, no exp); the projection is linear 1x1 (no unbounded exp, no softmax, no log). Append-only LAST construction; +~128 params; decoder `in_ch` untouched.

## §3 Why-not (enmass blanket-vetoed; peakcal discount-null; §11 compliance)

- Enmass (H0016): blanket additive energy residual, futility-HALTED ep24, proj norm 0.0536, mean weight -0.00079 (~0 quiet null). Diagnosis recorded: the `mhat` substrate is EXHAUSTED for blanket use — CellCal already mines it via gain, so the additive path is redundant and gradient ~0. PeakMass does not re-plough that substrate: its actuator (`mhat*peakness`) is zero almost everywhere enmass was live, so a quiet null here reads as a statement about peak support, not a repeat of the blanket veto.
- PeakCal (H0014 live in parent): discount gain `(0.7, 1]` on a diffuse-field `f`; null means discounts cannot create recall. Placement creates recall by construction if weights go positive — the F2 read separates the two outcomes by sign.
- §11 register: nothing proposed touches DDCA, spatial summaries/RGA, final-layer readout, backbone unfreeze, pre-condenser exemplar gating, or the frozen hs(2,3) readout + cross-attn condenser. The condenser (`model.py:111-125`), `FineFuser` (`model.py:51-74`), and `ExemplarEncoder` (`model.py:77-108`) are untouched; the head interface `forward(imgs, bboxes[, bboxes3]) -> {"density", ...}` is unchanged.

## §4 Predictions, failure modes, 4 reads

Predictions if the mechanism is real: dense-block recall rises (block preds move toward gt on 3481/3428/3488/3484/3482/5059/3476/3487/3483/2850) while sparse stays flat (peaks are rare on sparse backgrounds); the projection learns positive mass-placing weights; MAE and dense-tail bars fall together.

- F1 (placement learned): proj weight norm decisively off zero AND mean weight > 0. Reads mass-placing, not null.
- F2 (sign check): mean weight < 0 reads suppressor — refute (opposite claim realized).
- F3 (precision guard): sparse gt<50 slice must not exceed 5.9 during the run; blows past 6.5 at verdict reads peaks misfiring on texture — refute.
- F4 (conjunction): BOTH MAE bars (<= 19.0431, >= 0.30 below live parent 19.3431) AND dense-tail below 330.81 must hold. Any leg failing refutes; near-miss on one leg with the other two green is still refute (triple conjunctive, no partial credit).

Failure modes: (a) quiet null repeat — F1 norm ~= 0 like enmass (then peak-gating does not save it, refute, substrate conclusion extends to peak support); (b) texture misfire — F3 sparse breach (peaks fire on background texture, refute); (c) sign flip — F2 negative (the gate learns suppression, refute).

## §5 Risks, v2 pair protocol, futility_bar note

- Risk 1 — peaks misfire on texture: `peakness > 0` on sharp background texture would place false mass and breach sparse. Mitigations: ReLU(peak-1) is exactly zero on flat-elevated diffuse fields (the dominant false-mass source); F3 watches sparse every epoch with a 5.9 tripwire and 6.5 verdict refutation.
- Risk 2 — quiet null repeat: peak support may be too sparse to move the loss, repeating the enmass 0.05-norm death. This is priced in: F1 is the first read, and a null cleanly extends the substrate-exhaustion diagnosis rather than leaving ambiguity.
- v2 pair protocol: same-seed (20260830) paired contrast vs live N0015_h0014 under v2 (augment, EMA eval, seeded loaders, cudnn deterministic), 32ep/1800s; verdict numbers come from `result.json` + `test_perimage.json` reads only.
- Futility note: launch with `--set futility_bar=19.0431` (rule 16; ref slope 0.06/ep default, ep16 WARN-only / ep24 HALT, margin 2.0). A futility-halted run keeps best.pth: verdict = futility refutation + full eval_test (dense tail + F1/F2/F3 mechanism reads still valid).

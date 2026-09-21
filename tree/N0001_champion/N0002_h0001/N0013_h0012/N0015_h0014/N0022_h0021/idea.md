# H0021 idea — `use_exkern`: exemplar-kernel match + self-calibration residual

Parent: `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014` (live N0015). SOLO switch `use_exkern`, default false. Frozen backbone, head-only. Decoder untouched (in_ch 192 fixed). Protocol v2, seed 20260830 fixed. Triple conjunctive bars (§6 semantics, bound to live parent 19.3431): final EMA val MAE ≤19.0431 (≥0.30 below live) AND dense-tail MAE <330.81 AND sparse gt<50 MAE ≤5.45. Launch with `--set futility_bar=19.0431`.

## §0 Hypothesis one-liner (for `discovery hypo --new`)

IF `use_exkern` exemplar-kernel self-calibrated residual IN live-N0015 head-only child at fixed seed 20260830 under v2 protocol, THEN final EMA val MAE falls at least 0.30 below the live parent (to ≤19.0431) with dense-tail MAE below 330.81 and sparse gt<50 MAE at or below 5.45, BECAUSE full-size spatial exemplar kernels convolved depthwise over fine features plus an ROI self-calibration constraint supply exemplar-structure match evidence that pooled-vector similarity priors discard and restore the training-free tail wins without learned suppression, DISPROVED IF final EMA val MAE is not at least 0.30 below live-parent 19.3431 or dense-tail MAE is not below 330.81 or sparse gt<50 MAE exceeds 5.45 at the same seed.

## §1 Measured motivation: aligned tail wins that transfer

Lead-aligned per-image comparison (OUR val split; trained N0015 vs training-free CountingDINO pipeline, DINOv2 ViT-S):

| img | ours err | cdino err | cdino win |
|---|---|---|---|
| 935 | 1249 | 877 | 372 |
| 865 | 756 | 426 | 330 |
| 5059 | 257 | 20 | 237 |
| 3481 | 409 | 196 | 213 |
| 3482 | 287 | 81 | 206 |
| 2850 | 209 | 16 | 193 |
| 3483 | 220 | 36 | 184 |
| 7656 | 850 | 680 | 170 |

Overall cdino loses (39.70 vs 19.33) — training wins globally — but on OUR catastrophic tail the training-free path wins by 170–370 per image. Verified cdino ingredients (repo source): (1) FULL-SIZE exemplar ROI kernels as conv filters (spatial structure preserved, not pooled vectors); (2) ROI SELF-CALIBRATION (response normalized so exemplar region sums to scaling_coeff — a constraint, not a learned weight); (3) divide_et_impera tiling (inference-only resolution, no training overfit); (4) NO learning → no collapse/suppression/bias. Transferable under training: (1)+(2) as one extra evidence channel. (3) is inference-only and out of scope; (4) is the reason to keep the new path NON-PARAMETRIC (zero learned kernel weights) with exactly ONE zero-init fusion scalar, so training cannot re-learn suppression through it.

## §2 Mechanism: exact math + attach points

Path (per forward, K=3, bboxes3 in S=384 space → fine grid 96×96, s=96/384):
1. `fd = fine.detach()` (B,128,96,96). Never touch `e`.
2. Per exemplar k: box size in fine cells `(rh_k, rw_k) = clamp(box_wh_k * s, 2, 12)` (bounded kernel, no RNG). `RoI_k = roi_align(fd, box_k, output_size=(rh_k, rw_k))` → (1,128,rh_k,rw_k) spatial kernel, structure preserved.
3. Common size FIXED r=4 (determinism + bounded compute; median-clamp rejected as data-dependent shape): `Z_k = adaptive_avg_pool2d(RoI_k, (4,4))`. `Kbar = mean_k Z_k` → (128,1,4,4) depthwise kernel, fully non-parametric (recomputed each forward, zero learned weights).
4. `match = conv2d(fd, Kbar, groups=128, padding=2)` (B,128,96,96); `ch = match.mean(dim=1, keepdim=True)` (B,1,96,96) exemplar-structure response. No temperature, no softmax, no learned projection.
5. Self-calibration (CountingDINO roi-norm analog, FIXED scaling_coeff=1.0, not learned): `c_k = ch.detach()` mean-pooled inside each exemplar box; `ch_cal = ch / (mean_k(c_k) + 1e-6) * 1.0` (exemplar regions average 1.0 by construction).
6. Fuse: `cond_map = cond_map + w_x * ch_cal.expand(-1,64,-1,-1)`, scalar `w_x = nn.Parameter(zeros(()))` (+1 param). CHOICE: scalar broadcast, not ZeroInitConv(1,64,1): conv adds 65 params and a learned per-channel mixer that risks re-learning suppression; scalar is cheapest and identity at init.

Attach (file refs to live parent `model.py`): flag `use_exkern` beside H0014 switch in `CountingHead.__init__` (:146–163); scalar module constructed LAST in `Counter.__init__` after `peakcal` (:312, append-only, rule 13); forward branch at cond_map site (:172–186) `if self.use_exkern: cond_map = cond_map + ...`; decoder `in_ch` untouched (:156).
Step-0 proof: `w_x=0` → added term exactly zero → forward byte-identical to parent. Guards: kernel sizes clamped [2,12], fixed r=4, eps 1e-6, `.detach()` on fd and calibration pool, `nan_to_num` on division, no shape data-dependence. Temp-pin hygiene: SimPrior temp already pinned when peakcal on (:315–316); this path has no temperature/softmax/learned norm — nothing to pin, stated so the coder adds none.

## §3 Why-not + compliance

H0019 lanes learn projections over POOLED `e` — same weak substrate (pooled vectors) plus learned fusion, which is exactly what collapses on the tail. This uses FULL-SIZE SPATIAL kernels (different substrate: spatial structure) + a self-calibration CONSTRAINT (different operation: fixed normalization, not learned fusion). §11: no scalars on `e` — `e` is never touched (boxes + `fine.detach()` only); pre-condenser exemplar-gating ban does not apply (no gating of `e`, no rescaling of exemplar path; additive evidence into `cond_map` like H0001).

## §4 Predictions + failure modes + 4 reads

Predictions: (P1) `w_x > 0` decisively at end of training (path used, not ignored); (P2) 935/865/5059/3481-class catastrophic preds move toward gt vs same-seed parent (transferred wins); (P3) sparse gt<50 holds ≤5.45 (no dense-bias leak); (P4) BOTH MAE bars + dense bar pass conjunctively.
Failures: F1 `w_x ≈ 0` → quiet null, non-parametric path ignored, REFUTE. F2 val MAE worse than parent or sparse blows → harmful, REFUTE. F3 MAE bars pass but dense-tail unchanged → transfer failed, REFUTE.
Reads (all from result.json/test outputs + checkpoint): R1 final `w_x` scalar value; R2 per-image preds on the 8 tail images vs parent; R3 sparse gt<50 MAE; R4 final EMA val MAE + dense-tail MAE.

## §5 Risks + protocol

Risk: transfer may not survive training dynamics (constraint helps inference but gradient pressure through `w_x` alone may under-use it); bounded by futility (hopeless-but-improving caught at ep16 WARN / ep24 HALT). Same-seed paired contrast only (precision ~0.02); never cross-seed at the 0.30 bar. v2 pair protocol: v2 child vs v2 live parent at seed 20260830. Futility: launch carries `--set futility_bar=19.0431`. No v2-pair, no verdict.

## Citations

- Pacini et al., CountingDINO, WACV 2026, arXiv:2504.16570; repo github.com/lorebianchi98/CountingDINO (reproduced pipeline + aligned tail-win table = grounding).
- Ranjan et al., FamNet, arXiv:2104.08391 (origin of exemplar-kernel correlation maps: correlation between exemplar and image features as density-predictor input).

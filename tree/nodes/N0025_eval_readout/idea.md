# Idea — N0025_eval_readout (parent: N0021_dino_partialft; eval-only, zero training, zero params)
Champion checkpoint: `/data/runs/N0021_dino_partialft/best.pth` — val MAE 20.44 / RMSE 83.06.
Two targeted EVAL-ONLY readout changes on this frozen checkpoint; weights untouched. Targets the
research-ranked #1 inference lever (docs/research_notes_2026-08-24.md: TTA = biggest lever).

## Change 1 — TT-Norm per-image gain (tests H0033)
In FSC147 GT each exemplar box holds exactly ONE object ⇒ predicted density integral inside an
exemplar box should be ≈1; systematic deviation is a scalar gain drift of the intensity readout.
Per image:
  1. One champion forward @392 → raw rho [1,28,28] (grid ps=392/14=28, model.py:71).
  2. For EACH of the 3 exemplar boxes from `annotation_FSC147_384.json`
     (`box_examples_coordinates`, rescaled to S-space exactly like code/data/fsc147.py:57-61;
     NOTE the Dataset returns ONLY exemplar[0] as `bboxes` — the lab script must read all 3
     boxes from the annotation JSON directly):
     `integral_k = sum(rho over cells whose centers fall inside box_k)`.
     If ALL three integrals < 1e-6 → SKIP the gain for this image (keep raw prediction).
  3. `g = median_k(1 / max(integral_k, 1e-6))`; clamp g to [0.2, 5]; `rho' = g * rho`;
     `N_hat_ttnorm = g * N_hat_raw`.
Precedent: CountGD NeurIPS'24 ablation none→TT-Norm **val 8.69→7.99**, test 10.92→9.62;
CounTR TT-norm "significant boost" — exemplar-region density calibration is an established,
parameter-free val/test win we have never exercised.

## Change 2 — split-half isotonic recalibration + trimmed-sum arms (tests H0034)
MSE-trained heads regress toward the count prior (attenuation slope < 1); the L1-optimal readout
is the conditional MEDIAN, not mean ⇒ a monotone post-hoc map can recover it. On the SAME pass:
  - Split val deterministically into halves A/B (sorted img_id, even/odd or stable hash).
  - Fit `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` on (N_hat, N_gt) of ONE half,
    apply to the OTHER; report BOTH directions' MAE pre/post (A-fit→B and B-fit→A).
  - Regularize to identity outside p10–p90 of the fitting half's predictions (blend
    curve→identity outside the band) so sparse/dense tails are not extrapolated.
  - Variant arms (RMSE-focused, orthogonal): trimmed-sum readout of raw rho — drop top
    {0.5%, 1%, 2%} hottest cells, then sum.

## Leak analysis (binding for coding agent)
- Exemplar boxes are LEGITIMATE inference inputs: the model already consumes them as prompts
  (model.py forward(imgs, bboxes), model.py:68-83). Gain calibration adds NO label information.
- Isotonic MUST stay split-half: fitting and evaluating on the same val images overfits val and
  invalidates the comparator. Verdicts use cross-fit numbers ONLY.
- Test-set deployment (explicit rule): refit isotonic on FULL val and apply to test ONLY IF
  split-half confirms stability (both directions improve ≥0.3 AND fitted curves agree within
  tolerance); otherwise ship raw counts. Val labels never touch test beyond this sanctioned map.
- Trimmed-sum arms use no labels at all — deployable unconditionally if RMSE improves.

## Deliverable shape
One val pass dumps per-image JSONL (spec in tasks/T0004_pending_coding_N0025.md); every verdict
is computed OFFLINE from the dump — readout variants never need re-running the model.

## Risks / disproof informativeness
- Partial FT may already have absorbed most gain drift → >50% |g−1|<0.01 ⇒ H0033 disproved but
  that localizes where the 20.44 error lives (shape, not scale).
- 28×28 grid quantizes exemplar integrals (border-straddling cells); median-of-3 mitigates.
- Single-seed noise σ≈±0.3 MAE (N0021 synthesis #2): gaps below ~0.3 are not evidence — mirrors
  H0034's disproof floor of 0.3.

# N0029_tf_champion — training-free seed root (PIVOT)

## Why this tree exists

The supervised lineage under `N0001_champion` is **closed for train-time SOLO cards**:
six consecutive futility-HALTs (N0023–N0028 / H0022–H0027), every card in the
+2.7…+3.9 envelope vs live parent N0015 (19.3431). Amplitude/scale sites
(sign, substrate, residual, decoder-FiLM, similarity-ME, fine) are exhausted.

**User pivot 2026-09-22:** new direction = training-free / inference-side
(no further supervised training on this arc unless user re-opens).

This root is a **second filesystem root** under `tree/` (parent=null).
It is NOT nested under N0001_champion. Reference numbers from the supervised
tree may be quoted as external anchors; they are never this root's parent.

## Seed baselines (this root's starting metric set)

| Method | Setting | val MAE | test MAE | Source |
|---|---|---|---|---|
| CountingDINO local | dinov3_vits16, DEI+filter+ellipse | **39.696** | — | `/data/cdino_run/val.csv` (this root best_metric) |
| CountingDINO local | dinov3_vitl16, DEI only | 41.062 | — | results.csv; incomplete flags |
| CountingDINO local | dinov3_vitl16, full flags | *in flight* | — | N0030_h0028 / H0028, bar ≤38.696 |
| CountingDINO paper | DINOv2-L | 25.48 | 20.93 | arXiv 2504.16570 |
| TF-CAC (survey) | SAM training-free | — | — | survey only, not run |
| *external ref (other tree)* | N0015 trained | 19.343 | 19.066 | NOT parent; goal context only |

Goal context: training-free route should reach advanced level.
Inference-side train-fit scale on frozen N0015 already measures val **18.42** /
test **19.61** (see `calibration_trainfit.md`; formal card = H0029 if booked).

## Invariants for this root

1. **No supervised head training** on nodes under this root unless user re-opens training.
2. Allowed: CountingDINO config sweeps, other TF-CAC-class methods,
   inference-side calibration on frozen N0015 predictions (fit on train only),
   backbone/feature swaps, prompt/filter/multiscale switches.
3. Hardware: same RTX3060, same FSC147 splits/annotations.
4. Metric: FSC147 val MAE (no EMA — no train loop); report test too when available.
5. Evidence via `discovery evidence` with **new H-ids** (do not reuse supervised H0022–27).

## Open

- [ ] Collect N0030 full-flag vitl16 MAE (in flight) → update seed if better.
- [ ] Book/execute H0029 train-fit scale calibration as formal card (val≤18.50 AND test≤19.70).
- [ ] Optional: paper-faithful cosine/normalize A/B after full-flag lands.
- [ ] Optional: TF-CAC or other TF method port.

# N0030_h0028 — evolution of N0029_tf_champion (H0028 full-flag vitl16)

## Hypothesis Set

1. **H0028** — IF a training-free CountingDINO configuration sweep that adds the paper-faithful background filter and ellipse normalization flags (`filter_background=true`, `ellipse_normalization=true`, `ellipse_kernel_cleaning=true`) on top of divide-et-impera to the local `dinov3_vitl16` backbone while every other ROI-norm/minmax/resize flag stays byte-identical to the seed run, IN the N0029_tf_champion training-free tree on FSC147 val, THEN val MAE drops at least 1.0 below the DEI-only dinov3_vitl16 run and at least 1.0 below the seed best 39.696 (at or below **38.696**), BECAUSE the seed vits16=39.696 already runs with filter+ellipse+DEI all on while the first vitl16 attempt left those flags at default-off and scored 41.06, so the missing background suppression is the known gap between local vitl16 and the proven local config rather than a backbone-size effect. DISPROVED IF val MAE is above 38.696 after the flag-complete vitl16 run, or if the flag-complete run fails to beat the seed 39.696, or if filter/ellipse engagement is a no-op under vitl16 features.

→ conf 0.500 (fresh create) — one test event.

## Not a train_node card

This node is **training-free**: there is no `run_node.py` / no head training.
The single experiment is one CountingDINO val pass with the flag-complete config.

| Field | Value |
|---|---|
| script | `/data/cdino_run/convolutional_counting.py` |
| model_name | `dinov3_vitl16` |
| split | `val` (n=1286) |
| flags | `--divide_et_impera True --filter_background True --ellipse_normalization True --ellipse_kernel_cleaning True` |
| log | `/data/repro/logs/cdino_vitl16_full_val.log` |
| results.csv | `/data/cdino_run/results/results.csv` |
| seed baseline | vits16 full-flag **39.696** (N0029 best_metric) |
| DEI-only vitl16 (prior) | **41.062** (not the bar; configuration incomplete) |
| booked bar | **≤ 38.696** (39.696 − 1.0) |

## Bars (conjunctive)

- R1 val MAE ≤ **38.696**
- R2 val MAE ≤ DEI-only vitl16 (41.062) — implied by R1
- R3 not a no-op: flag-complete run must differ numerically from 41.062 (recompute from results.csv row)

Missing R1 ⇒ REFUTED. No futility_bar / no ep16 (no train loop).

## Procedure

1. Launch full-flag vitl16 (done this session, PID in flight).
2. When log prints `{'MAE': ...}` and results.csv gains the row: record val MAE.
3. `discovery evidence H0028 --type supports|contradicts --strength ... --node N0030_h0028`.
4. feedback/quant.md + synthesis note under this node; update N0029 best_metric if better.

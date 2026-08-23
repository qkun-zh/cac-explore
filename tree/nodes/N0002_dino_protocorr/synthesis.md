# synthesis.md — N0002_dino_protocorr

## Verdict
N0002 succeeded (10ep, 317s, 22.17M) with val MAE 42.05 (best) / RMSE 122.06, delta −4.64 vs S0001 baseline (46.69). H0001 contradicted: pure single-prototype cosine calibration at this capacity does not reach <30. Still trending down at ep10; time budget only 18% used. Quantitative signal is real but insufficient.

## Quality Gate (7 dims)
- mechanistic: pass — prototype cosine + tau is explicit & falsifiable
- scoped: pass — FSC147, frozen DINOv2-S reg4, 392 input, single box S-space
- predictive: pass — H0001 predicted <30, observed 42.05
- falsifiable: pass — DISPROVED IF ≥30, satisfied
- novel: pass — minimal 22M DINOv2 correlation floor under 32M (vs CountingDINO 300M training-free)
- transferable: pass — prototype pooling & decoder are category-agnostic
- actionable: pass — clear next steps (deeper head, scale context, longer training)

## Deduplicated Hypothesis Updates
- H0001 (IF prototype cosine IN FSC147 THEN MAE <30 BECAUSE DINOv2 semantics): **contradicts 0.75** — 42.05 @10ep, monotonic descent but far from 30. Confidence 0.50→0.42.
- H0002 (learnable τ ≥5% gain): neutral / untested — no ablation, softplus τ present but isolated effect unknown.
- H0003 (aux count head helps): neutral / untested — not implemented.
- H0011 (scale/magnitude embeddings ≥5% gain): neutral / untested — not implemented, but causal analysis predicts it matters (RMSE 2.9×MAE).

Cross-node hints: N0002's RMSE tail & scale-blindness support N0003 cross-attn and N0004 multi-scale branches.

## Booking List (→ memory/hypotheses.jsonl)
- create H0001, H0002, H0003, H0011 (if absent) with idea.md texts
- evidence H0001 contradicts 0.75 from N0002

## Tested Hypotheses
[H0001] (H0002/H0003/H0011 proposed but not exercised)

## Scores (for select_next)
best_metric 42.049, train 317.7s, quality ~0.38 (λ_acc 0.85), avail ~0.82, score ~0.56; status → synthesized

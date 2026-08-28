# Synthesis — N0054_xscale_exemplar (LOCKED champion, 19.647)

## Verdict
GCA + single fused coarse XScale exemplar prototype is the frozen-backbone optimum. **LOCKED as deliverable.**

## Evidence chain (30ep @384, frozen)
- N0051 GCA-only 20.599 (GCA genuine ~1.6)
- N0054 GCA+XScale 19.647 (XScale beats GCA-only by ~0.95)
- N0053 GCA+RGA 21.450 (density-bias aux hurts)
- N0055 GCA+XScale+XKey 20.835 (+1.19 vs parent: 2K condenser keys dilute cross-attn)
- N0056 GCA+XScale+XFine 24.313 ES@17 (+3.06 at E17: extra fine-scale exemplar entropy)

Adding MORE exemplar info (separate keys OR extra scale) degrades; the single fused coarse prototype is
already optimal. This mirrors the density-side verdict (DDCA/RGA/SALF/FILM/cross-attn/MoE all degrade).

## Booked hypotheses (calibration, eta=0.20)
- H0076 (XScale-Key 2K keys): refuted by N0055 20.835.
- H0077 (XFine fused fine summary): refuted by N0056 24.313 ES. Ledger conf is advisory; STATE operational list governs.
- STATE refuted set now: DDCA, RGA, SALF, FILM, cross-attn, MoE, bg-token, XScale-Key, XFine.
- Confirmed keeps: GCA (global-count aux), XScale (fused coarse exemplar summary).

## Deliverable
N0054 19.647 / RMSE 74.05 · 31.32M ≤ 32M · verify best.pth intact (125MB, 3 keys, OK on server).

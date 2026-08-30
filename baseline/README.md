# Baseline — Distributed Count-Adaptive Counter (Canonical Champion)

Official consolidated baseline incorporating **every empirically-proven-effective** design
decision from the CAC explore project. All validated through controlled single-switch ablations
(see `tree/nodes/N0054_xscale_exemplar/` lineage). **MAE 19.647 / RMSE 74.05 / 31.32M params** on FSC147 val.

## Effective components INCLUDED (each individually proven positive)
| Component | Where proven | Gain vs prior |
|---|---|---|
| **Frozen intermediate backbone readout hs(2,3)** | N0054 vs N0068 (2×2 table) | +~7 pts over final-layer readout |
| **GCA** (global-count auxiliary) | N0051 (GCA-only 20.60) | ~1.6 over no-GCA |
| **XScale** (multi-scale coarse exemplar summary, fused into single prototype) | N0054 (+0.95 over N0051) | sole positive exemplar-enrichment |
| **Cross-attention condenser** (learned MHA+FFN consumer) | N0057/58/59 all negative swaps | load-bearing attention |
| Proven hyperparams: AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep, 384px | N0054 reproducible | — |

## Proven-HARMFUL components EXCLUDED (each individually proven negative)
| Component | Node | Degradation |
|---|---|---|
| DDCA (dilated context branch) | N0052/53 | +1.8 |
| RGA (regional count aux) | N0053 | +0.85 |
| Extra spatial summaries (key, fine, max) | N0055/56/60 | +1.2..+3.2 |
| Aggregation operator swaps (matcher/poly) | N0057/58/59 | +1.3..+3.5 |
| Count-normalized readout | N0061 | +3.9 |
| Fine-decoder injection (hs[1]) | N0062 | +1.7 |
| Bigger backbone (small) | N0063 | +6.25 |
| Scale-embed | N0065 | +0.78 |
| **Any backbone unfreeze (FT)** | N0066/67 | +6..+9 (intermediate readout load-bearing; unfreeze net-harmful) |

## Config (frozen contract)
`config.py` → `build_model(cfg)`; forward `({"density", "n_aux"})`; MSE(+count) loss.
Regime: backbone frozen, head-only train. Engine: `code/engine/train.py`.
Params assert ≤ 32M.

## How to extend
Attach a new component via a **single pluggable switch** (`use_<name>`) on the N0054 interface
(shared frozen features hs(2,3) + exemplar embedding). Ablate on/off against this baseline;
early-stop gate = same-epoch ≥ +1.5 worse at ep16+.

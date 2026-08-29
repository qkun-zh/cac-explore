# Synthesis — N0058_pompart_exemplar (early-stopped E15, NEGATIVE)

## Verdict
**NEGATIVE — H0080 KILLED.** PMOM (2×2 part moments + gate, trainable 3.50→2.04M) best
**23.151 @E13** / E15 23.805 = **+2.17 same-epoch** vs N0054 21.635; **converged-gap floor +3.50**
(23.151 vs 19.647). Pre-registered KILL (≥20.4) fired **E7** and never left the lethal zone; the
ep16+ early-stop bar (+1.5 ⇒ ≥21.147) is met in effect (E15 same-epoch 23.805). **Early-stop correct.**
Per-part mean+mean² over a 2×2 pool destroys within-part layout and cross-part covariance that
self-attention on 49 ROI tokens supplied; the head is under-converged, not unstable (no NaN, loss
monotone down to 3.97). Use_pmom=False restores exact N0054 (single-switch smoke-proven).

## Evidence-chain additions (frozen-regime table)
N0058 joins N0055 / N0056 / N0057 — all NEGATIVE:
| Node | Swap axis | Delta vs N0054 (19.647) |
|---|---|---|
| N0055 XScale-Key | info-add (2K condenser keys) | +1.19 |
| N0056 XFine | info-add (extra fine scale) | +3.06 |
| N0057 cond-matcher | consumer (condenser MHA) swap | +1.43 |
| **N0058 PMOM** | **producer (exemplar agg) swap** | **+1.52 best / +2.17 same-epoch / +3.50 floor** |

Now **BOTH** producer (exemplar aggregation) and consumer (condenser) operator-swaps degrade, as do
info-adds. The **ONE staying-positive axis is XScale** — granularity enrichment that is additive and
single-prototype (coarse global summary, no operator change). 
**Refined lesson:** the benefit the frozen head extracts comes from ADDING one coarse, single-slot,
pre-attention summary — not from widening the attention interface (keys/extra-scale collapse the
prototype), and not from swapping the learned attention (self-attn on the exemplar pathway, producer,
and condenser cross-attn, consumer, are both load-bearing).

## Booked hypotheses (quality-gated via check_hypothesis)
- **H0080** (producer aggregation-operator swap → PMOM): **refuted** by N0058. cf. ledger refutes event.
- **H0081** (load-bearing attention law + boundary): book as new. Law: producer OR consumer learned
  attention swapped to a lower-capacity non-attention operator ⇒ val MAE degrades ≥ +1.4. BOUNDARY:
  applies to aggregation swaps, NOT XScale-type granularity/scale enrichment (the one positive axis)
  nor interface-cardinality/density-side changes. QUANT CAVEAT encoded: capacity and expressivity are
  CONFOUNDED (no param-matched control: PMOM 2.04M vs N0054 3.50M, −42%); falsifier requires a
  ≥1.5M-trainable param-matched swap beating 19.647. Chose the single most defensible new hypothesis
  (law+boundary+confound in one) to respect K_synth=2 alongside the H0080 create.

## Calibration bin table (calibration_report.py, verbatim, eta=0.20)
```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N  correct    rate
[0.25,0.50)      1        0      0%
[0.50,0.75)     13        4     31%
[0.75,1.00)      0        0       -
<0.25            0        0       -
overall         14        4     29%
```
Refutations here (H0080) are logged with `refutes` (unscored by the report), consistent with the
H0076/H0077/H0079 sibling convention; ledger conf is advisory, STATE operational refuted list governs.

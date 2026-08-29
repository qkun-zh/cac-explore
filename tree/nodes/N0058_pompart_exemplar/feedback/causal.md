# Causal feedback — N0058_pompart_exemplar (early-stopped E15)

## Intervention & outcome — a single-switch, gate-consistent kill
- Swap: exemplar aggregator {proj 98.5k + 2×TransformerEncoder 1.58M + attn-pool 257} → {2×2 part-pool → m=[h;h²] → softmax part-gate 12.3k → moment_proj 196.9k}. Condenser MHA, GCA, XScale, shape_mlp untouched; (B,K,256) fused-prototype interface shape-identical. Trainable 3.50M→2.04M (−1.46M); use_pmom=False restores N0054 exactly (smoke-proven single switch).
- Outcome: best val 23.151 @E13; E15 23.805 = **+2.17 same-epoch** vs N0054; converged-best gap **+3.50** vs 19.647. H0080 KILL (best≥20.4) met at E13 ⇒ early-stop is an honest pre-registered kill, not premature.
- Caveat: curve still descending at kill (train loss 4.02@E14) and N0054's gain came E18-27; measured gaps are FLOORS, direction reliably negative.

## Candidate causes (only the swap moved)
1. **(a) Head trainable capacity — REFUTED as a head-wide law, but inseparable within the aggregator.** Capacity-adds degraded everywhere: N0055 (2K keys, +1.19), N0056 (XFine, +3.06), density add-ons (DDCA/RGA/SALF/FILM/MoE). "More trainable ⇒ better" is false. But "capacity LOCATED in the aggregator" is confounded with operator bias: 1.46M and the mechanism moved together, no param-matched control.
2. **(b) Operator expressivity — NOT refuted; strongest live cause.** 7×7 ROI → 2×2 avg-pool → 4 part vectors; m=[h;h²] is per-part mean + 2nd raw moment on a fixed grid. Provably destroyed: within-part layout (49→4 spatial slots), cross-part interactions/covariance (linear gated sum, no mixing nonlinearity before one readout). Self-attention's 2 transformer layers + attention-pool express arbitrary data-dependent token mixtures with iterated nonlinearity — exactly that subspace.
3. **(c) Gate softmax bottleneck — SUB-CASE of (b), not independent.** α=softmax forces H = convex combination of 4 fixed moment vectors (4 scalars in 768-d). N0054 also pool-softmaxes (convex) but over 49 data-dependent weights AFTER nonlinear token mixing; PMOM has neither the 49 slots nor the mixing. Not separable from (b) in one run.
4. **(d) Exemplar interface — structurally REFUTED, distributionally OPEN.** Shape/wiring provably intact (single-switch smoke; condenser/GCA/XScale untouched). But PMOM's e is a fixed-basis summary: distinct exemplars colliding on pooled moments collapse condenser-key/GCA e_mean latent rank. That quality (measurable) is unfalsified — a downstream symptom of (b), not a break.

## Attributability (single-run honesty)
- Cleanly refuted: head-wide capacity-causality (a); structural interface corruption (d).
- Confounded, NOT disambiguated: capacity-vs-expressivity and gate-slot-count — all moved in ONE switch, N=1, early-stopped at a floor. Decisive counterfactual (a swap keeping ≥1.5M trainable, or learned attention under a different bias) was not run.

## N0057 precedent & candidate law
- N0057 (consumer-swap): condenser MHA+FFN→cosine matcher, ↓0.62M → +1.43, stalled E15=21.08, missed N0054's E18-27 drop. N0058 (producer-swap): exemplar self-attn→moments, ↓1.46M → +2.17@E15 / +3.50. Same direction; removed attention-capacity ratio (2.4×) ≈ gap ratio (2.4×) — suggestive, N=2.
- Law (candidate, 2 swaps + refuted add-on set): IN [frozen 30ep head-only] IF [learned attention replaced by a lower-capacity non-attention operator at the producer (exemplar ROI) OR consumer (condenser) point] THEN [val MAE ≥ +1.4] BECAUSE [data-dependent token mixing on the exemplar pathway is load-bearing, its benefit accruing in the late-training window (E18-27)].
- Boundary: operator-swap axis ONLY — NOT granularity/scale enrichment (XScale remains the one positive axis, +0.95), NOT interface cardinality (2K keys), NOT density-side modules; bounded to 30ep cosine + frozen backbone; magnitude is an early-stop floor.

## Booked causal hypothesis
H: IF [exemplar aggregation operator swapped away from learned self-attention] IN [frozen 30ep regime] THEN [val MAE ≥ +1.4] BECAUSE [attention on 49 ROI tokens is the load-bearing capacity]. DISPROVED IF [an operator swap that KEEPS ≥1.5M trainable or keeps a learned attention under a different inductive bias beats 19.647].
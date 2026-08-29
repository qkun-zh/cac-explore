# Feedback — N0059_pom_morph (Qual)

**Lead-booked deviation**: subagent unavailability (2 aborts) — Lead wrote this file directly, logged in journal.

## 1. Bell-curve-over-time pathology
The curve peaks at E18 (20.958) then oscillates 21–23 for the remaining 12 epochs while training loss keeps dropping (E18 3.37 → E30 2.13). Loss→scale keeps improving but val MAE stalls/regresses. This is the same "overfit-to-count-scale, not exemplar-drive" signature N0058 showed — the head learns a global scale proxy and stops benefiting from the exemplar aggregate. A healthy aggregation operator shows monotone-ish late improvement (N0054: 21.1@E19 → 19.65@E29); N0059's stall at ~21.4 is the tells.

## 2. Expressivity expectation of PoM as an exemplar aggregator
The block only retains the **token-averaged 2nd-order moments** H = mean_n(α₁⊙h + α₂⊙h²) and broadcasts that single shared state back over all 49 tokens through one shared sigmoid gate σ(W_s X). Two structural weaknesses for a 49-token ROI:
- **Cross-token order is collapsed to a mean**: self-attention's softmax normalizes across tokens per query so each token's context is *query-specific* (every token gets a different weight vector); PoM's H is a *global* statistic re-applied identically (modulo the gate) to all tokens. So per-token context specialization — the thing that lets attention pick "the distinguishing exemplar patch" — is unavailable.
- **Shared α/gate**: α and the gate are shared parameters; there is no per-head/per-query weighting, so the block cannot re-rank which exemplar tokens matter under IDF-style contrast. It lowers to a pooled-feature + gated-add, close to a Global-Average-Pool + MLP, which the data shows is insufficient.

## 3. Qualitative residual confounds (any that could void the attribution)
- **D≫n inversion**: D=352 moment channels vs n=49 averaged tokens → the 49-token mean populates a 352-dim state that is necessarily rank-≤49 — heavy redundancy; much of the 3.52M capacity is in a fat final projection that a rank-limited input cannot exploit.
- **Small α init ±0.001**: the 2nd-order term h² is damped near zero early (α tiny), so the polynomial advantage is initially inert; N0054's attention is immediately active. This is a *warm-up* handicap, documented in idea.md as GELU-deviation; not a design bug, but it explains the slow early descent (E1 37.1 vs N0054 ~28).
- **Gate sigmoid saturation**: σ(W_s X) in [0,1] can saturate, further flattening per-token contrast.

None of these void the operator-class attribution — they are *reasons the operator under-performs*, not confounds: capacity (matched), token count (49), residual path (matched), condenser/consumer (untouched), plugin isolation (single switch) are all clean. The attribution "non-attention polynomial-moment aggregator is materially under-expressive here" is qualitatively sound.

## 4. Conclusion
Operator-class attribution is clean. The PoM-PolyMorpher, though capacity- and residuals-matched and faithful in structure, lacks the per-query token-contrast that self-attention's shared-softmax provides — and the D≫n rank ceiling caps what its matched 3.52M can do. This reinforces that producer attention is load-bearing *because of per-token contrast*, not because of attention's parameter form.

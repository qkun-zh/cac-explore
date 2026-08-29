# Feedback — N0059_pom_morph (Diagnostic)

**Lead-booked deviation**: subagent unavailability (2 aborts) — Lead wrote this file directly, logged in journal.

## Root cause
Clean run (no OOM, no crash, status success, 30/30 ep, instability flag false). The KILL is a *design* failure, not an operational one. N0059 removed BOTH confounds that made N0058 non-attributable (capacity −42%→matched 3.522M, 2×2 part-pool→49 full tokens) and added residual-path matching, yet still degraded (floor +1.31, early-stop bar fired E16 +2.13). Attribution: the **non-attention operator class** is the differentiating variable.

## Implementation faithfulness check (model.py:83-93)
```
xn  = norm1(x)
h   = gelu(W_h xn)                                   # 49 tokens, D=352
H   = (α₁⊙h + α₂⊙h²).mean(dim=1)                     # (B*K,D) token-averaged 2nd-order moment
gate_v = sigmoid(W_s xn)                             # per-token gate
po  = gate_v * H.unsqueeze(1)                        # broadcast shared moment
po_parts = W_o(po)
x = x + po_parts                                     # residual
x = x + ff(norm2(x))                                 # residual
```
This faithfully realizes the paper Eq.3 structure (token-averaged moment H + per-token gate + W_o mixing + norm_first residual). The deviations from paper are minor and documented in idea.md (GELU vs clamp(LeakyReLU), α init ±0.001). **No implementation bug. The operator-class attribution stands.**

## Is this a NEW failure mode?
No. It is the same failure family as N0058/N0057 (aggregation-operator swap degrades), now with the capacity/pooling confounds cleared. No append to memory/failure_modes.md — no new operational pitfall (server-side training hygiene already covered; the one operational note is the run_node.sh git-pull hang, logged in STATE gotchas).

## Key insight from D≫n
The PolyMorpher's moment state is D=352 while it averages only 49 tokens → rank-≤49 moment state feeds a wide W_o; matched capacity is partly neutralized by this rank ceiling. This is the *mechanism* of its under-expressivity, distinct from attention's per-query softmax contrast.

## Recommendation for lineage
The aggregation-operator axis (producer AND consumer) is now empirically closed in this regime: attention swaps always degrade. Do NOT retry another non-attention producer. The H0081 boundary was "aggregation swap only, excluding XScale-type enrichment" — that boundary holds. The one remaining un-run control is a **param-matched alternative-ATTENTION** operator (H0081's "non-marginal" arm, e.g. cross-token cosine/linear attention at matched 3.5M) — but given three consecutive attention-swap failures (N0057 consumer, N0058 producer part-pool, N0059 producer matched), the marginal value is low; the load-bearing law is far better supported than a bespoke param-matched-attention ablate. Prioritize XScale-style single-slot coarse additive enrichment (the ONLY positive axis) or a regime/extent change, not further aggregation morphs.

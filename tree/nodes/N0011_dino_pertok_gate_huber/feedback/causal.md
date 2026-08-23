# Causal Feedback — N0011_dino_pertok_gate_huber

## reasoning
Result: best MAE 26.68 @E34 (+24% vs parent 21.53@E26); RMSE/MAE 3.52 vs parent-best 3.53. Three levers changed vs N0010 at once: scalar→per-token gate, MSE→Huber(δ=5), count-w 1.0→0.3. idea.md's claim that reverting count-w to 0.3 "isolates the Huber effect" is inverted — relative to parent it changes BOTH loss shape AND loss mix; nothing is isolated.

Trajectory: N0011 is worse than parent at every matched epoch (E13 28.33 vs 25.50; E21 28.89 vs 23.70; E26 27.00 vs 21.53); its best (26.68) never even beats parent's E13 checkpoint. Val MAE swings ±2.6 epoch-to-epoch while train loss falls monotonically (4.99→3.48) — a memorization/instability signature, not an optimization failure.

Gate mechanism: only +49k params (23.16M vs 23.11M), so raw capacity is NOT the story; the harm is input-dependent routing. The gate reads projected [z6;z11] and co-adapts with t6/t11 proj layers, becoming a sample-keyed router over FROZEN features — it can fit training-set patch idiosyncrasies instead of transferable depth preference, and per-token softmax re-mixing injects forward-pass variance into what was a fixed feature space (explains val oscillation). Parent's 2-param scalar gate acted as a regularizer locking a global compromise; N0011 removed that constraint.

Huber × count-w interaction: Huber binds PER PIXEL (δ=5 on density values), but catastrophic outliers are mass spread over many moderately-wrong pixels, so capping rarely engages where it matters; where it does engage (density blowups) it slows correction of exactly the errors RMSE punishes. Simultaneously, count-w 1.0→0.3 cuts 3.3× the supervision on the only term aligned with the eval metric (summed-count MAE). Net: objective drifted away from the metric while tail gradients were capped, not rebalanced.

Tail verdict: RMSE/MAE essentially unchanged (3.52 vs parent-best 3.53) → pixel-loss shape does NOT control the outlier tail; the tail lives at count level (systematic mass miscalibration on a few high-count samples).

## actionable_feedback
1. Single-lever rule for children of N0010: change ONE thing vs champion config; count-w stays 1.0 unless it is the lever under test.
2. Do not retry per-token gating as-is. If spatial adaptivity is retried: hidden ≤8, higher temp/near-uniform init, dropout on gate logits — and only as the sole edit.
3. Drop pixel-Huber for tails (engine keeps huber support, unused). Tail work must act on summed counts: log/sqrt-count transform, count-stratified sampling, or per-sample weighting that PRESERVES outlier gradient direction instead of capping it.
4. Confound ledger for synthesis: H0019/H0020 refutations are JOINT, not individual; a clean H0020 kill needs a Huber-only arm (scalar gate, w=1.0) — low priority since the ratio moved zero.

## hypothesis_updates
- **H0019** | contradicts | strength 0.85 | MAE +24% and worse at ALL matched epochs; disproof criterion ("MAE increase") met decisively. Mechanism refinement: content-dependent routing over frozen tokens overfits/destabilizes rather than merely failing to adapt.
- **H0020** | contradicts | strength 0.65 | ratio 3.52 breaches the >3.5 disproof bar, but confounded (count-w co-change) and mechanism mismatch (per-pixel δ=5 ≠ count-level robustness). Refutation scope: "pixel-space Huber doesn't fix CAC tails", NOT "robust losses can't".
- **H0022 implication** (note only, no evidence event): strengthens its motivation — the tail was untouched by BOTH pixel-Huber and gate change, leaving count-level reweighting as the live lever. It MUST hold count-w=1.0 and ship without any added adaptive module, or it repeats this node's confound; its existing disproof bar (ratio <3.2 without MAE regression) remains appropriate.

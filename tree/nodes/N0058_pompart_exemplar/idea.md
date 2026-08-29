# Idea — N0058_pompart_exemplar (parent: N0054_xscale_exemplar, frozen)

The untested structural axis after N0055–N0057: every N0054-child so far either ADDED exemplar info
(keys, extra scale) or REPLACED the load-bearing condenser — all degraded (+1.19 / +3.06 / +1.43).
N0058 changes ONLY the per-exemplar AGGREGATION OPERATOR (2-layer self-attention + attention-pool over
49 ROI patch tokens), swapping it for a part-wise 2nd-order polynomial-moment summary (PMOM). An
operator swap on the exemplar side, not another add-on, not a condenser touch.

## Change (structural REPLACEMENT, FROZEN, optimizer unchanged)
**PMOM (H0080, ~209k new vs ~1.68M removed).** Single config switch `use_pmom=True` inside
ExemplarEncoder (N0054 model.py lines 64-95). Everything else identical to N0054: frozen backbone,
Condenser (per-pixel MHA, load-bearing), GCA aux, XScale coarse-summary add, recipe AdamW 1e-3/wd0.05/
cosine/bs16/AMP/30ep @384/MSE+SmoothL1/augment=True. From ROI (B*K,384,7,7) = roi_align(h3):
1. part_pool: F.adaptive_avg_pool2d(roi,2) → (B*K,384,2,2) → h_p (B*K,4,384), p∈{1..4} (2×2 parts).
2. moments (order k=2): m_p = cat([h_p, h_p*h_p], -1) → (B*K,4,768).
3. contextual part-gate: α = softmax over 4 parts of Φ_g(m_p), Φ_g = Linear(768,16)+GELU+Linear(16,1);
   α (B*K,4,1).
4. PoM map: H = Σ_p(α_p ⊙ m_p) → (B*K,768).
5. fused prototype: e = W_o·H + b_o (Linear 768→256).
6. interface preserved EXACTLY: e→(B,K,256) + shape_mlp(wh) + xproj(coarse) XScale summary (both as in
   N0054, now on the fused prototype). Params: Φ_g ≈12.3k + W_o ≈196.9k ≈209k. REMOVED proj(98.5k) +
   TransformerEncoder(1.58M) + attn(257). Net head ≈ −1.5M ⇒ total ≈29.9M ≤32M ✓. Removed proj/tr/attn
   is nullified (dead params dropped); no other module changes.

## Why (grounding)
- N0055 (2K keys) and N0056 (extra scale) degraded: the condenser wants ONE fused prototype, not more
  evidence. N0057 proved the condenser cross-attention (consumer) is load-bearing. The remaining untested
  lever is the exemplar aggregation operator (producer), the allowed §5.14 interface.
- PoM (arXiv:2604.06129, CVPR-F 2026) replaces self-attention with a degree-k polynomial aggregation
  into a compact state + gating — linear-complexity, attention-matching quality. Applied here BELOW the
  condenser (exemplar side), NOT to the dense cross-attn that N0057 proved load-bearing.
- ParTY (arXiv:2603.09611, CVPR 2026) motivates part-wise decomposition with adaptive holistic-part
  fusion; the 2×2 part split with softmax part-gating over moment vectors is that adaptive fusion.
- Self-attention over 49 patch tokens of K=3 tiny exemplars mixes redundant intra-appearance tokens and
  overfits; 2nd-order part-wise moments (mean+square) are compact, permutation-oblivious, ~8× cheaper,
  with the same single-prototype interface the condenser already consumes.

## Hypothesis
**H0080** IF [per-exemplar aggregation operator replaced by part-wise 2nd-order polynomial-moment
summary (PMOM)] IN [N0054 base] THEN [val MAE < 19.647] BECAUSE [self-attention + attention-pool over
49 ROI patch tokens wastes capacity on redundant intra-appearance mixing and overfits K=3 tiny
exemplars, while 2×2 part-wise mean+square polynomial moments + part-gating capture category-discriminative
part statistics and inject them through the unchanged single fused prototype into the load-bearing
condenser cross-attention].

## Gates
- R1 smoke: stub backbone, use_gca/use_xscale/use_pmom on; params ≤32M; density (B,1,96,96); loss finite
  & drops. Verify use_pmom=False restores the N0054 attn path (single-switch).
- R2 30ep @384 frozen recipe vs N0054 (19.647). CONFIRM if best val MAE <19.647 (strict beat); WEAK-KEEP
  if best ≤20.0; KILL if ≥20.4 OR early-stop bar breached (ep16+, same-epoch train ≥ +1.5 vs parent best
  19.647).
- R3 pluggability: toggling use_pmom touches only ExemplarEncoder aggregation; no coupling to Condenser/
  GCA/XScale internals. Kill-clear falsifier: if PMOM degrades to N0057-magnitude, the 2-layer exemplar
  self-attention is itself load-bearing (like condenser MHA in N0057) and the operator-swap hypothesis is
  refuted.
# Causal feedback — N0062_fine_decoder (native-1/4 decoder injection, H0090)

## Verdict
**H0090 DISPROVED.** best 21.323 @E17 / final 22.298 / RMSE 75.96 (+1.676 floor, +1.91 RMSE vs N0054 19.647/74.05). FAIL >20.0 fired. Single-switch clean; attribution to h1 injector (1x1 96→8+GN, in_ch 192→200) sound.

## Why it hurt instead of helping the tail — three stacked causes
1. **Wrong level of abstraction.** hs[1] is stage-1 stem output (96ch, 1/4) — Gabor-like texture, not the dataset-specific count abstraction the decoder needs. The champion's `fine` is post-FineFuser: h2(192@1/8)+h3(384@1/16) fused+refined then upsampled — a learned, count-tuned 1/4 signal. Adding raw stem features forces the decoder to jointly denoise low-level texture while predicting density; under frozen-backbone/plain-MSE, the extra 8 channels act as noise, not signal. Train loss still fell to 2.29 (vs champion 2.37), so it fit train — but val plateau 21.3 for 13 epochs = overfit-to-texture.

2. **No identity-at-init; basin shift.** Unlike GCA (zero-init, additive, 0.02-attenuated) or CountNorm's W2-zero f=1, the decoder first conv widening is Kaiming-initialized on the new 8 columns — the champion optimum is destroyed at step 0. Early trace confirms: E03 +5.76 spike, then recovery to tie at E17, then late divergence as the head settles in a worse basin. Champion's late descent 21.25→19.65 (E17-29) relies on the exact 192→256 first-conv weights; perturbation prevents that descent.

3. **Tail mechanism mis-specified.** H0090's BECAUSE claims "tail fails by cell-quantization mass loss from reading only upsampled-from-1/8 fine; native 1/4 per-cell fidelity fixes it" — but champion's `fine` IS already at 1/4 (interpolated) and its cell is the density cell itself (96×96). The quantization the tail suffers is not about the decoder's input resolution but about GT density kernel scale vs object density (crowded images have overlapping Gaussians). More input res cannot disentangle overlapping kernels without a count-aware target or adaptive kernel — hence both MAE and RMSE worsened. The 21.323 best coincided with RMSE 75.96 (not better than 74.05), showing no tail-specific win.

## Relation to closed axes
Same outcome class as other frozen-head density-side input enrichments, but NEW mechanism: N0053 RGA (spatial output bias, +1.74), N0056 XFine (exemplar agg, +3.06), N0061 count-multiplier (+3.86) all hurt on density-side adds — but N0062 is the FIRST pure decoder-input-resolution probe and is directly comparable to none of them structurally (no spatial summary, no ROI, no count scale). Its milder +1.68 (vs +3.x) suggests decoder input is the least harmful density-side axis, but still negative. Distinct from N0058/59 (producer swaps, aggregation) and N0060 (MAX summary) — all exemplar-side.

## Direction closure
Decoder-receiver-resolution via raw frozen h1 injection (as 8ch 1x1+GN widening) is CLOSED. The frozen head's 1/4 representation is already optimal via FineFuser; adding earlier backbone levels without a zero-init, gated, or resolution-equivariant design is harmful. Future resolution probes must be (a) zero-init-preserving and (b) semantic (e.g., learned refinement of `fine` itself), not raw stem concatenation.

# Feedback — N0062_fine_decoder (Qual)

Verdict: FAIL. Root cause is **mechanism insufficiency (b) + feature-level mismatch (c)** — not an implementation bug, not an ops pitfall; the promised tail-resolution benefit never converted to val.

1. **(c) h1 is the wrong resolution signal.** hs[1] (96ch @1/4) is the stem-stage feature: low-level texture/edge, not semantic. The FineFuser's `fine` (128ch @1/4) already IS a 1/4 map — it is fused from h2/h3 then bilinear-upsampled, but the upsampling is learned (fuse+refine) and trained for 30ep to place count mass. Injecting raw h1 adds high-frequency, count-irrelevant channels; the decoder's first conv (192→256, now 200→256) must learn to ignore or use them under plain MSE. It overfits instead.

2. **(b) decoder input widening dilutes the learned basin.** First-conv delta only +18.4k params but changes the input distribution: 8 extra channels initialized Kaiming (not zero) perturb the champion's optimum at step 0 (no identity-at-init). Early epochs competitive (E01 -2.0, E02 -1.6 vs champion) then E03 spike +5.76 — the injector destabilized the first week's optimization. Even when it tied at E17 (21.32 vs 21.25), the head could not descend further while champion dropped another 1.6.

3. **Tail not visibly helped.** Best RMSE 75.96 vs 74.05 (+1.91): the dense-tail RMSE benefit H0090 posited did not appear; E17's best coincides with a generally lower RMSE across all images, not tail-specific. The 17-image N≥500 tail (75.86% SSE) is count-scaled MSE (error²∝c²); native 1/4 fidelity alone cannot fix it without target-side count weighting — same PREMISE LIMIT as N0061, though less severe.

4. **New axis, same outcome.** Distinct from RGA (N0053 spatial output bias), N0056 (exemplar agg entropy), N0059/60 (ROI summaries): this touched decoder INPUT only (receiver side), no exemplar/condenser/GCA change (verified h1 not used there). So it is a NEW structural probe, but the result (+1.68) joins the same failure class: frozen-head density-side input enrichments hurt under plain MSE.

5. **Closure.** Receiver-resolution via raw h1 is CLOSED as a +19k low-level injection. The decoder's current `fine` (fused, trained) is already the optimal 1/4 representation this head can use; naively adding stem features is harmful. If resolution matters, it must be via a learned, zero-init, count-aware path — not a free 8ch widening.

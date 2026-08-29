# Diagnostic — N0062_fine_decoder (FAIL, 30/30, best 21.323@E17 / final 22.298)

## Root cause — MECHANISM insufficiency, not implementation
Clean run (status=success, 30/30, oom=false, params 31.34M). FAIL gate (>20.0) fired (+1.676 vs 19.647; RMSE 75.96 vs 74.05). Train loss fell 153.58→2.29 (monotonic) while val best at E17 then plateau 21.3→22.3 for 13 epochs — overfit-to-input, not tail repair. No instability OOM.

## Implementation faithfulness — FAITHFUL (model.py:32-48, 125-148, 175-192)
```
Backbone: hs_map=(2,3) + hs_fine_idx=1 append hs[1] (96@1/4) when use_fine_decoder
CountingHead: extra=8 if use_fine else 0; decoder in_ch 192->200; injector=Conv2d(96,8,1)+GN(2,8); cat([fine,cond,injector(h1)])
Counter: feats=[h2,h3,h1]; h1=feats[2] if len>2 else None; head(h2,h3,bboxes,h1=h1)
```
Smoke verified: use_fine=False bit-identical (forward diff 0.0, param delta 0, state_dict keys equal, max weight diff 0.0); use_fine=True total 31.34M (+19,224), density (2,1,96,96) finite, n_aux retained. No coupling to exemplar/condenser/GCA (§5.14): h1 touches decoder INPUT only. Ordering fine injection post-FineFuser is sound; injector GN 2 groups correct for 8ch.

## Why the mechanism failed — three levels
1. **Feature-level mismatch:** hs[1] is stem-stage (纹理级), not count-semantic; champion's `fine` is already count-tuned. Adding raw texture adds noise the decoder must suppress.
2. **No identity preservation:** new 8 columns Kaiming-initialized perturb the champion basin at step 0 → E03 +5.76 spike, recovery to tie E17, then failure to descend the final 1.6 MAE the champion achieves (21.25→19.65). Unlike GCA-zero-init, this is a basin shift.
3. **Target-level cause remains:** dense-tail error (75.86% SSE in 17 imgs) is cell-quantization of overlapping Gaussians, not input-res-starved. Plain MSE on raw count-scaled blobs (train.py:344-345) cannot be fixed by input widening alone; target-side count weighting (dead `tail_reweight` 334-339) is the untested lever, not input res.

## Classify — NEW design-premise negative, no failure_modes.md append
First probe of decoder-receiver-resolution axis; not a re-run of N0056 (exemplar agg), N0053 (output bias), or N0060/55 (ROI summaries); not an implementation bug, not an ops pitfall. Design-premise negative ⇒ conservative NO append (cf. N0061 diagnostic). The mild +1.68 (least harmful density-side add yet) still decisively fails.

## Recommendation
Decoder input widening with raw frozen h1 is CLOSED. Do NOT retry naive stem concatenation. If resolution is to be probed again, it must be (a) zero-init basin-preserving (e.g., 1x1 zero + residual) and (b) learned refinement of the existing `fine` feature, not raw early-stage concatenation. The genuinely untested frozen lever remains count-as-SUPERVISION (`tail_reweight`), which needs a (currently forbidden) engine change; short of that, frozen-head LOS is hardening (exemplar + count + resolution all mapped negative).

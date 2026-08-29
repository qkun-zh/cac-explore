# Idea — N0060_xscale_max (parent: N0054_xscale_exemplar, LOCKED MAE 19.647)

**Executable test of H0083 / books H0084.** The ONE positive frozen-regime axis is XScale-style
coarse single-slot additive granularity enrichment. N0056 proved fine-scale entropy hurts (+3.06),
N0055 proved growing prototype cardinality hurts (+1.19), and the whole aggregation/operator-swap axis
is closed (N0057 consumer +1.43, N0058 producer part-pool +3.50floor, N0059 producer matched +1.31floor).
H0083 asks whether extending the positive axis with a SECOND coarse summary breaks 19.647.

## Change (ADDITIVE, FROZEN, optimizer unchanged)
Single switch `use_xscale_max` in ExemplarEncoder (N0054 model.py:90-94). Keep the existing XScale mean
summary untouched, add a SECOND coarse **MAX-order-statistic** summary onto the SAME fused prototype.
Producer self-attn (`self.tr`), condenser cross-attn, GCA, decoder, recipe ALL untouched. use_hxscale=False
restores the exact champion.

Forward after the existing XScale mean-add (model.py:90-94):
```
coarse2 = F.adaptive_max_pool2d(roi, (1,1)).squeeze(-1).squeeze(-1)   # (B*K,384) global max over 7×7 ROI
out = out + self.xproj_max(coarse2).view(B,K,-1)                      # additive to fused prototype
```
where `roi = roi_align(feat, rois, output_size=(7,7))` is the SAME aligned ROI already computed at
model.py:82 (zero extra FLOPs), and `xproj_max: Linear(384→256)`.

**Why MAX (order statistic), not a 2nd mean**: the existing XScale is a per-channel ROI-MEAN
(3×3-grid averaged to 1 scalar). A global mean, GAP'd over 7×7, is the SAME estimand with better
sampling (red-team: corr≈0.85–0.97) — redundant, WEAK-KEEP-at-best. The **max** is near-orthogonal to
the mean under heavy-tailed ConvNeXt activations and captures peak-activation/foreground signal that a
mean-only prototype cannot represent — attacking FSC147's dim-small-instance failure mode. Single scalar
per channel (no layout → not N0056's fine entropy), single slot (not N0055's cardinality).

## Param table
xproj_max: 384×256 + 256 = **98,560**. Head 3.505M → **~3.60M**; total 31.32M → **31.42M ≤ 32M ✓**.
Interface invariant: condenser still sees ONE (B,K,256) fused prototype; keys/attn byte-identical.

## Why (grounding)
- N0054's +0.95 = the sole validated win: coarse, single-slot, additive. 2nd-order statistic (max) is
  the orthogonal extension of that exactly-positive family; a 2nd mean is its redundant twin.
- N0055/N0056 bound: do NOT split prototype cardinality (N0055) or add fine-space entropy (N0056); a
  global coarse max violates neither.
- Remap of H0083's original BECAUSE: "orthogonal to the 3×3 LAYOUT" was false (the 3×3 is GAP'd to a
  per-channel mean, no layout survives). Correct premise: **max is an ORDER-STATE statistic, orthogonal
  to the mean estimator, on the same coarse single-slot additive axis.**

## Hypothesis (verbatim)
**H0084** IF [a coarse MAX-order-statistic summary (adaptive_max_pool2d over the already-aligned 7×7
ROI, then xproj to 256, additive to the SAME fused exemplar prototype) is added alongside the existing
mean XScale] IN [frozen N0054 champion, 30ep @384, attention paths untouched] THEN [val MAE < 19.647]
BECAUSE [the single positive portable axis is coarse single-slot additive granularity; the max is an
order statistic near-orthogonal to XScale's per-channel mean under heavy-tailed activations, supplying
peak/foreground signal a mean-only prototype lacks, without N0055's cardinality split or N0056's fine
entropy]. **DISPROVED IF best val MAE > 20.0** (leaving the positive axis / adding harmful redundancy).

## Gates (vs N0054 19.647; noise ±0.25)
- **R1 smoke**: stub/real backbone; use_xscale+use_hxscale on; params ≤32M; density (B,1,96,96); loss
  finite+decreasing; use_hxscale=False restores exact champion (31.32M/3.505M).
- **R2 CONFIRM (new champion, H0083/H0084 supported)** if best val MAE **< 19.45** (real gain).
  If first best < 19.40, run a 2nd seed before claiming champion.
- **R3 WEAK-KEEP (axis saturated / informative tie, H0084 neither confirmed nor refuted)** if
  **19.45 ≤ best ≤ 20.0** — a tie ≈19.6 is NOT a killer; it signals the positive axis is near-converged.
- **R4 FAIL (refutes H0084; escaped positive axis / harmful redundancy)** if best **> 20.0**.

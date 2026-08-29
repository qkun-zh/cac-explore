# Diagnostic — N0060_xscale_max (FAIL, 30/30 ep, best 22.886 / final 23.010)

## Root cause — DESIGN failure, not operational
Clean run (status=success, 30/30 ep, oom=false, instability=false, params_M=31.42 exactly). The FAIL gate
(>20.0) fired decisively: best 22.886 is **+3.24 vs champion 19.647**; tail E17-30 val plateaued 22.9-23.4
while train loss kept dropping 3.69→2.70 — the same overfit-to-scale signature seen on N0056. The added
MAX summary actively hurts.

## Implementation faithfulness — FAITHFUL (model.py:100-103)
```
if self.use_xscale_max:
    coarse2 = F.adaptive_max_pool2d(roi,(1,1)).squeeze(-1).squeeze(-1)   # (B*K,384) over SAME aligned 7x7 roi
    out = out + self.xproj_max(coarse2).view(B,K,-1)                      # additive
```
Uses the SAME `roi` (roi_align output_size=(7,7), model.py:88) as idea.md requires — zero extra FLOPs as
claimed. Order matches idea.md: mean XScale add (model.py:99, over its own 3x3 roi2) runs FIRST, then max
add (model.py:103). Both write into the same `out`. No coupling: `self.xproj` and `self.xproj_max` are
disjoint param sets, so the two adds backprop independently — a **natural additive sum, no gradient
conflict**. `use_xscale_max=False` restores the exact champion (single-switch, bit-identical path).

## Is this a NEW failure mode? NO — re-run of the known exemplar coarse-summary family
Params +98,560 (head 3.60M) — capacity barely moved; attention paths and prototype cardinality untouched,
so neither N0055 (cardinality) nor the operator-swap (N0057/58/59) axes are implicated. The axis is the
negative table's own: XScale (mean) is the lone positive; a 2nd MAX order-statistic over the SAME spatial
source is **harmful (+3.24)**, not just neutral.

## Axis established: the exemplar coarse-summary / aggregation slot is now fully mapped
N0060 (+3.24 max) + N0058 (+3.50 part-pool) + N0059 (+1.31 matched producer) + N0057 (+1.43 consumer):
**every second summary of the same ROI — mean, max, grid — and every attention/operator swap is negative;
N0054 XScale is the unique single positive coarse-slider.** The H0083 "headroom on the positive axis"
question is answered: no headroom — the axis is saturated at exactly one slot.

## Recommendation: STOP probing the exemplar coarse-summary slot
Do NOT retry another order-statistic / spatial summary on the aligned ROI (mean, max, grid, L2) — mapped,
negative. Move the lineage to a regime/extent change or a genuinely different pluggable component.

## failure_modes.md — NO append
Design/axis result, not an ops pitfall; nothing operationally new (clean run, faithful code).

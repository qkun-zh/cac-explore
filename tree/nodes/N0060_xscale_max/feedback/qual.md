# Qualitative Feedback — N0060_xscale_max (FAIL, best 22.886 / final 23.010 = +3.24 vs 19.647)

Verify switch, params (+98,560; head 3.60M / total 31.42M), use_xscale_max=False restores N0054 exactly;
attention self/cross + GCA + decoder untouched; condenser still sees single (B,K,256) fused prototype.

(a) Stacking a second coarse summary past the sweet spot. The champion's mean-XScale is the one validated
positive additive on the exemplar interface (+0.95). N0060 adds a second coarse single-slot stat onto
the SAME prototype and pays +3.24 of pure destruction with zero benefit anywhere in the curve (best at
E23 22.886, tail already 23.08-23.35 and climbing to 23.010 final — it never recovers toward 19.6, it
oscilates *above* the plateau 22.9-23.4). The mean already occupied the informative portion of this
axis; the additive prototype has no residual capacity that a second coarse scalar can fill. This reads
as "stacking past the sweet spot" — the axis is a single-additive-slot axis, and the second shovel hits
bedrock and then digs into noise.

(b) MAX mechanism: order-statistic noise swamps the denoised mean. Adaptive-max over the 7x7 ROI picks,
per channel, exactly one extreme cell; for a crowded exemplar box that extreme is frequently background,
occluder, or a shared-neighbor bleed, and under heavy-tailed ConvNeXt a single loud logit can inject a
large, unstable, sign-arbitrary offset into the (B,K,256) readout before the Condenser. Unlike the mean
(which averages away outliers), MAX *selects* the worst outlier, so it acts as adversarial/order-statistic
noise on top of a denoised prototype. That is precisely the size of effect seen: +3.24, worse than the
matched-capacity +1.31 of N0059 (which at least kept full-token mean structure) and only ~0.26 under the
part-pool +3.50 of N0058 — i.e. MAX behaves like the coarse-destruction family, not like benign capacity.

(c) Scale/harm dependence. The harm is scale-choice-dependent, not scale-invariant: pooling over 49 cells
maximizes over a large window, so the max settles on the single loudest peak in the box, collapsing all
MSBX/scale information into one value that is not robust to box resizing. This is the structurally same
failure as N0058's part-pool (which destroyed spatial layout), extended to the order-statistic: max-pool-
to-one is a layout-destroying, statistics-destroying reduction. The +3.2 magnitude (vs N0058 +3.50) is
the fingerprint of this family.

(d) Collision with the mean-XScale gradient direction. MAX and mean are nearly decorrelated *as
estimators* on heavy-tailed distributions, but on the SAME 2nd-scale ROI the mean's pooled support
(source channel dim 384) and the max's selected single cell overlap in input space; gradients through
the additive branch fight to represent a mean envelope and a point-spike simultaneously in one 256-d
readout. The collision does not appear as a stable cue but as a tug-of-war that the tail plateaus show:
train loss keeps dropping (2.69 at E30) while val MAE oscillates 23.0-23.4 — overfit to the max-noise,
generalization flat. This is a genuine sign the two summaries compete on the fused prototype.

Fingerprint verdict — NOT a genuinely new failure; this is a re-run of the known coarse-destruction
family (N0058 part-pool), now via the order-statistic instead of pooled layout. N0060 shares the
magnitude band (+3.2 vs N0058 +3.50) and the "flattened tail, no best-epoch gain" signature. The
distinct factor (MAX vs mean) is real but does not produce a distinctive failure shape.

Confounds — limited: +98,560 is small and capacity-matched modulo that (so not N0059's capacity-
confound); attention untouched; condenser sees an unchanged single fused prototype so the cardinality
axis (N0055) is NOT implicated. Remaining caveat: MAX and its xproj share the exemplar encoder's frozen
path, but forward is pure parameter (zero extra FLOPs on the reused ROI), leaving only the additive
signal itself as the cause. Single seed.

Conclusion — CLOSE the 'second-coarse-summary on same ROI' extension. The single-coarse-additive axis
is saturated at exactly one summary (the mean-XScale); a second coarse stat, whether mean (redundant) or
MAX (order-statistic destructive), only degrades. This closes the extension in both estimator directions;
further headroom, if any, lies outside coarse single-slot additive on this interface, not in a third stat.

# quantitative.md — N0017_dino_tailreweight

## reasoning
Best MAE 22.187 @E30 (RMSE 82.870, ratio 3.73); final E40 22.598/85.786 (ratio 3.80) vs parent
champion N0010 21.531/3.63. Δ=+0.66 (+3.1%) — WORSE on both axes; H0026 missed both claims
(≤19.8 ✗, <3.4 ✗) and fired its own disproof bar (>21.53 ✗). Cost identical to parent
(1272.6s vs 1275s, 23.11M, no OOM/instability). Trajectory: ahead of parent early (best ≤25.10
by E12/13 vs parent-E13 25.50), then a 3-epoch regression spike E13-15 (27.43→27.61, +2.5 off
its own best), recovery from E16, monotone grind to best @E30 (75% of schedule), flat 22.5-22.9
for last 10 epochs — same late-plateau waste pattern as parent. CRITICAL SPEC AUDIT: idea.md
motivation says upweight rare high-count images, but the registered H0026 formula AND engine
code (train.py:166 `w=1/clamp(gt_c,1)^0.5`, mean-1 normalized) DOWN-weight them. H0026 pinned
the formula, so the test is faithful to the hypothesis-as-booked; the motivation prose was the
sign error. As-coded reading is self-consistent: shrinking the gradient share of dense/error-
dominant samples let catastrophic outliers drift further → ratio worsened 3.63→3.80, and the
mild MAE penalty (+0.66) shows median-case accuracy bought nothing in return. Third failure of
loss-space outlier surgery on the champion (Huber cap N0011 26.68; tail-reweight 22.19;
count-w unisolated): the residual tail is not loss-shape-limited.

## actionable_feedback
- Book H0026 contradiction w≈0.95 (confound-free single-change design, decisive double miss).
- Book H0022 cross-ref contradiction w≈0.70: ratio criterion fires (3.80≥3.5) via adjacent
  variant (inverse-sqrt, not log-count/stratified); "no MAE loss>1.0" clause held (+0.66).
- STOP the loss-space tail-reweighting family (inverse-freq, stratified sampling) on champion
  features; remaining variants are predicted-neutral-to-harmful without new mechanism evidence.
- MANDATORY for any future loss-level node: stratified eval by true-count bucket (promised in
  idea.md tunable_aspects, never implemented) — without it loss-space attribution stays blind.
- Add coding-checklist rule: hypothesis formula must equal motivation direction (this node's
  prose/spec divergence nearly caused a false "untested retry").
- Do not retry flipped-sign upweighting (w∝sqrt(count)) except as a cheap rider on a
  capacity/resolution node; priority stays N0014/N0015 resolution merges where the tail likely
  lives (perception limit, not loss limit).
- Schedule note: two nodes in a row peaked at 65-75% then went flat ≥10ep — adopt patience-5
  early stop (~save 300s/run) once augreg merge lands.

## hypothesis_updates
- hyp_id: H0026 — evidence_type: quantitative_contradiction; strength: 0.95; reasoning: clean
  single-lever test of booked formula on champion recipe; both predictions missed decisively
  (22.19 vs ≤19.8; 3.73-3.80 vs <3.4) and disproof bar >21.53 fired; c 0.50→0.41.
- hyp_id: H0022 (cross-ref) — evidence_type: quantitative_contradiction_partial; strength: 0.70;
  reasoning: family member (inverse-sqrt reweight, mean-1 norm) failed ratio criterion 3.80≥3.5
  while MAE clause held (+0.66<1.0); combined with H0020/N0011 refutation, whole
  loss-space-outlier-surgery quadrant weighs against; c 0.50→0.43.

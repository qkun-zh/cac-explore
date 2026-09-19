# Synthesis — N0004_h0010 (JOINT: H0010 per-exemplar reliability gate × H0009 image-global channel gate)

## 1. Joint-test verdict narrative

Seeded, same-protocol, same-seed pair (both `seed 20260830`, 32 epochs, budget 1800 s, EMA-inference val MAE):

| | best val MAE | @ep | budget_hit |
|---|---|---|---|
| Parent N0001_champion (seeded) | **21.4589** | 32 | false |
| Child N0004_h0010 (both gates) | **23.3824** | 32 | false |

- **Δ = 23.3824 − 21.4589 = +1.9235 child-WORSE** (ratio 1.0896 ≈ +9.0%). Numbers read off `result.json`
  (`best_mae` 21.458858… / 23.382432…; M0004 `best_epoch` 32 = final epoch).
- Both hypotheses H0010 and H0009 carry the **same bar**: support iff ≥ 0.30 **lower** than parent,
  i.e. child ≤ 21.1589. Observed 23.3824 is **+2.2235 above that bar** (~7.4× the bar width).
  **Both falsifier bars are missed.** Neither gate passes confirmation; the joint configuration is
  clearly negative and both support conditions fail by a wide margin.
- **Recorded (already in ledger before this synthesis, two separate CLI evidence events,
  node=N0004_h0010 — NOT re-recorded here):**
  - `contradicts w=1.0` on **H0010** → conf 0.500 → **0.400** (source node N0004_h0010).
  - `contradicts w=1.0` on **H0009** → second test; conf 0.400 → **0.320** (source node N0004_h0010).
- Cross-node context (same protocol, indicative not a split test): H0009-only sibling N0003_h0009
  landed at 22.6076 (Δ = +1.1487 vs 21.4589, itself beyond its −0.30 bar); the joint node is
  **+0.7749 worse than H0009-only**, so layering the per-exemplar gate on top of the channel gate
  degrades further on an already-failing mechanism.

## 2. Causality / separation finding — measured vs attributable

This was a **joint set** (`use_exemplar_gate=true` AND `use_channel_gate=true`), so the node yields
ONE Δ against the gated-off champion and the two falsifier bars are evaluated against the same single
trajectory. A two-variable change measured at one point **cannot be deconvolved** into per-variable
effects (feedback/causal.md §1):

- The +0.77 "marginal for H0010-on-top-of-H0009" (23.38 − 22.61) is NOT H0010's solo effect. The gates
  are nested multiplicatively in the SAME norm-free K/V path of condenser cross-attention
  (`e * eg.unsqueeze(-1)` then `e * gate.unsqueeze(1)`, model.py:173–178); no separability is justified.
- **Attributable:** the damage of +1.92 is dominated by the known-harmful, re-encoded H0009 (individually
  established harmful on N0003, Δ = +1.149; per rule 11 its re-encoding must dominate attribution).
  For H0009 the node adds no separable evidence beyond N0003; it is consistent-with-harm at best.
- **Measured but NOT attributable to H0010's mechanism:** the joint configuration is +1.92 worse and
  both bars are missed — that is all this node measures. H0010's *solo* per-exemplar-gate effect
  remains **UNANSWERED** by this node; it is NOT booked as evidence against H0010's mechanism claim.
- Also inherited (causal.md §2): approximate identity init (sigmoid(3.0)≈0.953 → ~4.7% step-0 shrink),
  the exemplar branch's LR is modulated by gate activity, AdamW decay drifts bias+3 toward 0, near-
  saturation dead-head risk, and no gate statistics logged (mechanism clause untestable, same gap as
  N0003). Both gates can only attenuate — the exemplar path is never given a mechanism to amplify.

## 3. Calibration (verbatim, `discovery calibration`, post-booking)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      9       0      0%       0.36
[0.50,0.75)     16      10     62%       0.55
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall         25      10     40%      -
reliability error (weighted |rate − pred_conf|): 0.176
direction accuracy (decisive tests, c≠0.5): 15/15 = 100%
```

The two N0004 contradicts moved reliability error 0.195 → **0.176** (<0.2, WARN cleared, stable);
direction accuracy holds at **15/15 = 100%**.

## 4. Booked next hypotheses (K_SYNTH = 2)

Booked via `discovery hypo N0001_champion --new "..."` (machine selection/validation, format +
novelty gated — no hand-edits; both texts from tmp_ideas_round2 verbatim).

| Child node | Parent | New hypothesis | Pre-registered falsifier |
|---|---|---|---|
| **N0006_h0009** | N0001_champion | **H0012** — count-scaled condenser attention temperature (MLP over pooled fine + mean exemplar → per-image temperature before softmax) | final val MAE is not at least 0.40 lower than the champion with the fixed-temperature condenser |
| **N0007_h0013** | N0001_champion | **H0013** — learned depth-to-space subpixel upsample replacing bilinear in the FineFuser | final val MAE is not at least 0.30 lower than with bilinear upsampling |

Machine Q_t at booking (verbatim from CLI):
- `hypo N0001_champion --new <candidate_2>` → `Q_t (parent=N0001_champion) -> ['H0009', 'H0011', 'H0012']` →
  `[created] N0006_h0009 nested under N0001_champion`. Note: the machine pulled **H0009 and H0011 into the
  same child** as H0012 (multi-hypothesis selection, ≤2·K_HYPO); H0009/H0010 here will book as
  `contradicts`-events on the existing ids, not new duplicates (rule 11).
- `hypo N0001_champion --new <candidate_5>` → `Q_t (parent=N0001_champion) -> ['H0013', 'H0010', 'H0011', 'H0012']` →
  `[created] N0007_h0013 nested under N0001_champion`. The machine again **composed H0010, H0011 and H0012
  into the same child** as H0013.

Logic of the pair: candidate_2 (H0012) delivers the "count-dependent attention" H0009's mechanism text
claimed but its image-global channel gate could not implement — the condenser cross-attn logits are now
divided by a per-image temperature scalar, so the condenser genuinely changes behavior as count varies
(directional: sparse scenes concentrate on one prototype, dense scenes spread mass). Candidate_5 (H0013)
is a counter-intuitive low-cost trick confined to the FineFuser (1×1 conv 128→512 + PixelShuffle(2),
~66k params): sharpen the density-head support that the count sum integrates, no feature-interface change.

One watch-item passed to the Lead: both children are **multi-hypothesis nodes** (not the single-switch
clean ablation the Idea Agent wrote). N0006 tests H0009+H0011+H0012 jointly and N0007 tests
H0013+H0010+H0011+H0012 jointly — the same separability caveat this node hit applies to their
attribution; their falsifiers must be read as joint-set bars unless the Lead re-books clean single-shot
children for the champion line.
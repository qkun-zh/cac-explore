# Synthesis — N0003_h0009 (H0009, image-global channel gate on exemplar tokens)

## 1. Verdict narrative

**H0009 DISPROVED.**

Seeded, same-protocol, same-seed pair (both `seed 20260830`, 32 epochs, budget 1800 s):

| | best val MAE | @ep | budget_hit |
|---|---|---|---|
| Parent N0001_champion (seeded rerun) | **21.4589** | 32 | false |
| Child N0003_h0009 (gated) | **22.6076** | 32 | false |

- Δ = parent − child = 21.4589 − 22.6076 = **−1.149**.
- Pre-registered falsifier (idea.md): support requires val MAE **≥ 0.30 lower** than the
  unconditional condenser, i.e. Δ ≥ +0.30.
- Observed Δ = −1.149 is **far below the +0.30 bar** — the gate did not just fail to
  improve, it *hurt* by ~1.15 MAE. The contradicting evidence (w=1.0, source N0003_h0009)
  was already recorded in the ledger; confidence moved 0.500 → **0.400** (H0009 remains
  `uncertain` at c>0.25). Status per AGENTS §3/§11: this is a decisive miss, and H0009's
  mechanism is **not retried silently** — any re-encoding must book against H0009 with a
  NEW falsifier.
- No rerun/chip was fabricated: both numbers are read off `result.json` (parent
  `best_mae` 21.458858…, `best_epoch` 32; child `best_mae` 22.607566…, `best_epoch` 32),
  and the earlier unseeded migration artifact (19.647) is superseded as the comparator.

## 2. Mechanism-fidelity finding

The coded gate and the claimed mechanism mismatch; attribution is therefore weak:

- The BECAUSE clause claimed a **per-exemplar** channel-gain ("steering the condenser
  toward scale-consistent exemplar weights"). The implementation
  (`model.py:168-170`) computes ONE `sigmoid(Linear(GAP(fine)))` vector of shape (B, 256)
  and applies it identically to **all K exemplars** of the image
  (`e * gate.unsqueeze(1)`). It is an **image-global, per-channel** gain — it cannot
  down-regulate one exemplar against another of the same image, so the claimed
  mechanism is not expressible by this code (qual.md §1, causal.md §1).
- A near-constant gate is algebraically absorbable into the condenser's K/V projections;
  the surviving ingredients are ~33,024 added params (Linear 128→256) and conditioning on
  fine-content statistics — capacity/input effects, not the hypothesized attention path.
- Init confound: default Linear init ⇒ sigmoid ≈ 0.5 at step 0, halving exemplar
  magnitudes relative to the parent (identity-preserving init absent).
- No gate statistics (mean/saturation/variance) were logged, so the single scalar outcome
  cannot discriminate mechanism from inert reparam.
- This mechanism-fidelity lesson is exactly what the two booked children (below) address:
  H0010 restores the true per-exemplar gate; H0011 surgically severs the conditioning to
  isolate capacity-vs-conditioning.

## 3. Calibration (verbatim, `discovery calibration`, post-booking)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      8       0      0%       0.35
[0.50,0.75)     15      10     67%       0.55
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall         23      10     43%      -
reliability error (weighted |rate − pred_conf|): 0.195
direction accuracy (decisive tests, c≠0.5): 14/14 = 100%
```

H0009's contradict moved reliability error 0.227 → **0.195** (WARN cleared, <0.2);
direction accuracy holds at 14/14.

## 4. Booked next hypotheses (K_SYNTH = 2)

Booked via `discovery hypo <parent> --new` (machine selection, format + novelty gated —
no hand-edits). Both proposals passed the novelty gate in tmp_ideas_round2.

| Child node | Parent | Hypothesis | Pre-registered falsifier |
|---|---|---|---|
| **N0004_h0010** | N0001_champion | **H0010** — per-exemplar reliability gate (sigmoid of a linear readout of *each* exemplar token, rescaling that exemplar only) | final val MAE is not at least 0.30 lower than the champion with the gate removed |
| **N0005_h0010** | N0003_h0009 | **H0011** — equal-capacity frozen-random-input gate control (same +33,024 gate, input = frozen random vector, no fine-conditioning) | frozen-random-gate final val MAE is within 0.30 of the conditioned-gate result 22.608 |

Machine Q_t at booking:
- `N0001_champion` → `Q_t = ['H0010', 'H0009']` (H0009 re-qualifies on this fresh ancestry as an uncertain untested hypothesis; any future re-test must attach evidence to the existing id, per §11).
- `N0003_h0009` → `Q_t = ['H0010', 'H0011']` (H0009 dropped on theta; the control books as the child of the just-refuted node).

Logic of the pair: H0010 is the only strict per-exemplar gate that could ever realize the
attention-steering mechanism H0009 claimed but could not implement — an identity-preserving
init, ~257 params, and a per-exemplar scalar that cannot be absorbed as a constant rescale.
H0011 is causal.md §4's equal-capacity control verbatim: it keeps H0009's exact +33,024
affine and sigmoid multiply but feeds a content-free frozen input, so if the control
reproduces 22.608 the gate benefit is demoted from mechanism to inert capacity. Together
they separate mechanism (per-exemplar steering), capacity (equal-param control), and
conditioning (frozen-input control).
# Causal feedback — N0004_h0010 (JOINT: H0010 per-exemplar reliability gate + H0009 image-global channel gate)

Observed: best val MAE **23.3824 @ ep32** (final epoch), 32 real-data epochs, 1707.8s/1800s (epoch cap, not budget), EMA-inference MAE. No per-epoch val curve is readable from the log; final = best.

Reference points (all seeded regime, best-at-last-epoch):
| node | config | best val MAE | Δ vs seeded champion |
|---|---|---|---|
| N0001_champion | no gates | 21.4589 | — |
| N0003_h0009 | channel gate only | 22.6076 | +1.15 (worse) |
| N0004_h0010 | exemplar gate + channel gate | 23.3824 | **+1.92 (worse)** |

## 1. Joint-set inference — the −1.92 is NOT attributable to either mechanism
Both gates were encoded at once (config `use_exemplar_gate=true` AND `use_channel_gate=true`), so the node yields ONE joint Δ against the gated-off champion; the two falsifier bars are evaluated against the same single trajectory. This is the infeasibility case: a two-variable change measured at one point cannot be deconvolved into per-variable effects.

- Additive temptation: 23.38 (joint) minus 22.61 (channel-gate alone) = +0.77 "marginal" for H0010-on-top-of-H0009. This is NOT H0010's solo effect and must not be read as one. The two gates are NESTED MULTIPLICATIVELY (`e * eg.unsqueeze(-1)` then `e * gate.unsqueeze(1)`, model.py:173-178) inside the same K/V path of cross-attention; there is no justification for separability, and gates re-enter the same norm-free K/V, so they interact in the softmax, not additively in the loss.
- Strong prior (rule 11): H0009 was individually established as harmful on N0003 (+1.15 vs parent; its causal review found the mechanism inexpressible and the outcome beyond noise). Re-encoding a treated-as-harmful gate must dominate a joint node's damage attribution. The +1.92 joint outcome is therefore dominated by the known-harmful component; it says essentially nothing about H0010 by itself.
- **Conclusion: whether the identity-init per-exemplar gate would differ ALONE is UNANSWERED by this node.** Do not book this node as evidence against H0010's mechanism claim.

## 2. Residual confounds of the identity-init design (apply to any future clean H0010 test)
- **Step-0 perturbation ≈ 4.7%.** sigmoid(3.0)=0.9526, so at init every exemplar embedding is contracted by (1−0.9526)≈4.74% before it enters the condenser. This is a real input-level scale change present at step 0, uniform across exemplars this once (W=0) but non-identity, and it persists for exactly the first steps. The "identity init" is approximate, not exact.
- **Gradient-carrying changes the exemplar path's effective LR.** `e` is BOTH the gate input and the rescaled quantity, so the exemplar branch gradient carries factor σ(z) ≈ 0.9526 (≈4.7% attenuation at init) plus a second route through the gate: σ'(3) ≈ 0.0452 times `e` feeds ∂L/∂b, and once W leaves 0 a cross-term `e·σ'(z)·W` couples gate activity into the exemplar gradient. The exemplar path does NOT train with champion-equivalent dynamics even if the gate never discriminates; its effective step size is modulated by gate activity for the whole run.
- **Identity init is not stationary under the protocol.** AdamW weight decay (0.08) is decoupled and applies to the gate bias too: bias=+3 drifts toward 0 under L2 decay regardless of loss gradient, pulling sigmoid(bias) from 0.953 toward 0.5 early. The "gate≈1 pass-through" state is not preserved; a clean test must expect and log this.
- **Saturation / dead-head.** With σ'≈0.045 already small, any W growth saturates the gate; near-saturated gates emit near-zero gradient and freeze a per-exemplar rescale that softmax then treats as a hard trust/no-trust switch.
- **Unobservable mechanism clause.** No gate statistics (mean/std/saturation) are logged anywhere in run_node.log; we cannot even check the gate stayed near 1 or became content-dependent. Same measurement gap N0003 flagged — the mechanism clause is untestable from this node.
- **EMA lag (inherited).** The gate is EMA'd at 0.999 like all weights; a slowly-adapting multiplicative rescale under strong EMA lags its own gradient, consistent with "still improving at last epoch" regardless of mechanism. (Also e is used as K/V without LayerNorm, so the gate's only lever is raw magnitude scaling inside the attention softmax — the mechanism is expressible here, unlike in N0003, but it is the sole lever.)

## 3. Proposed clean single-shot test for H0010 (proposal only — NOT an action)
Switch set, minimal delta from seeded champion: copy N0001_champion config/model, set **`use_exemplar_gate = true`**, keep **`use_channel_gate = false`** (do NOT re-encode H0009), keep the exact identity init (weight=0, bias=3.0), keep seed 20260830, epochs=32, wd=0.08, lr=1e-3, all else identical. Add ONLY passive logging: per-epoch gate mean/std/saturation and gate-vs-exemplar-ROI-scale correlation (to make the BECAUSE clause observable). Falsifier, matched to the pre-registered H0010 bar against the seeded champion: **DISPROVED IF best val MAE is not ≤ 21.1589 (≥0.30 lower than 21.4589)**; endpoint = best EMA-inference val MAE across 32 epochs, one seeded trajectory, no re-roll. Nothing else changes.

## 4. Verdict framing vs the pre-registered bars
- **Measured:** the JOINT configuration is +1.92 worse than the seeded champion and +0.77 worse than the channel-gate-alone sibling. Both falsifier-bar conditions ("not ≥0.30 lower") are missed for this configuration. That is all this node measures.
- **Attributable:** neither falsifier bar is attributable to its hypothesis. For H0010 the node confers essentially **no legitimate evidence** — a joint configuration that did worse does not support or refute the solo mechanism, and the strong prior assigns the damage to the re-encoded H0009. For H0009 this node adds **no separable evidence**; it is consistent-with-harm at best, and the N0003 verdict already governs. H0010's solo status remains open.
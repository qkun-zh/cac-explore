# Qualitative Feedback — N0058 PMOM (E15, early-stopped)

## 1. What PMOM computes (model.py ExemplarEncoder use_pmom path)
From each K=3 ROI: `adaptive_avg_pool2d(roi,2)` → 4 part vectors h_p(384) → 2nd-order moments
`m_p=[h_p; h_p²]`(768) → learned gate `softmax(Linear→16→1)` over 4 parts → weighted sum `Σ α⊙m` →
`moment_proj(768→256)` → + shape_mlp + XScale coarse summary. Part-gating is adaptive per exemplar.

**RETAINS vs the 2-layer TransformerEncoder path**: per-part category statistics — the mean (h) and
intra-part intensity energy (h²) of 4 coarse quads, adaptively weighted by a learned relevance gate.
Crucially it preserves the **single fused prototype (B,K,256)** contract plus the XScale global summary
that the champion also keeps, so the condenser consumer sees an unchanged interface.

**LOSES**:
- **Per-part relative arrangement**: pool→flatten destroys 2×2 spatial topology; parts are unordered,
  position-agnostic after flatten. The 49-token self-attn path could, in principle, encode layout.
- **Within-part texture**: avg-pool2d(2) collapses a 7×7 ROI to 4 scalars-per-channel means; all local
  structure inside a part is averaged away (only mean+mean² survive). Fine edges/heads/tails of the
  exemplar category are gone before the condenser.
- **Global shape beyond a coarse mean**: only 4 quads + one XScale GAP channel-wise mean; no second-order
  moment over the whole ROI, no cross-part covariance. Any shape cue needing more than 4-bin Haar-like
  summary is lost.
- **Capacity**: 209k vs the removed ~1.68M transformer+proj; the expressivity drop is 8×.

## 2. Training dynamics signatures (train.log)
Loss: E1 spike 130.48 (AdamW 1e-3 cold-start, transient) → E2 6.66 → smooth decay 4.06@E13 → 3.97@E15.
MAE noisy but descending: 32.5→23.15(***BEST@13)→23.5→23.8. RMSE stays high (77.8–109) and oscillates
without a downward trend (83–95 typical). The E1 spike-then-quick-settle is the classic frozen-backbone,
large-lr warm transient — NOT instability; the curve is healthy, monotone-decreasing loss, no divergence,
no NaN (oom=false). It is **under-converged**, not unstable.

## 3. Mechanism plausibility (moments vs intra-object structure)
The mechanism predicts: if 2nd-order polynomial moments underfit intra-object appearance, expect
chronically high RMSE on dense/occluded scenes. Observed RMSE sits at 77.8–95.3 throughout — **yes,
consistent**. The operator can't route enough exemplar texture to the condenser; dense-scene counting
suffers precisely as a moment-summary would predict. RMSE≈77.8 best is still above the N0054 champion's
74.05, aligned with the dilution hypothesis.

## 4. Verdict: falsified vs under-converged at E15?
**Falsified, not merely under-converged.** At E13 best MAE=23.15 is already **+3.5 over parent 19.647**,
well past the pre-registered kill (≥20.4) and breaching the ep16+ early-stop bar (+1.5 ⇒ ≥21.147) by E13.
The curve has plateaued (23.2→23.5→23.8 over E13–15, essentially flat), so it is not trending toward 19.6.
A 30th epoch would NOT plausibly close the 2.2–3.5 gap: 17 more epochs at the observed ~0.2/ep drift would
buy ≤0.5, and there is no structural reason for a late hook — PMOM's ~209k params lack the capacity to
recover what avg-pool+mean/mean² discards. Consistent with the N0055/56/57 pattern and the R3 kill-clear
falsifier, the 2-layer exemplar self-attention is itself **load-bearing** here: the operator swap degrades
to N0057-magnitude. H0080 (moments>self-attn on K=3 tiny ROIs) is **refuted**, not under-converged.

**Action**: accept the early-stop kill; route the remaining budget to a producer-side change that KEEPS
transformer capacity (e.g. cheaper attention over all 49 tokens, not destructive pooling).

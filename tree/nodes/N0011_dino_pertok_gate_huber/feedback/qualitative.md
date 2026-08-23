# Qualitative Feedback — N0011_dino_pertok_gate_huber

## reasoning
Context: best MAE 26.68@E34 (+24% vs parent 21.53), final 26.89/RMSE 94.34 (ratio 3.51–3.52), 23.16M params, clean 40ep run (no OOM/instability), val plateaued ~27 from E20.

**Design clarity — triple-confound confirmed and central.** Three levers changed vs N0010 at once: (1) scalar gate → per-token gate MLP, (2) MSE → Huber(δ=5), (3) count-w 1.0 → 0.3. The two other reviewers' diagnosis is correct. Worse, idea.md's `## Delta vs Parent` claims the count-w revert "isolates the Huber effect" — this is inverted logic: relative to the parent it changes BOTH loss shape AND loss mix, so neither named hypothesis is tested cleanly. A +5.15 MAE regression cannot be attributed to per-token gating or Huber individually. Design intent was surgical; execution bundled an undeclared third variable.

**Naming ✅**: slug, `PerTokenGateMLP` (model.py:30), `t6_proj/t11_proj`, config keys (`gate_mlp_hidden`, `huber_delta`, `loss_function`) all descriptive and match spec. Legacy class-name reuse (DinoPromptV2/PromptEncoderV2) across lineage is tolerated by engine import isolation.

**Docstrings ⚠️**: module + PerTokenGateMLP documented (model.py:1,31). The novel mechanism itself — gate application (model.py:84-85) — has no inline rationale; PromptEncoderV2/build_model undocumented; magic `ch = 384` unexplained.

**MODIFICATIONS.md**: absent. PROTOCOL §2 does not require it, so protocol-compliant — but this node shows why it should exist: idea.md's delta description mislabels the third change, and no separate artifact forced an honest diff. An explicit change table (lever / parent value / child value / declared?) would have surfaced the confound before training.

**Hypothesis compliance vs idea.md claims**: H0019/H0020 follow the AGENTS §7 format (scoped, mechanistic, falsifiable). Two slips: (a) H0020's neutral band (<3.0 pass, >3.5 fail) is wide — observed 3.51 lands just past the line, making refutation marginal rather than decisive; (b) idea.md estimates the gate MLP at "~0.03M/30K params"; actual Linear(768→64)+Linear(64→2) ≈ 49.3K (0.05M). Cosmetic, but claims should be checked at coding time.

## actionable_feedback
1. Adopt a one-lever rule: >1 mechanism change requires ablation configs or dropping a lever; silent loss-weight changes are forbidden (they masquerade as "reverting to default").
2. Correct the isolation reasoning pattern in future idea.md deltas — reversion-to-default is only isolation if the parent also used the default.
3. Add 2–3 inline comments at model.py:84-85 (gate semantics) and name/document the channel constant.
4. Tighten falsification bands so thresholds sit where outcomes are discriminative (H0020's 3.5 line was inside the noise of the parent's own 3.6 ratio).
5. Log gate diagnostics (mean softmax weight per tap per epoch) — cheap, and would have separated "gate collapsed" from "fusion genuinely worse".

## hypothesis_updates
- H0019 | contradicts | 0.90 | Best MAE 26.68 vs parent 21.53, direction opposite to the predicted ≥1.0 decrease. Huber/count-w confounds prevent attributing the full regression to gating, but no positive signal survives; strong contradiction, not proof the mechanism is universally bad.
- H0020 | contradicts | 0.55 | RMSE/MAE 3.51 (final) / 3.52 (best) just crosses the >3.5 disproof line; outlier ratio essentially unchanged vs parent 3.6 despite capped gradients. Weak-moderate: δ=5 may simply be too large, and the count-w change contaminates attribution.

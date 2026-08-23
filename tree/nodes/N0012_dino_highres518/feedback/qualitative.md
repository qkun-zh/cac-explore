# Qualitative Feedback — N0012_dino_highres518

## reasoning

**Code quality (good).** `model.py` is clean: module docstring states the node's purpose, names are meaningful (`t6_proj`/`t11_proj` map to out_indices=(6,11), `layer_logits`, `PromptEncoderV2`), and the frozen-backbone invariant is enforced twice (`requires_grad_(False)` + persistent `eval()` override of `train()`). The BCHW/BCHW-token reshape guard (`f6.ndim == 3`) correctly applies the documented timm trap from memory/failure_modes.md. `config.py` is minimal and self-explanatory; all idea.md invariants are visible in code (input 518 divisible by 14, count_w=1.0, ≤32M).

**Gaps.** No `MODIFICATIONS.md` — acceptable here since delta vs parent N0010 is a single config value and idea.md §Delta documents it precisely, but the protocol habit of recording deltas per-node would help future clones. `model.py` hardcodes `ch=384` rather than deriving from cfg; fine for an exact-clone ablation but limits reuse.

**Hypothesis compliance (high).** Code matches idea.md exactly: frozen DINOv2-S reg4 dual taps at (6,11) with dynamic_img_size=True, scalar softmax gate over 2 layers, Fourier+log-area prompt (PromptEncoderV2 with clamp_min on w/h and log-area clamped to [-13.8,0]), adapter 384→768→384 GELU drop0.1, 1×1 conv head → [B,1,37,37]. No unauthorized additions; the "clone except input_size" claim holds.

**Design clarity (excellent).** The resolution-scaling design is the cleanest possible single-lever experiment: one variable changed against a validated champion, falsification criteria pre-registered (MAE >21.53 or RMSE/MAE ≥3.63 refutes).

**Run context.** Truncated at E18/40 (~45% schedule) by wall-clock, not by failure: loss still decreasing (16.88→16.30), MAE improving every epoch (27.65@E17 → 26.03@E18, both ***BEST), no OOM. Trajectory parallels N0010 which peaked at E26/40 — the run was killed before its LR-schedule sweet spot, so 26.03 is NOT a fair test of H0021 yet. RMSE/MAE = 3.66× already near the 3.63× refutation line though, and best-so-far trails parent 21.53 by +4.5 — direction is concerning but inconclusive.

## actionable_feedback
1. Do not refute H0021 on this truncated run; requeue or compare at matched-epoch (N0010 @E18 equivalent) before booking evidence.
2. Add a per-node MODIFICATIONS.md (even one line) for clone nodes.
3. Derive `ch` from cfg in model.py for future head-width experiments.
4. Consider gradient-accumulation or batch_size reduction to fit full 40ep within wall-clock at 518px (epoch time ~63s vs budget).

## hypothesis_updates
- hyp_id: H0021 | evidence_type: truncated_run_partial_signal | strength: weak_contra | reasoning: best 26.03 @E18 > parent 21.53 and RMSE/MAE 3.66 ≥ refutation threshold 3.63, but run stopped at 45% schedule while still monotonically improving; evidence is suggestive only — book as weak contradiction pending a complete run or matched-epoch comparison.
- hyp_id: H0017 | evidence_type: code_and_training_consistency | strength: neutral_support | reasoning: multi-layer gated taps train stably at 1369 tokens with gate still learning; no instability introduced by higher resolution, consistent with substrate robustness but no performance evidence yet.

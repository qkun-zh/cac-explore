# Qualitative Feedback — N0021_dino_partialft

## Code quality

- **Wrong module docstring** (model.py:1): claims "N0021_dino_fullft — FULL backbone
  fine-tuning", but this node is `partialft` and the code does PARTIAL FT. Copy-paste
  artifact from the sibling node; actively misleading to future readers/diffs. Fix first.
- **Missing idea.md**: node directory has no idea.md → falsification criteria were never
  pre-registered in-node (protocol Step 3 gap). Parent-bar (≤21.53) lives only in
  quantitative.md. Backfill before archiving.
- Naming: `DinoPartialFT`, `param_groups`, `backbone_lr_mult` are clear;
  `PromptEncoderV2` carries a meaningless version suffix; `t6_proj`/`t11_proj` are
  acceptable shorthand but `tap6_proj` would self-document; `ch` local is fine.
- Magic number `clamp(-13.8, 0.0)` (model.py:25) = log-area bounds [1e-6, 1] — deserves
  one comment or named constants. No method-level docstrings anywhere.
- Default drift: model defaults `dropout=0.15` vs config `0.1`; harmless here but a
  silent-divergence trap. Same pattern for `adapter_dim`.

## Hypothesis compliance

- Freeze mask (model.py:39-43) unfreezes EXACTLY `blocks.10.` / `blocks.11.` / `norm.` —
  matches the intended "last 2 blocks + final norm" scope precisely. String-prefix match
  is fragile (would also catch `blocks.100.` if depth grew) but correct for ViT-S/14.
- Differential LR implemented properly: `param_groups()` puts trainable backbone params at
  `base_lr * 0.1` (config `backbone_lr_mult=0.1`), non-backbone at base lr. Compliant.
- `result.json.diagnostics.smoke: false` — ambiguous field. If it means "smoke test not
  run", that violates Hard Rule 6 (smoke before real data). Needs disambiguation
  (`smoke_mode` vs `smoke_passed`) in the runner schema.
- Training itself was clean: 40/40 epochs, no OOM, no instability, best E25 20.438,
  monotone-ish loss decay 8.64→~4.4 mid-run.

## Design clarity

- Readable dataflow: two taps (blocks 6, 11) → learned softmax gate → prompt-conditioned
  adapter (bbox Fourier features prepended as one token) → 1x1-conv density head.
  Gating via `layer_logits` is elegant and inspectable post-hoc.
- Token-grid reshape handles timm's channels-first/3D ambiguity defensively (ndim check)
  — good robustness, mildly obscures intent.
- Frozen-param exclusion in `param_groups` prevents accidental weight-decay updates on
  dead params — subtle correctness point done right.

**Overall**: functional, well-scoped implementation of the hypothesis; documentation debt
(docstring lie + missing idea.md) is the main defect, none of it affects validity of results.

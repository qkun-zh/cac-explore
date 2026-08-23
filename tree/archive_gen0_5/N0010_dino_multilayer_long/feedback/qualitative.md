# Qualitative Feedback — N0010_dino_multilayer_long

## Naming Conventions Audit

- **Slug**: `N0010_dino_multilayer_long` — follows convention, descriptive of key change (multi-layer taps + 40ep schedule).
- **Class name**: `DinoPromptV2` — **CONFLICTS** with N0007's `DinoPromptV2`. Both files define the same class name. This works only because each node is imported in isolation by the engine; however, if two nodes were ever co-imported or the class name referenced in synthesis docs, ambiguity arises. Recommend future nodes use a unique class suffix (e.g. `DinoPromptV2Multitap`).
- **Registry path**: `build_model(cfg)` return `DinoPromptV2(cfg)` — standard contract met, but same class-name collision note applies.
- **Helper class**: `PromptEncoderV2` — duplicated verbatim from N0007. Acceptable for node isolation, but violates single-home rule if both exist in the same codebase simultaneously. Future extraction to a shared utility would be cleaner.
- **Constants**: `PATCH = 14` at module top, `BACKBONE = "vit_small_patch14_reg4_dinov2.lvd142m"` — clear and consistent with N0007.

## Docstring Presence & Self-Containedness

- Module docstring present (line 1): `"""N0010_dino_multilayer_long — champion recipe + mid/final layer-gated taps + 40ep + count-w 1.0."""` — good, self-contained summary of the node's delta.
- No class-level or method-level docstrings on `PromptEncoderV2`, `DinoPromptV2`, or `build_model`. The module docstring covers the "what" but not the "why" of architectural choices (e.g. why out_indices=(6,11), why softmax gate vs. learned weights). For a node with multiple architectural changes, inline comments or docstrings on `forward()` would help future readers.
- Inline comments absent in `model.py`. The code is readable but the layer-gate logic (lines 65–68) and the prompt-concat-then-slice pattern (line 70) would benefit from brief rationale notes.

## MODIFICATIONS.md / Notes Clarity

- No `MODIFICATIONS.md` or separate notes file exists in the node directory. Per PROTOCOL §2, this is not a required file, but given the multi-change nature of this node (layer taps + gate + schedule + loss weight), a changelog vs. parent would help synthesis agents disentangle which change drove the improvement. The `## Delta vs Parent` section in idea.md partially covers this but is brief.

## Hypothesis Format Compliance

- H0017 follows the `IF/THEN/BECAUSE/DISPROVED IF` format correctly. Scope (FSC147), mechanism (mid-layer correspondence + longer training + count pressure), and falsification criterion (MAE > 27.65) are all present.
- H0018–H0020 mentioned in the user prompt are not present in idea.md — only H0017 is documented. The idea.md file lists 4 hypotheses in the user's description but the actual file only contains H0017. This is a **documentation gap**: if the coding agent considered additional hypotheses, they should be recorded. As-is, only H0017 is falsifiable from the written record.
- Confidence initialization (0.5 default) applies; no deviations noted.

## Idea Clarity & Falsifiability

- The motivation is clear: compound three incremental improvements on the proven champion. The intuition about mid-layer tokens carrying richer correspondence structure is well-grounded (DINOv2 literature supports hierarchical feature specialization).
- The architecture spec is precise: out_indices=(6,11), softmax gate, prompt-after-fusion, adapter dims match parent.
- The falsification bar (MAE > 27.65 = no gain over parent) is appropriate and was met decisively (21.53 best).
- **Weakness**: The hypothesis bundles three changes (layer taps, 40ep, loss_count_weight=1.0) into a single claim. This makes it impossible to attribute the 6.12-point improvement to any specific change. A more rigorous design would have isolated at least one variable (e.g. same architecture, 40ep only) for attribution. The idea.md acknowledges this is "not novel per se" but the lack of ablation weakens the scientific value.

## Overall Assessment

| Dimension | Score | Notes |
|-----------|-------|-------|
| Naming | ⚠️ | Class-name collision with N0007; slug and constants fine |
| Docstrings | ⚠️ | Module docstring present; class/method docstrings missing |
| MODIFICATIONS | — | Not present; Delta section in idea.md partially compensates |
| Hypothesis format | ✅ | H0017 compliant; H0018–H0020 missing from file |
| Idea clarity | ✅ | Clear motivation, precise spec, appropriate falsification bar |
| Falsifiability | ✅ | Bar met decisively; bundling weakens attribution |

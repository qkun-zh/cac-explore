# Idea — N0024_ebc_partialft (parent: N0022_dino_ebc_fullft; dual selection Q_t=[H0032, H0031])

## Targeted change (1 of 1, PRIMARY): full FT → partial FT (blocks 10-11 + final norm) @ lr×0.1
Parent N0022 (EBC classification) unfroze ALL backbone params at lr_mult=0.05 → best 21.708@E23,
then val MAE drifted 21.7→27.4 while train loss kept falling (0.0058→0.0051) — the same
moving-target collapse as the density-paradigm full-FT sibling (N0021_fullft 48.4, E5 spike 93.9).
Full FT is refuted across two output heads; H0032 (c=0.575) is supported only under density
regression (champion 20.438@E25). This node tests the cross-paradigm transfer. Concrete diffs vs
parent — nothing else changes (taps 6+11 + scalar gate, Fourier+area prompt, adapter 384→768→384,
EBC head 16-bin CE, 40ep bs8 lr1e-3 cosine AMP all kept):
1. Freeze mask copied VERBATIM from champion (N0021 model.py:39-43): `requires_grad=True` only for
   backbone names matching `blocks.10.` / `blocks.11.` / `norm.`; all else frozen. Parent's
   param_groups() (model.py:67-79) already sends trainable backbone params to base_lr×mult.
2. config.py: `backbone_lr_mult` 0.05→0.1 (H0032's exact dose); `dropout` 0.15→0.1 — equalization
   mandated by N0021 synthesis §contradictions#1 (kills residual dropout confound vs champion).

## Pre-registered hypothesis (AGENTS.md §6 format)
**H0032 cross-paradigm transfer**: IF only backbone blocks 10-11 + final norm are unfrozen at
lr_mult=0.1 IN the EBC classification pipeline (parent recipe otherwise unchanged, dropout=0.1),
THEN best val MAE beats full-FT parent 21.708 and challenges density champion 20.44 BECAUSE
narrow-scope ×0.1 backbone drift restores timescale separation (drift rate < head fit rate)
regardless of whether the head regresses density or classifies count bins. DISPROVED IF ANY of:
- (a) best val MAE ≥ 21.708 (no gain over full-FT parent ⇒ H0032 does NOT transfer to EBC);
- (b) ep16 val MAE ≥ 22.80 (= N0021_partialft@E16 21.298 + 1.5; early-stop trigger, AGENTS.md Step5);
- (c) late-drift signature: any epoch's val MAE rises >2.0 above its own running best
      (parent hit +5.7 after E23; champion never exceeded +4.8 worst-epoch vs best).

Bar ladder: beat parent 21.708 ⇒ direction confirmed in-paradigm (weak support). Beat champion
20.44 ⇒ new champion (strong support, w≈0.85). Land in (20.44, 21.71) ⇒ partial transfer — EBC
keeps a paradigm handicap; book weak support with scope caveat noted for feedback.

## Why NO secondary change (deliberate single-variable design)
- TT-Norm at eval and jitter-only aug both live in SHARED code (engine `evaluate()` /
  data/fsc147.py whose `augment` flag is currently never enabled) — editing shared code violates
  the node-local targeted-change rule; both belong in dedicated follow-up nodes.
- H0031 (progressive upsampling decoder) was contradicted once (N0023: +17.5 MAE @E8 — extra
  output cells need more training than τ_max=30min allows) and maps poorly onto EBC's fixed
  28×28 block grid, which has the identical cell-count/training-budget failure surface.

## Grounding (docs/research_notes_2026-08-24.md)
- Backbone lr×0.1 is the de facto counting fine-tune recipe (MGCAC ACCV'24) — matches H0032 dose.
- CLIP-is-a-strong-fine-tuner (arXiv 2212.06138): top-half-layer FT ≈ full FT; explicitly
  recommends minimizing representation drift — timescale separation is the accepted mechanism.
- Full FT refuted twice across two paradigms/head families (N0021_fullft, N0022_ebc_fullft);
  bare full-FT retries are on STATE.md's refuted-lever list.

## Risks
- EBC vocab/class-imbalance × slower backbone: ~95% of 28×28 cells are bin 0, so CE gradients are
  background-dominated; with backbone adaptation at 0.1× lr, rare high-count bins (≥8) may remain
  underfit → plausibly lands BETWEEN the two bars instead of clearing them. Log per-bin confusion
  at synthesis if (a) triggers.
- num_bins=16 clamps dense-block targets (>16 objects/block) — inherited from parent, unchanged.
- dropout 0.15→0.1 is a second diff vs parent (confound control, not a treatment); attribution
  stays clean because the champion comparator ran dropout=0.1 identically.
- Single-seed noise σ≈±0.3 MAE (N0021 synthesis #2): gaps <0.5 are not evidence.

## Param-budget sanity (≤32M hard assert, engine counts ALL params incl. frozen)
DINOv2-S 22.05M + taps/prompt/adapter ≈0.99M + EBC conv head ≈0.134M (vs champion head 0.049M,
+0.085M) ≈ **23.2M total** — well under 32M. Trainable ≈ 3.6M top-block/norm + ≈1.0M head stack.

# STATE — Current Situation

**Stage**: MISSION REDEFINED — target changed from MAE<16 to **MAE≤4** (beyond published SOTA). Full backbone fine-tuning unlocked. Redesign required.
**Blockers**: none — server OK, GPU free

## Verified Facts (do not re-learn)
- Frozen-backbone ceiling: N0010 = 21.53 val / ~22 test (19 nodes, 9 refuted variants confirmed saturation)
- Best frozen recipe: DINOv2-S reg4 taps(6,11) gate + area-prompt + adapter768 + MLP head, 392px, 40ep
- SOTA reference (test): CountGD 5.74 · GeCo2 7.64 · DAVE 8.66 · CACViT 9.13 · LOCA 10.79
- MAE≤4 requires: full FT + large backbone + novel paradigm + possibly TTA/ensemble
- Engine supports huber loss; needs param-group optimizer for differential LR
- 27 hypotheses banked; subagent git hallucination + network failures logged in failure_modes.md
- τ_max currently 30min — may need relaxation for large-model full FT

## Next Steps (in order)
1. Update README/AGENTS/research_direction with new target ≤4 + relaxed constraints
2. Design gen-6: full-FT large backbone + paradigm innovation
3. Execute fast iteration loop

## Active Tasks
- N0021_dino_fullft created but superseded by new target — archive or evolve
EOF_MARKER_NOT_NEEDED

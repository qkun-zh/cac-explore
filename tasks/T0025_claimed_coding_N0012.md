# T0025 — pending coding N0012_dino_highres518

- status: pending
- created: 2026-08-23T12:50:00+08:00
- role: coding
- node: tree/nodes/N0012_dino_highres518
- parent: N0010_dino_multilayer_long (val MAE 21.53 champion)
- inputs: tree/nodes/N0012_dino_highres518/idea.md, tree/nodes/N0010_dino_multilayer_long/model.py, tree/nodes/N0010_dino_multilayer_long/config.py, memory/failure_modes.md
- outputs: tree/nodes/N0012_dino_highres518/model.py (build_model), tree/nodes/N0012_dino_highres518/config.py; tree.json status→coded after smoke
- hypotheses: H0021 (resolution 392→518 → MAE≤19.0, RMSE/MAE<3.4) + H0017 reuse

## Notes
Identical architecture to N0010 — frozen vit_small_patch14_reg4_dinov2 dynamic_img_size=True features_only out_indices=(6,11), scalar layer_logits gate, Fourier prompt (freqs=8), adapter 384→768→384 drop0.1, conv head 384→128→1. Only cfg.input_size=518 (from 392); 518/14=37 → 1369 tokens vs 784 (+75%). Keep epochs=40, lr=1e-3, count_w=1.0, amp True. Param est ~23.11M (no increase). Copy N0010 model.py verbatim; only config changes. Must pass --smoke (2ep synthetic) before executor.

## Falsifiable bars (from idea.md)
- DISPROVED IF MAE >21.53 (no gain over parent) OR RMSE/MAE ≥3.63; target MAE ≤19.0 and ratio <3.4
- Do NOT build on N0011 (trending worse 27.0@E24) — clone N0010 directly

# T0029 — pending coding N0014_dino_highres_augreg

- status: pending
- created: 2026-08-23T14:06:15+08:00
- role: coding
- node: tree/nodes/N0014_dino_highres_augreg
- parent: N0010_dino_multilayer_long (val MAE 21.53 champion, frozen DINOv2-S reg4 dual taps scalar gate, 392px, 40ep, 23.11M)
- inputs: tree/nodes/N0014_dino_highres_augreg/idea.md, tree/nodes/N0010_dino_multilayer_long/model.py, tree/nodes/N0010_dino_multilayer_long/config.py, tree/nodes/N0013_dino_mosaic/model.py, tree/nodes/N0013_dino_mosaic/config.py, tree/nodes/N0012_dino_highres518/config.py, memory/failure_modes.md, memory/index.json
- outputs: tree/nodes/N0014_dino_highres_augreg/model.py (build_model), tree/nodes/N0014_dino_highres_augreg/config.py; tree.json status→coded after smoke
- hypotheses: H0023 (highres+augreg additive → MAE≤18.0, ≤min(N0012,N0013), RMSE/MAE<3.4) + H0021/H0022 reuse + H0017 reuse

## Notes
Merge-winners child of N0010 combining N0012 (518px) + N0013 (augreg) orthogonal levers — valid before siblings finish. Clone N0010 verbatim. ONLY changes: (a) config: input_size 392→518 (37×37, 1369 tokens +75%), batch_size 8→4 (518 OOM safety), dropout 0.1→0.2, weight_decay 1e-4→5e-4, keep epochs 40, lr1e-3, count_w1.0, amp True, adapter_dim 768; aug params jitter_prob 0.5, jitter_brightness 0.2, jitter_contrast 0.2, jitter_saturation 0.15, jitter_noise_std 0.02, bbox_jitter 0.15. (b) model.py: start from N0013 model.py (already has photometric+bbox jitter, training-gated, clamp-safe) — verify it handles ps=S//PATCH at 518 (37) via flatten(2).transpose/f6.ndim check, PATCH=14, dynamic_img_size=True, BCHW, head produces [B,1,37,37]. No N0011 Huber/per-token. Expected ~23.11M (identical, 0 extra). Must pass `python code/engine/train.py --node_dir tree/nodes/N0014_dino_highres_augreg --smoke --epochs 2` synthetic before executor. Do NOT commit/push — Lead owns git.

## Falsifiable bars (from idea.md)
- DISPROVED IF MAE >18.0 (champion -3.5 bar miss) OR MAE > min(N0012 best, N0013 best) (no additive gain) OR RMSE/MAE ≥3.63 (no tail improvement); target MAE ≤18.0 AND ≤best sibling AND RMSE/MAE <3.4
- H0023 supported if both levers stack (better than either alone); H0021/H0022 each supported if respective lever evidence persists at combined setting
- Do NOT build on N0011 (26.68 REFUTED) — clone N0010 directly; N0012 ep5 29.43 early, N0013 smoke green — keep pipeline full per never-idle.

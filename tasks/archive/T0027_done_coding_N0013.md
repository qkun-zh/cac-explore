# T0027 — pending coding N0013_dino_mosaic

- status: pending
- created: 2026-08-23T13:00:00+08:00
- role: coding
- node: tree/nodes/N0013_dino_mosaic
- parent: N0010_dino_multilayer_long (val MAE 21.53 champion, frozen DINOv2-S reg4 dual taps scalar gate, 392px, 40ep, 23.11M)
- inputs: tree/nodes/N0013_dino_mosaic/idea.md, tree/nodes/N0010_dino_multilayer_long/model.py, tree/nodes/N0010_dino_multilayer_long/config.py, memory/failure_modes.md
- outputs: tree/nodes/N0013_dino_mosaic/model.py (build_model), tree/nodes/N0013_dino_mosaic/config.py; tree.json status→coded after smoke
- hypotheses: H0022 (augreg mosaic-lite → MAE≤20.0, overfit gap reduced, RMSE/MAE<3.4) + H0017 reuse

## Notes
Clone N0010 verbatim. Only changes: (a) model.py forward adds training-gated photometric jitter (ColorJitter brightness0.2/contrast0.2/sat0.15 + Gaussian σ0.02, p0.5) on imgs tensor + bbox jitter ±15% (uniform scale/translate, clamp [0,S]) before prompt_enc — guard with `if self.training`; no GT density warp needed. (b) config.py: dropout 0.1→0.2, weight_decay 1e-4→5e-4; keep input_size=392, epochs=40, lr1e-3, count_w=1.0, amp True. DO NOT copy N0011 Huber/per-token. Must stay ≤32M (~23.11M, 0 extra). Orthogonal to N0012 (518px). Must pass `--smoke --epochs 2` (synthetic) before executor. See idea.md falsifiable bars.

## Falsifiable bars (from idea.md)
- DISPROVED IF MAE >21.53 (no gain over parent) OR best epoch ≤26 with same overfit divergence OR RMSE/MAE ≥3.63; target MAE ≤20.0, RMSE/MAE <3.4, best epoch >26
- Do NOT build on N0011 (regressing 27.0) — clone N0010 directly

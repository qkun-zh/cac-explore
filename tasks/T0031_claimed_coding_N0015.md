# T0031 — pending coding N0015_dino_highres672

- status: pending
- created: 2026-08-23T14:12:00+08:00
- role: coding
- node: tree/nodes/N0015_dino_highres672
- parent: N0010_dino_multilayer_long (val MAE 21.53 champion, frozen DINOv2-S reg4 dual taps scalar gate, 392px, 40ep, 23.11M)
- inputs: tree/nodes/N0015_dino_highres672/idea.md, tree/nodes/N0010_dino_multilayer_long/model.py, tree/nodes/N0010_dino_multilayer_long/config.py, tree/nodes/N0012_dino_highres518/config.py, memory/failure_modes.md, memory/index.json
- outputs: tree/nodes/N0015_dino_highres672/model.py (build_model), tree/nodes/N0015_dino_highres672/config.py; tree.json status→coded after smoke
- hypotheses: H0024 (392→672 → MAE≤18.5, RMSE/MAE<3.4, DISPROVED if MAE>21.53 or OOM or ratio≥3.63) + H0017 reuse

## Notes
Contingency branch of N0010 isolating extreme resolution lever. Clone N0010 verbatim. ONLY changes: (a) config: input_size 392→672 (28→48 patches/side, 784→2304 tokens, +194% vs 392; +68% vs 518), batch_size 8→2 for 12GB OOM safety (672 needs ~2.9× memory of 392; tokens 4608 per batch at bs2 vs 6272 at 392×bs8, similar footprint), keep epochs 40, lr1e-3, wd1e-4, eta_min1e-5, amp True, adapter_dim 768, dropout 0.1, loss_count_weight 1.0, num_workers 4, max_params_M 32. Do NOT add augreg (wd5e-4/drop0.2/jitter) — keep wd1e-4 drop0.1 like champion to isolate resolution vs N0013/N0014. (b) model.py: copy N0010 model.py verbatim; ensure `timm.create_model(..., dynamic_img_size=True)` (672%14==0 required, native 518), PATCH=14, BCHW, head outputs [B,1,48,48] via flatten(2).transpose/f6.ndim checks, Linear on tokens only — read memory/failure_modes.md before coding (BCHW memory, PATCH const, Linear-on-tokens traps). Expected ~23.11M total (0 extra params, resolution adds activations only). Must pass `python code/engine/train.py --node_dir tree/nodes/N0015_dino_highres672 --smoke --epochs 2` synthetic before executor. Fallback if OOM: reduce further to bs1.

## Falsifiable bars (from idea.md)
- H0024 SUPPORTED IF MAE ≤18.5 AND RMSE/MAE <3.4; DISPROVED IF MAE >21.53 (no gain over N0010) OR OOM/timeout at batch2 OR RMSE/MAE ≥3.63 (no tail improvement vs parent 3.63×).
- If 518 (N0012) succeeds, 672 should improve further ceiling test; if 518 fails/OOM, 672 provides scaling-limit data point. Do NOT build on N0011 (26.68 REFUTED) — clone N0010 directly. Keep pipeline full per never-idle (N0012 running ep11 best28.7 still early, N0013/N0014 queued).

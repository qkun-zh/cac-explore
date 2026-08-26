# Archived session block — 2026-08-25 evening (from STATE.md, archived at session start 2026-08-26)

**Mode**: 用户指导模式 (User-Guided)
**Champion ckpt**: CAC-D simplified (cnt+density only) — best MAE **19.15** (no-OT ablation, Ep16)
**GPU**: RTX3060 idle · creds `local/address_and_password.md`

## Loss Ablation Results (FSC147 val, cached 56×56, 40ep)

| Experiment | Best MAE | Δ |
|---|---|---|
| Baseline (den+cnt+sim+ot) | 19.65 | — |
| No OT | **19.15** | -0.5 |
| No Sim | **19.22** | -0.4 |
| No Density | 21.30 | +1.7 |
| Count Only | 20.90 | +1.3 |
| No Count | 47.09 | 废了 |

## Conclusions
1. Count loss = sole core (MAE 20.9 alone)
2. OT = useless (removing improves)
3. Sim ≈ useless (removing barely changes)
4. Density = weak help (+1.7 if removed)
5. Architecture simplified: backbone → fuser → condenser → density decoder; loss = MSE(density) + smooth_L1(log(count))

## Next
1. Train simplified model (no sim, no OT) for clean baseline
2. Try higher cnt_weight or different count loss formulations
3. Consider backbone fine-tuning for further improvement

## Rules
- Probe scripts live in /data/asset/r0i_probe (server) + /tmp/opencode (local)
- Gates unchanged: novelty_check → check_hypothesis → calibration_report

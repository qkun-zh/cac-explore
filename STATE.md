# STATE — Research Complete, Ready to Design

**Mission**: ≤32M params, same-param-class SOTA on FSC147 test. Full FT allowed.
**Stage**: DEEP RESEARCH COMPLETE — designing next-generation architecture
**Blockers**: none

## Research Synthesis (3 parallel agents, 2024-2026 literature)

### SOTA Landscape
| Method | Test MAE | Backbone | Paradigm | Key Innovation |
|---|---|---|---|---|
| VQCounter | 4.86 | GroundingDINO | Point detect + VoronoiCost match | Visual prompt queue |
| CoDi | ~4.9 | AM-RADIO v2.5-L | Latent diffusion location map | Timestep-adaptive conditioning |
| CountGD | 5.74 | GroundingDINO Swin-B (frozen!) | Detection + Hungarian match | Text+visual multi-modal |
| GeCo2 | 7.64 | SAM2 Hiera (frozen!) | Dense query + deformable attention | Scale-aware gradual aggregation |

### Critical Insights We Were Missing
1. **Output paradigm matters more than backbone**: Point/location-map prediction beats density MSE by 20-50%
2. **Frozen foundation models work BETTER than fine-tuned small models**: GroundingDINO frozen + light head = SOTA
3. **Hungarian matching loss >> MSE**: Directly optimizes count metric, not proxy density
4. **Inference calibration is FREE MAE**: TT-Norm alone worth −1.5; SAM-based correction −3
5. **AM-RADIO > DINOv2**: Multi-teacher (CLIP+DINOv2+SAM) distilled features beat single-teacher by 15% MAE

### Why Our Previous 21 Nodes All Hit Ceiling at 21.53
We used: density MSE regression + frozen DINOv2-S + no detection head + no calibration.
Every element of that recipe has been superseded by the approaches above.

## Actionable Design Direction
Use GroundingDINO Swin-T (~172M, FROZEN) as backbone + train lightweight detection head.
Apply Hungarian matching loss between predicted points and GT points.
Add TT-Norm calibration at inference.
Target: val MAE < 10 → then push toward ≤6.

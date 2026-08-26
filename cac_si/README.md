# cac_si — Scale-Invariant Prompt-Aware Counter (SI-INR line)

Design per docs/Xu_Scale-Invariant_Implicit_Neural_Representation.md + user-approved architecture (see docs/si_inr_prompt_counter.png).

## Hypothesis
IF [scale-invariant encoding (B_H multi-scale mean + S canonical grid) on both streams, INR continuous decoding, sampling loss]
IN [frozen DINOv3 dual-stream counter],
THEN [val MAE improves over cac_d baseline at equal resolution, and degrades less under test-time resolution shift],
BECAUSE [scale invariance is injected as structural prior; continuous density removes fixed-grid information loss].
DISPROVED IF [val MAE @224 not better than cac_d q_mse line (20.41) after 32ep].

## Architecture
- Image path: Resize{0.75,1.0,1.25} -> frozen DINOv3 (3 forwards) -> B_H mean -> S(.) to H0xW0 -> a'
- Prompt path: K=3 crops (margin 0.25) -> frozen DINOv3 (single scale) -> S(.) -> b'
- Cross-Attention (Q=a', K/V=b', trainable Condenser) -> c=[a'||cond]
- INR decoder: u(x)=softplus(MLP([z_x, fourier(x), x])), 4 Linear + residual
- Loss: sampling L2 vs continuous Gaussian-mixture GT (Eq.7/10) + log-count smooth-L1 (quadrature)

## Notes
- Backbone frozen; trainable = Condenser + INR (~1M). Total incl backbone ~30M <= 32M budget.
- No feature cache (multi-scale online forward). Fast screening at image_size=224.
- S(.) is identity-by-construction at fixed square input; kept explicit for variable-size inference.

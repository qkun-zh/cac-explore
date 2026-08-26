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

## Paper deviations (documented)
- No Equi-Tuning stage (Eq.3): backbone stays frozen per mission constraint; B_H is a
  multi-scale ensemble mean, not a tuned equivariant map.
- Dual-stream prompt + cross-attention is OUR extension (paper has no exemplar branch).
- lr 1e-3 vs paper 1e-4 (32-epoch budget vs paper 300; Adam is loss-scale invariant).
- GT sigma 0.02 normalized (~4.5px@224) vs paper 8/15px on larger RSOC images.
- Multi-scale sizes snapped to /16 multiples (0.714/1.0/1.286 effective at 224) for
  exact B_H alignment (grid flooring otherwise drops edges).
- INR decoder now EXACT per Eq.9: input z_x only, 4 residual FC + 1 output FC (5
  Linears), raw output, init N(0,0.01^2). Hidden width/activation (128/GELU) are our
  choice (paper unspecified). Loss Eq.10 exact; GT via discrete-map interpolation
  (paper §3.4 operational definition); count = integral(u)*S^2 (self-consistent with
  the discrete-map convention).

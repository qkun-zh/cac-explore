# Probe — dual-channel rho+/rho- (10ep dynamics check)
Parent: N0027_norm_flip_swa. Only delta: head 384->128->2 (rho+ / rho-), loss = MSE(rho+,gt)+0.3*BCE(rho-,bg) + 0.5*L1(sum(rho+ -0.2*rho-), N). Euler logged only. Purpose: watch rho+/rho- separation dynamics, not to beat SOTA.

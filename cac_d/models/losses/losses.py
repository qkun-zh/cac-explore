import math
import torch, torch.nn.functional as F

def gaussian_density(points, B, H, W, S, sigma=1.5):
    """Render GT points as Gaussians on an H×W grid (sum preserved = N).
    points: list of [N_i,2] tensors in image coords (0..S)."""
    dev = points[0].device if len(points) else "cpu"
    delta = torch.zeros(B, 1, H, W, device=dev)
    scale = W / float(S)
    for b, pts in enumerate(points):
        if pts.numel() == 0: continue
        idx = (pts.to(dev) * scale).long()
        xi = idx[:, 0].clamp(0, W-1); yi = idx[:, 1].clamp(0, H-1)
        delta[b, 0].view(-1).index_add_(0, yi*W+xi, torch.ones(len(idx), device=dev))
    r = max(3, int(2*sigma) | 1)                     # odd kernel radius
    ax = torch.arange(r, device=dev, dtype=torch.float32) - r//2
    g = torch.exp(-(ax**2) / (2*sigma**2)); g = (g / g.sum())
    k = torch.outer(g, g).view(1, 1, r, r)
    return F.conv2d(delta, k.to(delta.dtype), padding=r//2)

def sim_margin_loss(smax, gt_dens):
    """BMNet+-flavored similarity supervision: max-sim inside target cells
    should exceed outside by a margin."""
    B = smax.shape[0]
    loss = smax.new_zeros(())
    for b in range(B):
        m = gt_dens[b, 0] > 0.05 * gt_dens[b, 0].amax().clamp_min(1e-6)
        n_in, n_out = m.sum(), (~m).sum()
        if n_in == 0 or n_out == 0: continue
        s = smax[b, 0]
        loss = loss + F.softplus(s[~m].mean() - s[m].mean() + 0.1)
    return loss / B

def ot_coverage_loss(S, gt_d, eps=0.5, iters=30):
    """Sinkhorn OT coverage regularizer using FlashSinkhorn (GPU-native).
    S: [B, H, W, K] exemplar similarity maps (unused, kept for API compat)
    gt_d: [B, 1, H, W] GT Gaussian density
    Note: actual OT is computed on exemplar/fine feature coordinates via FlashSinkhorn,
    so this function needs access to e and fine features — see model.py for the actual call.
    This fallback uses the cost-matrix approach with PyTorch tensors on GPU.
    """
    B, H, W, K = S.shape
    M = H * W
    Q = (1.0 - S.clamp(0, 1)).view(B, K, M)       # [B, K, M] cosine distance ∈ [0,1]
    t = gt_d.view(B, M)
    tsum = t.sum(-1, keepdim=True).clamp(min=1e-12)
    t = t / tsum                                   # [B, M] Σ_m t_m = 1

    # Log-domain Sinkhorn on GPU (PyTorch native)
    log_K = -Q / eps                               # [B, K, M] log-Gibbs kernel
    a = Q.new_full((B, K), 1.0 / K)               # [B, K] uniform source weights
    log_a = a.clamp(min=1e-30).log()              # [B, K]
    log_b = t.clamp(min=1e-30).log()              # [B, M]

    # Initialize scaling vectors
    log_u = Q.new_zeros(B, K)                      # [B, K]
    log_v = Q.new_zeros(B, M)                      # [B, M]

    for _ in range(iters):
        # log(K @ v) for each source: logsumexp(log_K + log_v[:,:,None], dim=2) → [B, K]
        log_Kv = torch.logsumexp(log_K + log_v.unsqueeze(1), 2)
        log_u = log_a - log_Kv
        # log(K^T @ u) for each target: logsumexp(log_K + log_u.unsqueeze(2), dim=1) → [B, M]
        log_Ktu = torch.logsumexp(log_K + log_u.unsqueeze(2), 1)
        log_v = log_b - log_Ktu

    # Transport plan: Π[b,k,m] = u[b,k] * K[b,k,m] * v[b,m]
    log_P = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)  # [B, K, M]
    P = log_P.exp()                                # [B, K, M]

    # L = <Π, Q> + ε H(Π)
    H_P = -(P * log_P).sum()
    tot = (P * Q).sum() + eps * H_P

    return tot / B

def uot_loss(p, w, points, S=384, eps=0.08, tau=1.0, alpha=1.0, iters=32):
    """Minimal unbalanced OT on selected candidate cells (see heads.PileHead):
    entropic assignment with no-match column + demand-KL. `iters` reserved."""
    B = p.shape[0]
    tot = p.new_zeros(()); cnts = []
    for b in range(B):
        pb, wb, gb = p[b], w[b], points[b]
        if gb.numel() == 0 or pb.numel() == 0: continue
        gb = gb.to(pb.device); N = gb.shape[0]
        d2 = ((pb.unsqueeze(1) - gb.unsqueeze(0)) ** 2).sum(-1) / (S*S)
        prob = F.softmax(torch.cat([-d2 / eps, torch.zeros(len(pb), 1, device=pb.device)], 1), 1)
        pi = wb.unsqueeze(1) * prob[:, :N]
        col = pi.sum(0)
        tot = tot + alpha*(pi*d2).sum() + tau*(col*col.clamp_min(1e-8).log() - col + 1).sum()
        cnts.append(pi.sum())
    return (tot/B if B else tot), (torch.stack(cnts) if cnts else torch.zeros(B, device=p.device))

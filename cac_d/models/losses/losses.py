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

def ot_coverage_loss(S, gt_d, eps=0.1, iters=10):
    """Sinkhorn OT coverage regularizer (inspired by Proto4DME §3.3):
    balanced assignment of GT density mass K exemplars, minimizing
    expected cosine distance + entropy regularization.
    S: [B, H, W, K] exemplar similarity maps (SimModule output)
    gt_d: [B, 1, H, W] GT Gaussian density
    """
    B, H, W, K = S.shape
    M = H * W
    Q = (1.0 - S).view(B, K, M)                   # [B, K, M] cosine distance ∈ [0,1]
    t = gt_d.view(B, M)
    tsum = t.sum(-1, keepdim=True).clamp(min=1e-12)
    t = t / tsum                                   # [B, M] Σ_m t_m = 1

    INVALID = -1e4
    log_a = math.log(1.0 / K)                      # uniform row marginal

    tot = S.new_zeros(())
    for b in range(B):
        if t[b].sum() < 1e-12:
            continue                               # skip empty images
        Qb = Q[b]                                  # [K, M]
        tb = t[b]                                  # [M]

        # log-domain Sinkhorn: column then row normalization, T iterations
        log_P = (log_a + tb.clamp(min=1e-12).log()).unsqueeze(0).expand(K, -1).clone()  # [K, M]
        for _ in range(iters):
            # column norm: Σ_k P[k,m] = t[m]
            log_P = log_P - torch.logsumexp(log_P, 0, keepdim=True) + tb.clamp(min=1e-12).log()
            # row norm: Σ_m P[k,m] = a[k] = 1/K
            log_P = log_P - torch.logsumexp(log_P, 1, keepdim=True) + log_a

        # transport plan (differentiable via logsumexp)
        log_P = log_P.clamp(min=INVALID)
        P = log_P.exp()                            # [K, M]

        # L_ot = <P, Q> + ε H(P),  H(P) = -Σ P log P
        H_P = -(P * log_P.clamp(min=INVALID)).sum()
        tot = tot + (P * Qb).sum() + eps * H_P

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

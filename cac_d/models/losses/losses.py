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

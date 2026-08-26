import torch, torch.nn.functional as F


def _knn_sigma_grid(pts, W, S, k, smin, smax, beta):
    """Per-point sigma in GRID units from mean distance to k nearest neighbors.
    pts: [N,2] image coords 0..S. Singletons fall back to smin."""
    N = pts.shape[0]
    if N <= 1:
        return torch.full((N,), smin, device=pts.device)
    d = torch.cdist(pts, pts)
    d.fill_diagonal_(float("inf"))
    dbar = d.topk(min(k, N - 1), largest=False).values.mean(1)      # px
    return (beta * dbar * (W / float(S))).clamp(smin, smax)


def _gauss_kernel(sigma, device):
    r = max(3, int(2 * float(sigma)) | 1)      # same truncation rule as baseline
    ax = torch.arange(r, device=device, dtype=torch.float32) - r // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return torch.outer(g, g).view(1, 1, r, r)


def _scatter_delta(pts, H, W, S):
    delta = torch.zeros(1, 1, H, W, device=pts.device)
    idx = (pts * (W / float(S))).long()
    xi = idx[:, 0].clamp(0, W - 1); yi = idx[:, 1].clamp(0, H - 1)
    delta.view(-1).index_add_(0, yi * W + xi, torch.ones(len(idx), device=pts.device))
    return delta


def gaussian_density(points, B, H, W, S, sigma=1.5):
    """Fixed-sigma GT density, sum preserved (= N). Baseline."""
    dev = points[0].device if len(points) else "cpu"
    delta = torch.zeros(B, 1, H, W, device=dev)
    for b, pts in enumerate(points):
        if pts.numel():
            delta[b:b+1] = _scatter_delta(pts.to(dev), H, W, S)
    kk = _gauss_kernel(sigma, dev)
    return F.conv2d(delta, kk, padding=kk.shape[-1] // 2)


def adaptive_gaussian_density(points, B, H, W, S, k=3, smin=1.0, smax=8.0,
                              beta=1.0):
    """Per-IMAGE adaptive sigma (mean per-point kNN sigma), sum preserved.
    Drop-in replacement for gaussian_density."""
    dev = points[0].device if len(points) else "cpu"
    out = torch.zeros(B, 1, H, W, device=dev)
    for b, pts in enumerate(points):
        if pts.numel() == 0: continue
        sig = float(_knn_sigma_grid(pts, W, S, k, smin, smax, beta).mean())
        kk = _gauss_kernel(sig, dev)
        out[b:b+1] = F.conv2d(_scatter_delta(pts, H, W, S), kk, padding=kk.shape[-1] // 2)
    return out


def bayesian_density_targets(points, B, H, W, S, k=3, smin=1.0, smax=8.0,
                             beta=1.0, nbins=10):
    """Per-POINT adaptive sigma targets (BL ICCV'19 GT construction), mass=N.
    Points grouped into sigma bins -> one conv per bin per image."""
    dev = points[0].device if len(points) else "cpu"
    out = torch.zeros(B, 1, H, W, device=dev)
    edges = torch.linspace(smin, smax, nbins + 1, device=dev)
    for b, pts in enumerate(points):
        if pts.numel() == 0: continue
        sig = _knn_sigma_grid(pts, W, S, k, smin, smax, beta)
        idx = torch.clamp(torch.bucketize(sig, edges) - 1, 0, nbins - 1)
        for bi in idx.unique():
            m = idx == bi
            kk = _gauss_kernel(float(sig[m].mean()), dev)
            out[b:b+1] += F.conv2d(_scatter_delta(pts[m], H, W, S), kk,
                                   padding=kk.shape[-1] // 2)
    return out


def bayesian_density_loss(dens, points, H, W, S, **kw):
    """L1 between normalized prediction and normalized BL targets.
    Shape-only supervision (scale-free); magnitude comes from the count loss."""
    q = bayesian_density_targets(points, dens.shape[0], H, W, S, **kw).flatten(1)
    p = dens.flatten(1).clamp_min(0)
    p = p / p.sum(1, keepdim=True).clamp_min(1e-6)
    q = q / q.sum(1, keepdim=True).clamp_min(1e-6)
    return (p - q).abs().sum(1).mean()

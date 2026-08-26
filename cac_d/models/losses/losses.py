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

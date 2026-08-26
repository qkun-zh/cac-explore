"""INR decoder + continuous-density supervision utilities (SI-INR Eq.7/9/10)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier(x, freqs):                               # x [M,2] -> [M,4*len(freqs)]
    outs = []
    for f in freqs:
        outs += [torch.sin(2 * math.pi * f * x), torch.cos(2 * math.pi * f * x)]
    return torch.cat(outs, -1)


class INRDecoder(nn.Module):
    """u(x) = softplus(MLP([z_x, fourier(x), x])). 4 linear layers, residual on middle."""
    def __init__(self, z_dim, hidden=128, layers=4, freqs=(1, 2, 4, 8)):
        super().__init__()
        self.freqs = tuple(freqs)
        in_dim = z_dim + 2 + 4 * len(self.freqs)
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers - 2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers - 2)])
        self.out = nn.Linear(hidden, 1)
        for m in [self.inp, *self.blocks, self.out]:
            nn.init.normal_(m.weight, 0.0, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, z_x, x):                        # z_x [M,D], x [M,2] in [0,1]^2
        h = torch.cat([z_x, fourier(x, self.freqs), x], -1)
        h = F.gelu(self.inp(h))
        for lin, ln in zip(self.blocks, self.norms):
            h = F.gelu(ln(h + lin(h)))
        return F.softplus(self.out(h)).squeeze(-1)    # [M] >= 0


def gt_density_at(points, S, x, sigma):
    """Continuous GT (Eq.7): D_gt(x) = sum_i N(x; m_i, sigma^2), m in [0,1]^2.
    points: list of [N_i,2] in S-coords; x: [M,2] normalized. Returns [B,M]."""
    B = len(points)
    M = x.shape[0]
    out = x.new_zeros(B, M)
    two_sig2 = 2.0 * sigma * sigma
    norm = 1.0 / (2.0 * math.pi * sigma * sigma)
    for b, pts in enumerate(points):
        if pts.numel() == 0:
            continue
        m = pts / float(S)                            # [N,2]
        d2 = ((x.unsqueeze(0) - m.unsqueeze(1)) ** 2).sum(-1)   # [N,M]
        out[b] = (torch.exp(-d2 / two_sig2) * norm).sum(0)
    return out


def sample_map(cmap, xs):
    """Bilinear-sample feature map [B,D,H,W] at normalized coords xs [M,2] -> [B,M,D].
    border padding: x=0/1 edges replicate instead of zeroing features."""
    B = cmap.shape[0]
    M = xs.shape[0]
    grid = (xs.view(1, 1, M, 2) * 2.0 - 1.0).expand(B, 1, M, 2)   # (x,y)=(col,row)
    out = F.grid_sample(cmap, grid, mode="bilinear",
                        padding_mode="border", align_corners=False)  # [B,D,1,M]
    return out.squeeze(2).transpose(1, 2)             # [B,M,D]

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
    """Eq.9 exact: u(x) = phi_L(W_L(...phi_1(W_1 z_x + b_1)...) + b_L).
    Input is z_x ONLY (spatial variation comes from the sampled feature).
    4 FC layers with residual + additional output FC = 5 Linears (paper §4.2).
    Raw output (no softplus). Init N(0, 0.01^2) (paper §4.2). No LayerNorm."""
    def __init__(self, z_dim, hidden=128, layers=4):
        super().__init__()
        self.inp = nn.Linear(z_dim, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers - 1)])
        self.out = nn.Linear(hidden, 1)
        for m in [self.inp, *self.blocks, self.out]:
            nn.init.normal_(m.weight, 0.0, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, z_x):                          # z_x [M,D]
        h = F.gelu(self.inp(z_x))
        for lin in self.blocks:
            h = F.gelu(h + lin(h))
        return self.out(h).squeeze(-1)               # [M], raw density


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
    """Bilinear-sample feature map [B,D,H,W] at normalized coords.
    xs: [M,2] shared across batch, or [B,M,2] per-image. Returns [B,M,D].
    border padding: x=0/1 edges replicate instead of zeroing features."""
    B = cmap.shape[0]
    if xs.dim() == 2:
        xs = xs.unsqueeze(0).expand(B, -1, -1)
    M = xs.shape[1]
    grid = (xs * 2.0 - 1.0).reshape(B, 1, M, 2)      # (x,y)=(col,row)
    out = F.grid_sample(cmap, grid, mode="bilinear",
                        padding_mode="border", align_corners=False)  # [B,D,1,M]
    return out.squeeze(2).transpose(1, 2)             # [B,M,D]

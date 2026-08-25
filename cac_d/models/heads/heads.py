import torch, torch.nn as nn, torch.nn.functional as F

def grid_centers(S, patch=16):
    g=S//patch
    ys=(torch.arange(g)+0.5)*patch; xs=(torch.arange(g)+0.5)*patch
    yy,xx=torch.meshgrid(ys,xs,indexing="ij")
    return torch.stack([xx,yy],-1).reshape(-1,2), g

class FineFuser(nn.Module):
    """Multi-scale fuse: hs3@24 upsampled + hs2 lateral @48 -> refined map @96."""
    def __init__(self, ch_coarse, ch_mid, d_fine=128):
        super().__init__()
        self.top = nn.Sequential(nn.Conv2d(ch_coarse, d_fine, 1), nn.GroupNorm(8, d_fine))
        self.lat = nn.Sequential(nn.Conv2d(ch_mid, d_fine, 1), nn.GroupNorm(8, d_fine))
        self.fuse = nn.Sequential(nn.Conv2d(2*d_fine, d_fine, 3, padding=1),
                                  nn.GroupNorm(8, d_fine), nn.GELU())
        self.refine = nn.Conv2d(d_fine, d_fine, 3, padding=1, groups=d_fine)
    def forward(self, h2, h3):                       # [B,c2,48,48], [B,c3,24,24]
        top = F.interpolate(self.top(h3), scale_factor=2, mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([self.lat(h2), top], 1))
        f = F.gelu(self.refine(f) + f)
        return F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)

class SimModule(nn.Module):
    """Explicit exemplar<->cell correlation (BMNet+/SAFECount style):
    cosine sim maps per exemplar with learnable temperature; max/mean stats."""
    def __init__(self, d_fine=128, d_sim=256, tau0=0.07):
        super().__init__()
        self.proj = nn.Linear(d_fine, d_sim)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau0))))
    def forward(self, fmap, e):                      # fmap [B,H,W,d], e [B,K,D]
        t = F.normalize(self.proj(fmap), dim=-1)     # [B,H,W,D_sim]
        en = F.normalize(e, dim=-1)
        S = torch.einsum('bhwd,bkd->bhwk', t, en) / self.log_tau.exp()  # [B,H,W,K]
        return S, S.amax(-1).unsqueeze(1), S.mean(-1).unsqueeze(1), t.flatten(1, 2)

class Condenser(nn.Module):
    """Exemplar->cell cross-attention: each cell queries the exemplar bank
    (prompt-aware features without touching the frozen backbone) -> 64ch map."""
    def __init__(self, d_sim=256, n_heads=4, ff=512, d_out=64):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_sim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_sim); self.norm2 = nn.LayerNorm(d_sim)
        self.ffn = nn.Sequential(nn.Linear(d_sim, ff), nn.GELU(), nn.Linear(ff, d_sim))
        self.out = nn.Linear(d_sim, d_out)
    def forward(self, tok, e):                       # tok [B,M,D], e [B,K,D]
        a, _ = self.attn(self.norm1(tok), e, e, need_weights=False)
        q = self.norm1(tok + a)
        return self.out(self.norm2(q + self.ffn(q)))                     # [B,M,d_out]

class DensityDecoder(nn.Module):
    """Conv decoder at fine grid -> per-cell density (softplus, sum=count)."""
    def __init__(self, in_ch, hidden=128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden//2, 3, padding=2, dilation=2),
            nn.GroupNorm(4, hidden//2), nn.GELU())
        self.head = nn.Conv2d(hidden//2, 1, 1)
        for m in [self.block[0], self.block[3]]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(self.head.bias)
    def forward(self, x): return F.softplus(self.head(self.block(x)))

class PileHead(nn.Module):
    """Aux UOT branch: masses+offsets per cell on [tok ‖ simstats ‖ gap];
    loss applied on top-K cells only. Offsets bounded to half a fine cell."""
    def __init__(self, d_fine=128, hidden=128):
        super().__init__()
        self.mlp_w = nn.Sequential(nn.Linear(d_fine+3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.mlp_p = nn.Sequential(nn.Linear(d_fine+3, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp_p[-1].weight); nn.init.zeros_(self.mlp_p[-1].bias)
        nn.init.constant_(self.mlp_w[-1].bias, -3.5)
    def forward(self, tok, sstats, centers, cell):   # tok [B,M,d], sstats [B,M,2]
        gap = sstats[..., :1] - sstats[..., 1:2]
        x = torch.cat([tok, sstats, gap], -1)
        w = F.softplus(self.mlp_w(x)).squeeze(-1)
        p = centers.to(tok.device).unsqueeze(0) + torch.tanh(self.mlp_p(x))*(cell/2)
        return w, p                                  # w [B,M], p [B,M,2] in img px

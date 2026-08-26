import torch, torch.nn as nn, torch.nn.functional as F

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

class Condenser(nn.Module):
    """Exemplar->cell cross-attention: each cell queries the exemplar bank
    (prompt-aware features without touching the frozen backbone) -> 64ch map."""
    def __init__(self, d_in, d_sim=256, n_heads=4, ff=512, d_out=64):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_sim)
        self.attn = nn.MultiheadAttention(d_sim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_sim); self.norm2 = nn.LayerNorm(d_sim)
        self.ffn = nn.Sequential(nn.Linear(d_sim, ff), nn.GELU(), nn.Linear(ff, d_sim))
        self.out = nn.Linear(d_sim, d_out)
    def forward(self, tok, e):                       # tok [B,M,D], e [B,K,D]
        tok = self.proj_in(tok)
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

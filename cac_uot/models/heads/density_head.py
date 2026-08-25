import torch, torch.nn as nn, torch.nn.functional as F

class DensityHead(nn.Module):
    """Density map head: feature map -> density map [B,1,H,W]."""
    def __init__(self, in_dim=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1))
    def forward(self, feat):
        # feat [B,C,H,W] -> density [B,1,H,W] with softplus for positivity
        return F.softplus(self.net(feat))

import torch, torch.nn as nn, torch.nn.functional as F

def grid_centers(S, patch=16):
    g = S // patch
    ys = (torch.arange(g, dtype=torch.float32) + 0.5) * patch
    xs = (torch.arange(g, dtype=torch.float32) + 0.5) * patch
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2), g  # [M,2]

class PilePredictor(nn.Module):
    """PilePredictor: tokens + gate -> pile mass w + position p. No backbone coupling."""
    def __init__(self, dim=384, hidden=128):
        super().__init__()
        self.mlp_w = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.mlp_p = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp_p[-1].weight); nn.init.zeros_(self.mlp_p[-1].bias)
        nn.init.constant_(self.mlp_w[-1].bias, -3.5)
    def forward(self, tokens, gate, centers):
        w = F.softplus(self.mlp_w(tokens).squeeze(-1)) * gate.squeeze(-1)  # [B,M]
        dp = torch.tanh(self.mlp_p(tokens)) * 8.0
        p = centers.to(tokens.device).unsqueeze(0) + dp                   # [B,M,2]
        return w, p

import torch, torch.nn as nn, torch.nn.functional as F

def grid_centers(S, patch=16):
    g=S//patch
    ys=(torch.arange(g)+0.5)*patch; xs=(torch.arange(g)+0.5)*patch
    yy,xx=torch.meshgrid(ys,xs,indexing="ij")
    return torch.stack([xx,yy],-1).reshape(-1,2), g

class PileHead(nn.Module):
    """Masses+offsets per cell from token content + explicit similarity stats
    (max-sim, mean-sim, top1-top2 gap = class discriminability)."""
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.mlp_w=nn.Sequential(nn.Linear(d_model+3,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.mlp_p=nn.Sequential(nn.Linear(d_model+3,hidden),nn.GELU(),nn.Linear(hidden,2))
        nn.init.zeros_(self.mlp_p[-1].weight); nn.init.zeros_(self.mlp_p[-1].bias)
        nn.init.constant_(self.mlp_w[-1].bias, -3.5)
    def forward(self, tokens, sim, centers):  # tokens [B,M,D], sim [B,M,K]
        K=sim.shape[-1]
        top2=sim.topk(min(2,K),dim=-1).values
        gap=(top2[...,0]-top2[...,1]).unsqueeze(-1) if K>1 else torch.zeros_like(top2[...,:1])
        x=torch.cat([tokens, sim.max(-1,keepdim=True).values,
                     sim.mean(-1,keepdim=True), gap], -1)
        w=F.softplus(self.mlp_w(x)).squeeze(-1)
        p=centers.to(tokens.device).unsqueeze(0)+torch.tanh(self.mlp_p(x))*8
        return w,p

class DensityHead(nn.Module):
    """Per-cell expected count from token map + similarity stats."""
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(d_model+2,hidden,1),nn.GELU(),nn.Conv2d(hidden,1,1))
    def forward(self, tmap, sstats):  # [B,H,W,D], [B,H,W,2] -> [B,1,H,W]
        return F.softplus(self.net(torch.cat([tmap,sstats],-1).permute(0,3,1,2)))

import torch, torch.nn as nn, torch.nn.functional as F
def grid_centers(S, patch=16):
    g=S//patch
    ys=(torch.arange(g)+0.5)*patch; xs=(torch.arange(g)+0.5)*patch
    yy,xx=torch.meshgrid(ys,xs,indexing="ij")
    return torch.stack([xx,yy],-1).reshape(-1,2), g

class PileHead(nn.Module):
    def __init__(self, dim, hidden, cond_dim):
        super().__init__()
        self.mlp_w=nn.Sequential(nn.Linear(dim+cond_dim,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.mlp_p=nn.Sequential(nn.Linear(dim+cond_dim,hidden),nn.GELU(),nn.Linear(hidden,2))
        nn.init.zeros_(self.mlp_p[-1].weight); nn.init.zeros_(self.mlp_p[-1].bias)
        nn.init.constant_(self.mlp_w[-1].bias, -3.5)
    def forward(self, tokens, cond, centers):
        feats=torch.cat([tokens, cond.flatten(2).transpose(1,2)],-1)
        w=F.softplus(self.mlp_w(feats).squeeze(-1))
        p=centers.to(tokens.device).unsqueeze(0)+torch.tanh(self.mlp_p(feats))*8
        return w,p

class DensityHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(in_dim,hidden,1),nn.GELU(),nn.Conv2d(hidden,1,1))
    def forward(self, feat): return F.softplus(self.net(feat))

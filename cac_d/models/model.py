import math, torch, torch.nn as nn
import torch.nn.functional as F
from .backbone.backbone import ConvNeXtBackbone
from .prompt.prompt import ExemplarEncoder
from .heads.heads import PileHead, DensityHead, grid_centers
from .losses.losses import uot_loss

class Counter(nn.Module):
    """Frozen backbone -> exemplar ROI embeddings + token projection ->
    similarity field drives pile (UOT) and density branches."""
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg; self.S=cfg.image_size; D=cfg.embed_dim
        self.backbone=ConvNeXtBackbone(cfg)
        self.exemplar=ExemplarEncoder(cfg.backbone_dim,D,cfg.exemplar_layers,roi_size=cfg.roi_size)
        self.tproj=nn.Linear(cfg.backbone_dim,D)
        self.norm_t=nn.LayerNorm(D); self.norm_e=nn.LayerNorm(D)
        self.pile=PileHead(D,cfg.pile_hidden)
        self.density=DensityHead(D,cfg.density_hidden)
        centers,_=grid_centers(self.S,16)
        self.register_buffer("centers",centers)
    def forward(self,x,bboxes,points=None):
        feat=self.backbone.forward_feature_map(x)          # [B,C,h,w]
        B,_,H,W=feat.shape
        e=self.norm_e(self.exemplar(feat,bboxes,self.S))   # [B,K,D]
        t=self.norm_t(self.tproj(feat.flatten(2).transpose(1,2)))  # [B,M,D]
        sim=torch.einsum('bmd,bkd->bmk',t,e)/math.sqrt(t.shape[-1])  # [B,M,K]
        sstats=torch.stack([sim.max(-1).values,sim.mean(-1)],-1)     # [B,M,2]
        w,p=self.pile(t,sim,self.centers)
        dens=self.density(t.view(B,H,W,-1),sstats.view(B,H,W,2))     # [B,1,H,W]
        if points is None:
            return {"pred_counts":dens.sum((1,2,3)),"pile_count":w.sum(1),
                    "w":w,"p":p,"density":dens}
        gt=torch.zeros_like(dens)
        for b in range(len(points)):
            for (xi_,yi_) in points[b]:
                xi=int(max(0,min(W-1, xi_/self.S*W))); yi=int(max(0,min(H-1, yi_/self.S*H)))
                gt[b,0,yi,xi]+=1
        loss_den=F.mse_loss(dens,gt)
        loss_uot,_=uot_loss(p,w,points,S=self.S)
        loss_consist=F.mse_loss((w.sum(1)+1).log(),(dens.sum((1,2,3))+1).log())
        loss=loss_uot+self.cfg.density_weight*loss_den+self.cfg.consist_weight*loss_consist
        return {"loss":loss,"pred_counts":dens.sum((1,2,3)).detach(),
                "pile_count":w.sum(1).detach(),"w":w,"density":dens}

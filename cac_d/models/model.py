import torch, torch.nn as nn
from .backbone.backbone import ConvNeXtBackbone
from .prompt.prompt import PromptEncoder
from .heads.heads import PileHead, DensityHead, grid_centers
from .losses.losses import uot_loss
import torch.nn.functional as F
class Counter(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.S=cfg.image_size
        self.backbone=ConvNeXtBackbone(cfg)
        self.prompt=PromptEncoder(cfg.hidden_dim if hasattr(cfg,'hidden_dim') else 384, 256)
        g=self.S//16
        self.pile=PileHead(384,128,256)
        self.density=DensityHead(256,128)
        centers,_=grid_centers(self.S,16)
        self.register_buffer("centers",centers)
    def forward(self, x, bboxes, points=None):
        feat=self.backbone.forward_feature_map(x)  # [B,C,H,W]
        cond=self.prompt(feat,bboxes)
        B,C,H,W=feat.shape
        tokens=feat.flatten(2).transpose(1,2)  # [B,M,C]
        w,p=self.pile(tokens, cond.flatten(2).transpose(1,2), self.centers)
        dens=self.density(cond)
        if points is None:
            return {"pred_counts":dens.sum((1,2,3)), "w":w, "p":p, "density":dens}
        # GT density for auxiliary
        gt=torch.zeros_like(dens)
        for b in range(len(points)):
            for (x,y) in points[b]:
                xi=int(max(0,min(W-1, x/self.S*W))); yi=int(max(0,min(H-1, y/self.S*H)))
                gt[b,0,yi,xi]+=1
        loss_den=F.mse_loss(dens, gt)
        loss_uot,_=uot_loss(p,w,points,S=self.S)
        loss=loss_uot + self.cfg.density_weight*loss_den
        return {"loss":loss, "pred_counts":dens.sum((1,2,3)).detach(), "w":w, "density":dens}

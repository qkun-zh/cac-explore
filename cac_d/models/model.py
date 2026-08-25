import math, torch, torch.nn as nn
import torch.nn.functional as F
from .backbone.backbone import ConvNeXtBackbone
from .prompt.prompt import ExemplarEncoder
from .heads.heads import PileHead, DensityHead, grid_centers
from .losses.losses import uot_loss

class Counter(nn.Module):
    """Frozen HF backbone -> ROI exemplar embeddings + token projection ->
    similarity field drives pile (UOT) and density branches."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg; self.S = cfg.image_size; D = cfg.embed_dim
        self.backbone = ConvNeXtBackbone(cfg)
        C = self.backbone.out_channels
        self.exemplar = ExemplarEncoder(C, D, cfg.exemplar_layers, roi_size=cfg.roi_size)
        self.tproj = nn.Linear(C, D)
        self.norm_t = nn.LayerNorm(D); self.norm_e = nn.LayerNorm(D)
        self.pile = PileHead(D, cfg.pile_hidden)
        self.density = DensityHead(D, cfg.density_hidden)
        centers, _ = grid_centers(self.S, cfg.patch_stride)
        self.register_buffer("centers", centers)

    def train(self, mode=True):     # backbone stays eval: no dropout/BN drift
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x, bboxes, points=None):
        feat = self.backbone.forward_feature_map(x)                # [B,C,h,w]
        B, _, H, W = feat.shape
        e = self.norm_e(self.exemplar(feat, bboxes, self.S))       # [B,K,D]
        t = self.norm_t(self.tproj(feat.flatten(2).transpose(1, 2)))  # [B,M,D]
        sim = torch.einsum('bmd,bkd->bmk', t, e) / math.sqrt(t.shape[-1])  # [B,M,K]
        sstats = torch.stack([sim.max(-1).values, sim.mean(-1)], -1)       # [B,M,2]
        w, p = self.pile(t, sim, self.centers)
        dens = self.density(t.view(B, H, W, -1), sstats.view(B, H, W, 2))  # [B,1,H,W]
        if points is None:
            return {"pred_counts": dens.sum((1, 2, 3)), "pile_count": w.sum(1),
                    "w": w, "p": p, "density": dens}
        gt = self._gt_density(points, B, H, W)
        loss_uot, _ = uot_loss(p, w, points, S=self.S, eps=self.cfg.entropy_reg,
                               tau=self.cfg.demand_tau, alpha=self.cfg.transport_weight,
                               iters=self.cfg.sinkhorn_iters)
        loss_den = F.mse_loss(dens, gt)
        loss_consist = F.mse_loss((w.sum(1)+1).log(), (dens.sum((1,2,3))+1).log())
        loss = loss_uot + self.cfg.density_weight*loss_den + self.cfg.consist_weight*loss_consist
        return {"loss": loss, "pred_counts": dens.sum((1,2,3)).detach(),
                "pile_count": w.sum(1).detach(), "w": w, "density": dens}

    def _gt_density(self, points, B, H, W):
        dev = next(self.parameters()).device
        flat = torch.zeros(B, H*W, device=dev)
        scale = W / float(self.S)
        for b, pts in enumerate(points):
            if pts.numel() == 0: continue
            idx = (pts.to(dev) * scale).long()
            xi = idx[:, 0].clamp(0, W-1); yi = idx[:, 1].clamp(0, H-1)
            flat[b].index_add_(0, yi*W+xi, torch.ones(len(idx), device=dev))
        return flat.view(B, 1, H, W)

import torch, torch.nn as nn
import torch.nn.functional as F
from .backbone.backbone import ConvNeXtBackbone
from .prompt.prompt import ExemplarEncoder
from .heads.heads import FineFuser, Condenser, DensityDecoder
from .losses.losses import gaussian_density

class Counter(nn.Module):
    """Frozen backbone → fuser → exemplar cross-attention → density map.
    Losses: density MSE + count smooth-L1 only."""
    def __init__(self, cfg, cached=False):
        super().__init__()
        self.cfg = cfg; self.S = cfg.image_size; self.cached = cached
        ch_mid, ch_coarse = cfg.backbone_dims
        D = cfg.d_fine
        self.fuser = FineFuser(ch_coarse, ch_mid, d_fine=D)
        if not cached:
            self.backbone = ConvNeXtBackbone(cfg)
            ch_mid, ch_coarse = self.backbone.out_channels
            self.exemplar = ExemplarEncoder(ch_coarse, cfg.embed_dim,
                                            cfg.exemplar_layers, roi_size=cfg.roi_size)
        else:
            self.backbone = None
            self.exemplar = None
        self.cond = Condenser(d_in=D, d_sim=cfg.embed_dim, d_out=cfg.cond_dim)
        self.density = DensityDecoder(in_ch=D + cfg.cond_dim, hidden=2*D)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def forward(self, x, bboxes, points=None, h2=None, h3=None, e=None):
        if self.cached:
            assert h2 is not None and h3 is not None and e is not None
        else:
            h2, h3 = self.backbone.forward_feature_map(x)
            e = self.exemplar(h3, bboxes, self.S)
        B, _, Hh, _ = h3.shape
        Hf = Wf = Hh * 4
        fine = self.fuser(h2, h3)                                   # [B,D,Hf,Wf]
        fmap = fine.permute(0, 2, 3, 1).flatten(1, 2)              # [B,M,D]
        cond = self.cond(fmap, e)                                   # [B,M,cond_dim]
        dens = self.density(torch.cat([fine, cond.transpose(1, 2).reshape(B, -1, Hf, Wf)], 1))
        counts = dens.sum((1, 2, 3))
        if points is None:
            return {"pred_counts": counts, "density": dens}
        gt_d = gaussian_density(points, B, Hf, Wf, self.S, sigma=self.cfg.gauss_sigma)
        loss_den = F.mse_loss(dens, gt_d)
        N = torch.tensor([len(q) for q in points], device=dens.device, dtype=torch.float32)
        loss_cnt = F.smooth_l1_loss((counts+1).log(), (N+1).log())
        loss = self.cfg.density_weight*loss_den + self.cfg.cnt_weight*loss_cnt
        return {"loss": loss, "pred_counts": counts.detach(), "density": dens,
                "loss_den": loss_den.detach(), "loss_cnt": loss_cnt.detach()}

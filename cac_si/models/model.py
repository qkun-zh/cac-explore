"""SICounter: frozen DINOv3 dual-stream + cross-attention + INR continuous density."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from cac_d.models.backbone.backbone import ConvNeXtBackbone
from cac_d.models.heads.heads import Condenser
from .encoder import ScaleInvariantEncoder, PromptEncoder
from .inr import INRDecoder, gt_density_at, sample_map


class SICounter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bb = ConvNeXtBackbone(cfg)              # frozen
        H0 = W0 = cfg.image_size // 16
        self.grid = (H0, W0)
        self.enc_img = ScaleInvariantEncoder(self.bb, cfg.scales, self.grid)
        self.enc_pmt = PromptEncoder(self.bb, self.grid, cfg.prompt_size, cfg.prompt_margin)
        C = cfg.backbone_dims[1]
        self.cond = Condenser(d_in=C, d_sim=cfg.d_sim, n_heads=cfg.n_heads,
                              ff=cfg.ff, d_out=cfg.cond_dim)
        self.inr = INRDecoder(C + cfg.cond_dim, cfg.inr_hidden,
                              cfg.inr_layers, cfg.fourier_freqs)

    def _regular_grid(self, g, device):
        c = (torch.arange(g, device=device, dtype=torch.float32) + 0.5) / g
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        return torch.stack([gx.flatten(), gy.flatten()], -1)     # [g*g,2]

    def forward(self, img, bboxes, points=None):
        cfg = self.cfg
        dev = img.device
        a = self.enc_img(img)                            # [B,C,H0,W0]
        bp = self.enc_pmt(img, bboxes)                   # [B,K,C,H0,W0]
        B, K, C, H0, W0 = bp.shape
        q = a.flatten(2).transpose(1, 2)                 # [B,M,C]
        kv = bp.flatten(2).transpose(1, 2).reshape(B, K * H0 * W0, C)
        cond = self.cond(q, kv)                          # [B,M,cond_dim]
        c = torch.cat([q, cond], -1)                     # [B,M,C+cond]
        cmap = c.transpose(1, 2).reshape(B, -1, H0, W0)

        # count via quadrature (integral of u over [0,1]^2)
        xq = self._regular_grid(cfg.quad_grid, dev)
        uq = self.inr(sample_map(cmap, xq).reshape(-1, c.shape[-1]),
                      xq.repeat(B, 1))
        count = uq.view(B, -1).mean(1)                   # ∫u ≈ mean on unit square
        if points is None:
            return {"pred_counts": count, "density_map": cmap}

        xs = torch.rand(cfg.n_samples, 2, device=dev)    # shared across batch
        u = self.inr(sample_map(cmap, xs).reshape(-1, c.shape[-1]),
                     xs.repeat(B, 1)).view(B, -1)        # [B,M]
        gt = gt_density_at(points, cfg.image_size, xs, cfg.inr_sigma)
        loss_den = F.mse_loss(u, gt)
        N = torch.tensor([len(p) for p in points], device=dev, dtype=torch.float32)
        loss_cnt = F.smooth_l1_loss((count + 1).log(), (N + 1).log())
        loss = cfg.density_weight * loss_den + cfg.cnt_weight * loss_cnt
        return {"loss": loss, "pred_counts": count.detach(),
                "loss_den": loss_den.detach(), "loss_cnt": loss_cnt.detach()}

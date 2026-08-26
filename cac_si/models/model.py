"""SICounter: frozen DINOv3 dual-stream + cross-attention + INR continuous density."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from cac_d.models.backbone.backbone import ConvNeXtBackbone
from cac_d.models.heads.heads import Condenser
from cac_d.models.losses.losses import gaussian_density
from .encoder import ScaleInvariantEncoder, PromptEncoder
from .inr import INRDecoder, sample_map


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
        self.kv_proj = nn.Linear(C, cfg.d_sim)       # b' 384 -> d_sim before MHA
        self.cond = Condenser(d_in=C, d_sim=cfg.d_sim, n_heads=cfg.n_heads,
                              ff=cfg.ff, d_out=cfg.cond_dim)
        self.inr = INRDecoder(C + cfg.cond_dim, cfg.inr_hidden, cfg.inr_layers)

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
        q = a.flatten(2).transpose(1, 2)                 # [B,M,C]  M=H0*W0
        kv = bp.permute(0, 1, 3, 4, 2).reshape(B, K * H0 * W0, C)
        kv = self.kv_proj(kv)                            # [B,K*M,C_sim]
        cond = self.cond(q, kv)                          # [B,M,cond_dim]
        c = torch.cat([q, cond], -1)                     # [B,M,C+cond]
        cmap = c.transpose(1, 2).reshape(B, -1, H0, W0)

        # count via quadrature: u matches DISCRETE-map values (sum=N on S-grid),
        # whose integral over the unit square is N/S^2 -> count = int(u) * S^2.
        # INR runs in fp32: DINOv3 outlier features blow past fp16 range in u^2.
        g = cfg.quad_grid if self.training else cfg.eval_grid
        xq = self._regular_grid(g, dev)
        with torch.autocast("cuda", enabled=False):
            cmap32 = cmap.float()
            uq = self.inr(sample_map(cmap32, xq).reshape(-1, cmap32.shape[1])).view(B, -1)
            count = uq.mean(1) * float(cfg.image_size) ** 2
        if points is None:
            return {"pred_counts": count, "density_map": cmap}

        xs = torch.rand(cfg.n_samples, 2, device=dev)    # shared across batch
        # paper §3.4: D_gt(x) via interpolation from the DISCRETE density map
        # (standard DME convention: kernel sums to 1/point, map sums to N),
        # NOT the analytic pdf — value scale ~1e-3 keeps losses balanced.
        with torch.autocast("cuda", enabled=False):
            u = self.inr(sample_map(cmap32, xs).reshape(-1, cmap32.shape[1])).view(B, -1)
            gt_maps = gaussian_density([p.float() for p in points], B,
                                       cfg.image_size, cfg.image_size,
                                       cfg.image_size,
                                       sigma=cfg.inr_sigma * cfg.image_size)
            gt = sample_map(gt_maps, xs.float()).squeeze(-1)   # [B,M] bilinear interp
            loss_den = F.mse_loss(u, gt)
        N = torch.tensor([len(p) for p in points], device=dev, dtype=torch.float32)
        loss_cnt = F.smooth_l1_loss((count + 1).log(), (N + 1).log())
        loss = cfg.density_weight * loss_den + cfg.cnt_weight * loss_cnt
        return {"loss": loss, "pred_counts": count.detach(),
                "loss_den": loss_den.detach(), "loss_cnt": loss_cnt.detach()}

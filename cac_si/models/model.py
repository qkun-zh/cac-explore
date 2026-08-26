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
        # Kendall&Gal uncertainty weighting: auto-balance L_den / L_cnt scales
        self.uncertainty_weight = bool(getattr(cfg, "uncertainty_weight", False))
        if self.uncertainty_weight:
            self.log_s = nn.Parameter(torch.zeros(2))    # [den, cnt]
        # 2D sincos positional encoding on attention tokens (DETR-style)
        self.pos_enc = bool(getattr(cfg, "pos_enc", False))
        if self.pos_enc:
            self.register_buffer("pe", self._build_2d_sincos(H0, W0, C))

    @staticmethod
    def _build_2d_sincos(H, W, dim):
        import math
        assert dim % 4 == 0, "pos-enc dim must be divisible by 4"
        d = dim // 2                                     # half for y, half for x
        pe = torch.zeros(H * W, dim)
        gy, gx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing="ij")
        gy = gy.flatten(); gx = gx.flatten()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0:d:2] = torch.sin(gy.unsqueeze(1) * div)
        pe[:, 1:d:2] = torch.cos(gy.unsqueeze(1) * div)
        pe[:, d::2] = torch.sin(gx.unsqueeze(1) * div)
        pe[:, d + 1::2] = torch.cos(gx.unsqueeze(1) * div)
        return pe

    def _sample_xs(self, points, B, dev):
        """Mixed sampling: (1-fg) uniform + fg near GT points (jitter 3sigma)."""
        cfg = self.cfg
        n_fg = int(cfg.n_samples * float(cfg.fg_sampling))
        n_uni = cfg.n_samples - n_fg
        xs_uni = torch.rand(max(n_uni, 1), 2, device=dev)
        if n_fg == 0:
            return xs_uni                                # [M,2] shared
        fg = []
        for p in points:
            if p.numel() == 0:
                fg.append(torch.rand(n_fg, 2, device=dev))
            else:
                idx = torch.randint(0, p.shape[0], (n_fg,), device=dev)
                base = p.to(dev)[idx] / float(cfg.image_size)
                jit = torch.randn(n_fg, 2, device=dev) * (3.0 * cfg.inr_sigma)
                fg.append((base + jit).clamp(0.0, 1.0))
        xs = torch.stack(fg, 0)                          # [B,n_fg,2]
        if n_uni > 0:
            xs = torch.cat([xs_uni.unsqueeze(0).expand(B, -1, -1), xs], 1)
        return xs                                        # [B,M,2] per-image

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
        if self.pos_enc:
            q = q + self.pe.unsqueeze(0)
        kv = bp.permute(0, 1, 3, 4, 2).reshape(B, K * H0 * W0, C)
        if self.pos_enc:
            kv = kv + self.pe.repeat(K, 1).unsqueeze(0)  # same grid per exemplar
        kv = self.kv_proj(kv)                            # [B,K*M,C_sim]
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

        xs = self._sample_xs(points, B, dev)             # [M,2] or [B,M,2]
        # paper §3.4: D_gt(x) via interpolation from the DISCRETE density map
        # (standard DME convention: kernel sums to 1/point, map sums to N),
        # NOT the analytic pdf — value scale ~1e-3 keeps losses balanced.
        with torch.autocast("cuda", enabled=False):
            cmap32 = cmap.float()
            u = self.inr(sample_map(cmap32, xs).reshape(-1, cmap32.shape[1])).view(B, -1)
            gt_maps = gaussian_density([p.float() for p in points], B,
                                       cfg.image_size, cfg.image_size,
                                       cfg.image_size,
                                       sigma=cfg.inr_sigma * cfg.image_size)
            gt = sample_map(gt_maps, xs.float()).squeeze(-1)   # [B,M] bilinear interp
            loss_den = F.mse_loss(u, gt)
        N = torch.tensor([len(p) for p in points], device=dev, dtype=torch.float32)
        if self.uncertainty_weight:
            loss_cnt = F.smooth_l1_loss((count + 1).clamp_min(1e-6).log(), (N + 1).log())
            loss = (loss_den * torch.exp(-2 * self.log_s[0]) + 2 * self.log_s[0]
                    + loss_cnt * torch.exp(-2 * self.log_s[1]) + 2 * self.log_s[1])
        else:
            if cfg.cnt_weight > 0:
                loss_cnt = F.smooth_l1_loss((count + 1).log(), (N + 1).log())
            else:
                loss_cnt = torch.zeros((), device=dev)   # paper: no count term at all
            loss = cfg.density_weight * loss_den + cfg.cnt_weight * loss_cnt
        return {"loss": loss, "pred_counts": count.detach(),
                "loss_den": loss_den.detach(), "loss_cnt": loss_cnt.detach()}

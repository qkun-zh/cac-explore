import torch, torch.nn as nn
import torch.nn.functional as F
from .backbone.backbone import ConvNeXtBackbone
from .prompt.prompt import ExemplarEncoder
from .heads.heads import FineFuser, SimModule, Condenser, DensityDecoder, PileHead, grid_centers
from .losses.losses import gaussian_density, sim_margin_loss, uot_loss, ot_coverage_loss

class Counter(nn.Module):
    """Frozen HF backbone (multi-scale) -> fine token map @1/4 + explicit
    exemplar similarity (supervised) + exemplar->cell cross-attention ->
    density map at 96x96 (primary count); top-K UOT point branch auxiliary.
    Supports cached mode: pass precomputed h2/h3/e to skip backbone."""
    def __init__(self, cfg, cached=False):
        super().__init__()
        self.cfg = cfg; self.S = cfg.image_size; self.cached = cached
        ch_mid, ch_coarse = cfg.backbone_dims                       # [192, 384]
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
        self.sim = SimModule(d_fine=D, d_sim=cfg.embed_dim)
        self.cond = Condenser(d_sim=cfg.embed_dim, d_out=cfg.cond_dim)
        self.cell = self.S // (cfg.image_size // 4)                 # fine cell px (=4)
        centers, _ = grid_centers(self.S, self.cell)
        self.register_buffer("centers", centers)
        self.pile = PileHead(D, cfg.pile_hidden)
        self.density = DensityDecoder(in_ch=D + 2 + cfg.cond_dim, hidden=2*D)

    def train(self, mode=True):                                     # backbone stays eval
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
        Hf = Wf = Hh * 4                                            # 96 @ 384 input
        fine = self.fuser(h2, h3)                                   # [B,D,96,96]
        S, smax, smean, tok256 = self.sim(fine.permute(0, 2, 3, 1), e)
        cond = self.cond(tok256, e)                                 # [B,M,cond_dim]
        dens = self.density(torch.cat([fine, smax, smean,
                                       cond.transpose(1, 2).reshape(B, -1, Hf, Wf)], 1))
        counts = dens.sum((1, 2, 3))
        if points is None:
            return {"pred_counts": counts, "density": dens}
        gt_d = gaussian_density(points, B, Hf, Wf, self.S, sigma=self.cfg.gauss_sigma)
        loss_den = F.mse_loss(dens, gt_d)
        N = torch.tensor([len(q) for q in points], device=dens.device, dtype=torch.float32)
        loss_cnt = F.smooth_l1_loss((counts+1).log(), (N+1).log())
        loss_sim = sim_margin_loss(smax, gt_d)
        if self.cfg.use_ot_coverage:
            loss_ot = ot_coverage_loss(S, gt_d,
                                        eps=self.cfg.ot_epsilon,
                                        iters=self.cfg.ot_iters)
            ot_w = self.cfg.ot_weight
        else:
            sstats = torch.stack([smax.squeeze(1), smean.squeeze(1)], -1).flatten(1, 2)
            w, p = self.pile(fine.flatten(2).transpose(1, 2), sstats, self.centers, self.cell)
            loss_ot = self._uot_topk(w, p, points)
            ot_w = self.cfg.transport_weight
        loss = (self.cfg.density_weight*loss_den + self.cfg.cnt_weight*loss_cnt +
                self.cfg.sim_weight*loss_sim + ot_w*loss_ot)
        return {"loss": loss, "pred_counts": counts.detach(), "density": dens}

    def _uot_topk(self, w, p, points):
        K = min(self.cfg.uot_topk, w.shape[1])
        idx = w.detach().topk(K, dim=1).indices                     # most-mass cells
        ws = torch.gather(w, 1, idx)
        ps = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
        out, _ = uot_loss(ps, ws, points, S=self.S, eps=self.cfg.entropy_reg,
                          tau=self.cfg.demand_tau, alpha=1.0,
                          iters=self.cfg.sinkhorn_iters)
        return out

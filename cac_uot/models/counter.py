import torch, torch.nn as nn, torch.nn.functional as F
from .backbone.dinov3 import DINOv3HFBackbone
from .prompt.ope_prototype import OPEModule, PositionalEncodingsFixed as OPEPosEmb, ope_response_maps
from .heads.pile_predictor import PilePredictor, grid_centers
from .losses.unbalanced_ot import unbalanced_ot_loss
from .losses.repulsion import repulsion
from .losses.uw import UncertaintyWeighting

class UOTCounter(nn.Module):
    """Counter via unbalanced OT. Assembled via dependency injection."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.S, self.patch = cfg.image_size, cfg.patch_size
        self.backbone = DINOv3HFBackbone(cfg)
        g = self.S // self.patch
        self.ope_input_proj = nn.Conv2d(cfg.hidden_dim, cfg.ope_emb_dim, kernel_size=1)
        self.ope = OPEModule(
            num_iterative_steps=cfg.ope_iters, emb_dim=cfg.ope_emb_dim,
            kernel_dim=cfg.ope_kernel_dim, num_objects=3, num_heads=cfg.ope_heads,
            reduction=cfg.ope_reduction)
        self.pos_emb_ope = OPEPosEmb(cfg.ope_emb_dim)
        self.cond_dim = cfg.ope_emb_dim
        self.head = PilePredictor(cfg.hidden_dim, cfg.head_hidden, cond_dim=self.cond_dim)
        centers,_ = grid_centers(self.S, self.patch)
        self.register_buffer("centers", centers)
        self.use_uw = bool(getattr(cfg, "use_uw", False))
        if self.use_uw:
            # terms: [uot, cnt_mass, rep]
            self.uw = UncertaintyWeighting(3, init=cfg.uw_init)

    def forward(self, pixel_values, bboxes3, points=None):
        tokens = self.backbone.forward_tokens(pixel_values)   # [B,M,C]
        B, M, C = tokens.shape
        fm_raw = tokens.transpose(1, 2).reshape(B, C, self.S // self.patch, self.S // self.patch)
        fm = self.ope_input_proj(fm_raw)                      # project to ope_emb_dim
        pos = self.pos_emb_ope(B, fm.shape[2], fm.shape[3], fm.device).flatten(2).permute(2, 0, 1)
        protos = self.ope(fm, pos, bboxes3)[-1]               # last iteration [k^2*n, B, D]
        resp = ope_response_maps(fm, protos, self.cfg.ope_kernel_dim, 3)  # [B,D,h,w]
        cond = resp.flatten(2).permute(0, 2, 1)           # [B,M,D]
        w, p = self.head(tokens, cond, self.centers)          # [B,M], [B,M,2]
        if points is None:
            return {"w": w, "p": p, "pred_counts": w.sum(1), "counts_sumw": w.sum(1)}
        wh = (bboxes3[:,:,2:4]-bboxes3[:,:,0:2]).clamp_min(1); sigma = wh.mean().item() * self.cfg.repulsion_sigma_scale
        loss_uot, met, cnt_open = unbalanced_ot_loss(p, w, points,
            transport_weight=self.cfg.transport_weight, supply_tau=self.cfg.supply_tau,
            demand_tau=self.cfg.demand_tau, entropy_reg=self.cfg.entropy_reg,
            S=self.S, sinkhorn_iters=self.cfg.sinkhorn_iters)
        rep = sum(repulsion(p[b:b+1], w[b:b+1], self.cfg.repulsion_weight, max(sigma,8), self.S) for b in range(w.shape[0])) / w.shape[0]
        # P1 direct count-mass supervision |Σw − N|
        n_gt = torch.tensor([len(g) for g in points], dtype=torch.float32, device=w.device)
        count_mass = (w.sum(1) - n_gt).abs().mean() * self.cfg.count_mass_weight
        if self.use_uw:
            loss, uw_w = self.uw([loss_uot, count_mass, rep])
            met = {**met, "rep": rep.item(), "cnt_mass": count_mass.item(),
                   "uw_uot": uw_w["w0"], "uw_cnt": uw_w["w1"], "uw_rep": uw_w["w2"]}
        else:
            loss = loss_uot + rep + count_mass
        return {"w": w, "p": p, "loss": loss, "pred_counts": cnt_open,
                "counts_sumw": w.sum(1).detach(), "metrics": {**met, "cnt_mass": count_mass.item()}}

def build_model(cfg): return UOTCounter(cfg)

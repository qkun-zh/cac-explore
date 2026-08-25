import torch, torch.nn as nn, torch.nn.functional as F
from .backbone.dinov3 import DINOv3HFBackbone
from .prompt.cosine_gate import CosineGate, StandardizedCosineGate
from .heads.pile_predictor import PilePredictor, grid_centers
from .losses.unbalanced_ot import unbalanced_ot_loss
from .losses.repulsion import repulsion
from .losses.anchor import box_mass_anchor

class UOTCounter(nn.Module):
    """Counter via unbalanced OT. Assembled via dependency injection."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.S, self.patch = cfg.image_size, cfg.patch_size
        self.backbone = DINOv3HFBackbone(cfg)
        Gate = StandardizedCosineGate if cfg.use_standardized_gate else CosineGate
        self.gate = Gate(cfg.hidden_dim)
        self.head = PilePredictor(cfg.hidden_dim, cfg.head_hidden)
        centers,_ = grid_centers(self.S, self.patch)
        self.register_buffer("centers", centers)

    def forward(self, pixel_values, bboxes3, points=None):
        tokens = self.backbone.forward_tokens(pixel_values)   # [B,M,C]
        gate = self.gate(tokens, bboxes3, self.S, self.patch)  # [B,M,1]
        w, p = self.head(tokens, gate, self.centers)           # [B,M], [B,M,2]
        if points is None:
            return {"w": w, "p": p, "gate": gate, "pred_counts": w.sum(1), "counts_sumw": w.sum(1)}
        wh = (bboxes3[:,:,2:4]-bboxes3[:,:,0:2]).clamp_min(1); sigma = wh.mean().item() * self.cfg.repulsion_sigma_scale
        loss_uot, met, cnt_open = unbalanced_ot_loss(p, w, points,
            transport_weight=self.cfg.transport_weight, supply_tau=self.cfg.supply_tau,
            demand_tau=self.cfg.demand_tau, entropy_reg=self.cfg.entropy_reg,
            S=self.S, sinkhorn_iters=self.cfg.sinkhorn_iters)
        rep = sum(repulsion(p[b:b+1], w[b:b+1], self.cfg.repulsion_weight, max(sigma,8), self.S) for b in range(w.shape[0])) / w.shape[0]
        anchor = box_mass_anchor(w, p, bboxes3, self.cfg.box_anchor_weight)
        loss = loss_uot + rep + anchor
        if self.cfg.loss_normalize == "demand_size":
            avg_n = sum(len(g) for g in points) / max(len(points),1)
            loss = loss / max(avg_n, 1)
        return {"w": w, "p": p, "gate": gate, "loss": loss, "pred_counts": cnt_open,
                "counts_sumw": w.sum(1).detach(), "metrics": {**met, "rep": rep.item(), "anchor": anchor.item() if torch.is_tensor(anchor) else anchor}}

def build_model(cfg): return UOTCounter(cfg)

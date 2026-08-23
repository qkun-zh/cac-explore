"""N0002_dino_protocorr — frozen DINOv2-S prototype-correlation counting."""
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "vit_small_patch14_reg4_dinov2.lvd142m"
PATCH = 14


class ProtoCorrHead(nn.Module):
    def __init__(self, dim=384, proj_dim=256, dec_width=32):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.LayerNorm(proj_dim))
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.decoder = nn.Sequential(
            nn.Conv2d(1, dec_width, 3, padding=1), nn.GELU(),
            nn.Conv2d(dec_width, dec_width, 3, padding=1), nn.GELU(),
            nn.Conv2d(dec_width, 1, 3, padding=1),
        )

    def forward(self, tokens, patches_side, size, bboxes):
        t = self.proj(tokens)
        proto = self._roi_mean(t, patches_side, size, bboxes)
        sim = torch.einsum("bc,bpc->bp", proto, t)
        sim = sim / (t.norm(dim=-1) * proto.norm(dim=-1, keepdim=True)).clamp_min(1e-6)
        sim_map = (sim * F.softplus(self.tau)).reshape(-1, 1, patches_side, patches_side)
        return self.decoder(sim_map)

    @staticmethod
    def _roi_mean(t, ps, s, bboxes):
        dev = t.device
        ys, xs = torch.meshgrid(torch.arange(ps, device=dev), torch.arange(ps, device=dev), indexing="ij")
        step = s / ps
        cx, cy = (xs + 0.5) * step, (ys + 0.5) * step
        inside = ((cx[None] >= bboxes[:, 0, None, None]) & (cy[None] >= bboxes[:, 1, None, None]) &
                  (cx[None] <= bboxes[:, 2, None, None]) & (cy[None] <= bboxes[:, 3, None, None]))
        inside = inside.reshape(inside.shape[0], -1).float()
        w = inside / inside.sum(-1, keepdim=True).clamp_min(1.0)
        return torch.einsum("bp,bpc->bc", w.to(t.dtype), t)


class DinoProtoCorr(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        self.backbone = timm.create_model(cfg.get("backbone", BACKBONE), pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.patch = int(self.backbone.patch_embed.patch_size[0])
        self.head = ProtoCorrHead(dim=self.backbone.embed_dim, proj_dim=int(cfg.get("proj_dim", 256)))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, imgs, bboxes):
        s = imgs.shape[-1]
        with torch.no_grad():
            toks = self.backbone.forward_features(imgs)
        ps = s // self.patch
        tokens = toks[:, -ps * ps:, :]
        dens = self.head(tokens.float(), ps, s, bboxes)
        dens = F.interpolate(dens, scale_factor=2, mode="bilinear", align_corners=False)
        return {"density": dens}


def build_model(cfg):
    return DinoProtoCorr(cfg)

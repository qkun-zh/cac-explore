"""N0004_effnet_pyrmatch — frozen EfficientNet-B0 multi-scale exemplar matching with learned scale gating."""
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "efficientnet_b0.ra_in1k"


class ScaleGateHead(nn.Module):
    def __init__(self, chans=(40, 112, 320), proj=64, strides=(8, 16, 32), dec_width=48):
        super().__init__()
        self.strides = strides
        self.proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, proj, 1), nn.BatchNorm2d(proj), nn.GELU()) for c in chans)
        self.gate = nn.Conv2d(proj * len(chans), len(chans), 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(1, dec_width, 3, padding=1), nn.GELU(),
            nn.Conv2d(dec_width, dec_width, 3, padding=1), nn.GELU(),
            nn.Conv2d(dec_width, 1, 3, padding=1),
        )

    def forward(self, feats, bboxes, size):
        ref = feats[0].shape[-2:]
        sims = []
        for f, proj, st in zip(feats, self.proj, self.strides):
            t = proj(f.float())
            proto = self._roi_mean(t, st, size, bboxes)
            sim = torch.einsum("bc,bchw->bhw", proto, t) / (
                t.norm(dim=1) * proto.norm(dim=-1).view(-1, 1, 1)).clamp_min(1e-6)
            sims.append(F.interpolate(sim.unsqueeze(1), size=ref, mode="bilinear", align_corners=False))
        cat = torch.cat(sims, dim=1)
        w = torch.softmax(self.gate(cat), dim=1)
        sim_final = (w * cat).sum(1, keepdim=True)
        return self.decoder(sim_final)

    @staticmethod
    def _roi_mean(t, stride, size, bboxes):
        _, _, h, w = t.shape
        dev = t.device
        ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
        cx, cy = (xs + 0.5) * (size / w), (ys + 0.5) * (size / h)
        inside = ((cx[None] >= bboxes[:, 0, None, None]) & (cy[None] >= bboxes[:, 1, None, None]) &
                  (cx[None] <= bboxes[:, 2, None, None]) & (cy[None] <= bboxes[:, 3, None, None]))
        m = inside.reshape(inside.shape[0], -1).float()
        m = m / m.sum(-1, keepdim=True).clamp_min(1.0)
        B = t.flatten(2).transpose(1, 2)
        return torch.einsum("bp,bpc->bc", m.to(B.dtype), B)


class EffNetPyrMatch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        self.backbone = timm.create_model(cfg.get("backbone", BACKBONE), pretrained=True,
                                          features_only=True, out_indices=(2, 3, 4))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        chans = tuple(self.backbone.feature_info.channels())
        self.head = ScaleGateHead(chans=chans)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, imgs, bboxes):
        with torch.no_grad():
            feats = self.backbone(imgs)
        dens = self.head(feats, bboxes, imgs.shape[-1])
        return {"density": dens}


def build_model(cfg):
    return EffNetPyrMatch(cfg)

"""N0020_dino_scaleaware — champion + exemplar-scale deformable sampling between adapter and head."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "vit_small_patch14_reg4_dinov2.lvd142m"
PATCH = 14


class PromptEncoderV2(nn.Module):
    def __init__(self, freqs=8, hidden=256, out_dim=384):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(freqs) * math.pi)
        self.mlp = nn.Sequential(nn.Linear(4 * freqs * 2 + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, bboxes, size):
        b = bboxes / float(size)
        w = (b[:, 2] - b[:, 0]).clamp_min(1e-4)
        h = (b[:, 3] - b[:, 1]).clamp_min(1e-4)
        cxywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, w, h], dim=1)
        ang = cxywh[..., None] * self.freqs
        fourier = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)
        log_area = torch.log(w * h).unsqueeze(1).clamp(-13.8, 0.0)
        return self.mlp(torch.cat([fourier, log_area], dim=1))


class ScaleAwareSampler(nn.Module):
    def __init__(self, ch=384, grid=3):
        super().__init__()
        self.G = grid
        self.proj = nn.Linear(ch * grid * grid, ch)

    def forward(self, fmap, bboxes, S):
        B, C, H, W = fmap.shape
        G = self.G
        dx = bboxes[:, 2] - bboxes[:, 0]
        dy = bboxes[:, 3] - bboxes[:, 1]
        diag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        spacing = (diag / float(S) * H).clamp(0.5, 4.0).view(B, 1, 1)
        ys = torch.arange(H, device=fmap.device, dtype=torch.float32)
        xs = torch.arange(W, device=fmap.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys + .5, xs + .5, indexing="ij")
        cy, cx = yy.flatten().view(1, -1, 1), xx.flatten().view(1, -1, 1)
        offs = torch.arange(G, device=fmap.device, dtype=torch.float32) - (G - 1) / 2
        oy, ox = [t.flatten().view(1, 1, -1) for t in torch.meshgrid(offs, offs, indexing="ij")]
        sy, sx = cy + oy * spacing, cx + ox * spacing
        grid = torch.stack([sx / W * 2 - 1, sy / H * 2 - 1], dim=-1)
        sampled = F.grid_sample(fmap, grid, align_corners=False)
        return self.proj(sampled.permute(0, 2, 1, 3).reshape(B, H * W, C * G * G))


class DinoScaleAware(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        ch = 384
        dim = int(cfg.get("adapter_dim", 768))
        drop = float(cfg.get("dropout", 0.1))
        self.backbone = timm.create_model(cfg.get("backbone", BACKBONE), pretrained=True,
                                          dynamic_img_size=True, features_only=True, out_indices=(6, 11))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.patch = PATCH
        self.t6_proj = nn.Linear(ch, ch)
        self.t11_proj = nn.Linear(ch, ch)
        self.layer_logits = nn.Parameter(torch.zeros(2))
        self.prompt_enc = PromptEncoderV2(out_dim=ch)
        self.adapter = nn.Sequential(nn.Linear(ch, dim), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(dim, ch))
        sa_g = int(cfg.get("sa_grid", 3))
        self.sampler = ScaleAwareSampler(ch=ch, grid=sa_g)
        self.head = nn.Sequential(nn.Conv2d(ch, 128, 1), nn.GELU(), nn.Dropout(drop),
                                  nn.Conv2d(128, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, imgs, bboxes):
        B, S = imgs.shape[0], imgs.shape[-1]
        with torch.no_grad():
            taps = self.backbone(imgs)
        ps = S // self.patch
        f6, f11 = taps[0].float(), taps[1].float()
        if f6.ndim == 3:
            f6 = f6.transpose(1, 2).reshape(f6.shape[0], f6.shape[2], ps, ps)
            f11 = f11.transpose(1, 2).reshape(f11.shape[0], f11.shape[2], ps, ps)
        gate = torch.softmax(self.layer_logits, dim=0)
        z6 = self.t6_proj(f6.flatten(2).transpose(1, 2))
        z11 = self.t11_proj(f11.flatten(2).transpose(1, 2))
        tokens = gate[0] * z6 + gate[1] * z11
        prompt = self.prompt_enc(bboxes, S)
        adapted = self.adapter(torch.cat([prompt[:, None, :], tokens], dim=1))[:, 1:]
        amap = adapted.transpose(1, 2).reshape(B, adapted.shape[-1], ps, ps)
        sampled = self.sampler(amap, bboxes, S)
        mass = self.head(sampled.transpose(1, 2).reshape(B, sampled.shape[-1], ps, ps))
        return {"density": mass}


def build_model(cfg):
    return DinoScaleAware(cfg)

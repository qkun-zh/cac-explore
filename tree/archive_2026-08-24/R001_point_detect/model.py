"""R001_point_detect — paradigm shift: dense point detection, no density regression.

Frozen DINOv2-S reg4 @392 -> tokens [B,784,384] (N0010 champion substrate),
area-prompt conditioning (PromptEncoderV2) + champion adapter, then TWO heads:
  cls_head: raw objectness logits [B,1,28,28] (sigmoid applied in loss)
  reg_head: (dx,dy) pixel offset from cell center [B,2,28,28]
"""
import math

import torch
import torch.nn as nn

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


class DinoPointDetect(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        ch = 384
        dim = int(cfg.get("adapter_dim", 768))
        drop = float(cfg.get("dropout", 0.1))
        self.backbone = timm.create_model(cfg.get("backbone", BACKBONE), pretrained=True,
                                          dynamic_img_size=True, features_only=True,
                                          out_indices=(int(cfg.get("tap", 11)),))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.patch = PATCH
        self.prompt_enc = PromptEncoderV2(out_dim=ch)
        self.adapter = nn.Sequential(nn.Linear(ch, dim), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(dim, ch))
        self.cls_head = nn.Sequential(nn.Conv2d(ch, 256, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(256, 1, 1))
        self.reg_head = nn.Sequential(nn.Conv2d(ch, 256, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(256, 2, 1))
        nn.init.constant_(self.cls_head[-1].bias, -math.log((1 - 0.01) / 0.01))  # RetinaNet prior π=0.01
        nn.init.zeros_(self.reg_head[-1].weight)
        nn.init.zeros_(self.reg_head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, imgs, bboxes):
        B, S = imgs.shape[0], imgs.shape[-1]
        with torch.no_grad():
            t = self.backbone(imgs)[0].float()
        ps = S // self.patch
        if t.ndim == 3:  # [B,N,C] -> [B,C,ps,ps]
            t = t.transpose(1, 2).reshape(B, t.shape[-1], ps, ps)
        tokens = t.flatten(2).transpose(1, 2)  # [B,N,C]
        prompt = self.prompt_enc(bboxes, S)
        adapted = self.adapter(torch.cat([prompt[:, None, :], tokens], dim=1))[:, 1:]
        f = adapted.transpose(1, 2).reshape(B, adapted.shape[-1], ps, ps)
        return {"cls_logits": self.cls_head(f), "reg_offsets": self.reg_head(f)}


def build_model(cfg):
    return DinoPointDetect(cfg)

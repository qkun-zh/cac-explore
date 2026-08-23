"""N0006_swin_promptv2 — frozen Swin-Tiny + area-aware Fourier prompt + dual-scale gated fusion."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "swin_tiny_patch4_window7_224.ms_in22k"


def to_bchw(f, ch):
    if f.ndim == 4 and f.shape[1] != ch and f.shape[-1] == ch:
        f = f.permute(0, 3, 1, 2).contiguous()
    return f


class PromptEncoderV2(nn.Module):
    def __init__(self, freqs=8, hidden=256, out_dim=768):
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


class SwinPromptV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        ch2, ch3 = 384, 768
        dim = int(cfg.get("adapter_dim", 384))
        self.backbone = timm.create_model(cfg.get("backbone", BACKBONE), pretrained=True,
                                          features_only=True, out_indices=(2, 3))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.prompt_enc = PromptEncoderV2(out_dim=ch3)
        self.adapter = nn.Sequential(nn.Linear(ch3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.s2_proj = nn.Conv2d(ch2, dim, 1)
        self.gate = nn.Conv2d(2 * dim, 2, 1)
        self.head = nn.Sequential(nn.Conv2d(dim, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, imgs, bboxes):
        B, S = imgs.shape[0], imgs.shape[-1]
        with torch.no_grad():
            f2, f3 = self.backbone(imgs)
        f2 = to_bchw(f2.float(), 384)
        f3 = to_bchw(f3.float(), 768)
        tokens = f3.flatten(2).transpose(1, 2)
        prompt = self.prompt_enc(bboxes, S)
        adapted = self.adapter(torch.cat([prompt[:, None, :], tokens], dim=1))[:, 1:]
        s3_map = adapted.transpose(1, 2).reshape(B, adapted.shape[-1], *f3.shape[-2:])
        s3_up = F.interpolate(s3_map, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        s2_p = self.s2_proj(f2)
        w = torch.softmax(self.gate(torch.cat([s2_p, s3_up], dim=1)), dim=1)
        fused = w[:, 0:1] * s2_p + w[:, 1:2] * s3_up
        return {"density": self.head(fused)}


def build_model(cfg):
    return SwinPromptV2(cfg)

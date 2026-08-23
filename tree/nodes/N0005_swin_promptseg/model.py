"""N0005_swin_promptseg — frozen Swin-Tiny + exemplar Fourier prompt token, segmentation-view density."""
import math

import torch
import torch.nn as nn

BACKBONE = "swin_tiny_patch4_window7_224.ms_in22k"


class FourierPromptEncoder(nn.Module):
    def __init__(self, freqs=8, hidden=256, out_dim=768):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(freqs) * math.pi)
        self.mlp = nn.Sequential(nn.Linear(4 * freqs * 2, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, bboxes, size):
        b = bboxes / float(size)
        cxywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                             (b[:, 2] - b[:, 0]).clamp_min(0), (b[:, 3] - b[:, 1]).clamp_min(0)], dim=1)
        ang = cxywh[..., None] * self.freqs
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1))


class SwinPromptSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        dim = int(cfg.get("adapter_dim", 384))
        self.backbone = timm.create_model(cfg.get("backbone", BACKBONE), pretrained=True,
                                          features_only=True, out_indices=(3,))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        ch = self.backbone.feature_info.channels()[0]
        self.prompt_enc = FourierPromptEncoder(out_dim=ch)
        self.adapter = nn.Sequential(nn.Linear(ch, dim), nn.GELU(), nn.Linear(dim, dim))
        self.head = nn.Sequential(nn.Conv2d(dim, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, imgs, bboxes):
        B = imgs.shape[0]
        with torch.no_grad():
            feats = self.backbone(imgs)[-1].float()
        tokens = feats.flatten(2).transpose(1, 2)
        prompt = self.prompt_enc(bboxes, imgs.shape[-1])
        adapted = self.adapter(torch.cat([prompt[:, None, :], tokens], dim=1))
        mass = self.head(adapted[:, 1:].transpose(1, 2).reshape(B, adapted.shape[-1], *feats.shape[-2:]))
        return {"density": mass}


def build_model(cfg):
    return SwinPromptSeg(cfg)

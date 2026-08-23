"""N0018_dino_protoiter — champion recipe + LOCA-style iterative pseudo-prototype refinement (T=2, K=16)."""
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


class DinoPromptV2(nn.Module):
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
        self.head = nn.Sequential(nn.Conv2d(ch, 128, 1), nn.GELU(), nn.Dropout(drop),
                                  nn.Conv2d(128, 1, 1))
        self.proto_topk = int(cfg.get("proto_topk", 16))
        self.proto_iters = int(cfg.get("proto_iters", 2))
        self.proto_proj = nn.Linear(ch, ch)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _decode(self, tokens, prompt, B, ps):
        adapted = self.adapter(torch.cat([prompt[:, None, :], tokens], dim=1))[:, 1:]
        mass = self.head(adapted.transpose(1, 2).reshape(B, adapted.shape[-1], ps, ps))
        return adapted, mass

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

        adapted, mass = self._decode(tokens, prompt, B, ps)
        densities = [mass]
        for _ in range(self.proto_iters):
            scores = mass.flatten(2).squeeze(1)
            _, sel = scores.topk(self.proto_topk, dim=1)
            gather_idx = sel.unsqueeze(-1).expand(-1, -1, adapted.shape[-1])
            proto = adapted.gather(1, gather_idx).mean(dim=1)
            cond = tokens + F.gelu(self.proto_proj(proto))[:, None, :]
            adapted, mass = self._decode(cond, prompt, B, ps)
            densities.append(mass)
        density = torch.stack(densities, dim=0).mean(dim=0)
        return {"density": density}


def build_model(cfg):
    return DinoPromptV2(cfg)

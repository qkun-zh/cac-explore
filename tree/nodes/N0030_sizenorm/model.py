"""N0030_sizenorm — champion recipe + multiplicative size-conditioned output (GOD v4 seed 2).

Delta vs N0027: head emits 2 channels — density + log_size_scale; final density is
dens_raw * exp(clamp(log_scale)). Training target unchanged (engine sees final density).
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


class DinoSizeNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        ch = 384
        dim = int(cfg.get("adapter_dim", 768))
        drop = float(cfg.get("dropout", 0.15))
        self.backbone_lr_mult = float(cfg.get("backbone_lr_mult", 0.1))
        self.backbone = timm.create_model(cfg.get("backbone", BACKBONE), pretrained=True,
                                          dynamic_img_size=True, features_only=True, out_indices=(6, 11))
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.register_buffer("in_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor(std).view(1, 3, 1, 1))
        for name, p in self.backbone.named_parameters():
            if "blocks.10." in name or "blocks.11." in name or "norm." in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        self.patch = PATCH
        self.t6_proj = nn.Linear(ch, ch)
        self.t11_proj = nn.Linear(ch, ch)
        self.layer_logits = nn.Parameter(torch.zeros(2))
        self.prompt_enc = PromptEncoderV2(out_dim=ch)
        self.adapter = nn.Sequential(nn.Linear(ch, dim), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(dim, ch))
        # 2-channel head: [density_raw, log_scale]
        self.head = nn.Sequential(nn.Conv2d(ch, 128, 1), nn.GELU(), nn.Dropout(drop),
                                  nn.Conv2d(128, 2, 1))

    def param_groups(self, base_lr, weight_decay):
        bb_params, rest_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                bb_params.append(p)
            else:
                rest_params.append(p)
        return [{"params": bb_params, "lr": base_lr * self.backbone_lr_mult},
                {"params": rest_params, "lr": base_lr}]

    def forward(self, imgs, bboxes):
        B, S = imgs.shape[0], imgs.shape[-1]
        imgs = (imgs - self.in_mean) / self.in_std
        taps = self.backbone(imgs)
        ps = S // self.patch
        f6, f11 = taps[0].float(), taps[1].float()
        if f6.ndim == 3:
            f6 = f6.transpose(1, 2).reshape(B, -1, ps, ps)
            f11 = f11.transpose(1, 2).reshape(B, -1, ps, ps)
        gate = torch.softmax(self.layer_logits, dim=0)
        tokens = gate[0] * self.t6_proj(f6.flatten(2).transpose(1, 2)) + \
                 gate[1] * self.t11_proj(f11.flatten(2).transpose(1, 2))
        prompt = self.prompt_enc(bboxes, S)
        adapted = self.adapter(torch.cat([prompt[:, None, :], tokens], dim=1))[:, 1:]
        out2 = self.head(adapted.transpose(1, 2).reshape(B, -1, ps, ps))
        dens_raw = out2[:, 0:1]
        log_scale = out2[:, 1:2]
        scale = torch.exp(torch.clamp(log_scale, -5.0, 5.0))
        mass = dens_raw * scale
        return {"density": mass}


def build_model(cfg):
    return DinoSizeNorm(cfg)

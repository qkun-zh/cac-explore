"""N0024_ebc_partialft — EBC pipeline with PARTIAL backbone fine-tuning (H0032 cross-paradigm test).

Single targeted change vs parent N0022_dino_ebc_fullft:
1. Full FT -> partial FT: requires_grad=True ONLY for backbone names matching
   `blocks.10.` / `blocks.11.` / `norm.` (freeze mask copied VERBATIM from champion
   N0021_dino_partialft model.py:39-43); trainable backbone params go to base_lr x
   backbone_lr_mult via the existing param_groups().
2. config.py: backbone_lr_mult 0.05->0.1 (H0032 dose); dropout 0.15->0.1 (equalization
   vs champion, confound control).

Everything else inherited from parent unchanged: taps 6+11 + scalar gate, Fourier+area
prompt, adapter 384->768->384, EBC head over num_bins=16 count bins (paradigm="ebc").
"""
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


class DinoEBCPartialFT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        ch = 384
        dim = int(cfg.get("adapter_dim", 768))
        drop = float(cfg.get("dropout", 0.1))
        self.backbone_lr_mult = float(cfg.get("backbone_lr_mult", 0.1))
        # PARTIAL FINE-TUNING: freeze mask copied verbatim from champion N0021_dino_partialft
        self.backbone = timm.create_model(cfg.get("backbone", BACKBONE), pretrained=True,
                                          dynamic_img_size=True, features_only=True, out_indices=(6, 11))
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
        # EBC head: classify each block into count bins [0..num_bins-1]
        self.num_bins = int(cfg.get("num_bins", 16))  # max count per block
        self.ebc_head = nn.Sequential(
            nn.Conv2d(ch, 256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, self.num_bins, 1)  # logits over count bins per spatial position
        )

    def param_groups(self, base_lr, weight_decay):
        bb_params, rest_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                bb_params.append(p)
            else:
                rest_params.append(p)
        return [
            {"params": bb_params, "lr": base_lr * self.backbone_lr_mult},
            {"params": rest_params, "lr": base_lr},
        ]

    def forward(self, imgs, bboxes):
        B, S = imgs.shape[0], imgs.shape[-1]
        taps = self.backbone(imgs)
        ps = S // self.patch  # 28
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
        logits = self.ebc_head(amap)  # [B, num_bins, ps, ps]
        return {"ebc_logits": logits}


def build_model(cfg):
    model = DinoEBCPartialFT(cfg)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    bb_trainable = sum(p.numel() for n, p in model.backbone.named_parameters() if p.requires_grad) / 1e6
    print(f"[model] N0024_ebc_partialft params total={total:.2f}M trainable={trainable:.2f}M "
          f"(backbone trainable={bb_trainable:.2f}M @ lr_mult={model.backbone_lr_mult})", flush=True)
    assert total <= float(cfg.get("max_params_M", 32)), f"params {total:.2f}M over budget"
    return model

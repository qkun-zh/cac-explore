"""N0013_dino_mosaic — clone of N0010 champion + in-model photometric+bbox jitter + dropout 0.2 (mosaic-lite)."""
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


class DinoPromptV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        ch = 384
        dim = int(cfg.get("adapter_dim", 768))
        drop = float(cfg.get("dropout", 0.2))
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
        # augmentation hyperparams (config-driven, defaults per idea.md)
        self.jitter_prob = float(cfg.get("jitter_prob", 0.5))
        self.jitter_brightness = float(cfg.get("jitter_brightness", 0.2))
        self.jitter_contrast = float(cfg.get("jitter_contrast", 0.2))
        self.jitter_saturation = float(cfg.get("jitter_saturation", 0.15))
        self.jitter_noise_std = float(cfg.get("jitter_noise_std", 0.02))
        self.bbox_jitter = float(cfg.get("bbox_jitter", 0.15))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _photometric_jitter(self, imgs):
        """Training-gated photometric jitter: brightness/contrast/saturation + gaussian noise, p=0.5."""
        if not self.training or self.jitter_prob <= 0:
            return imgs
        # one global coin flip per batch to keep SMOKE deterministic-ish, but per-forward random is fine
        # Use batch-level probability; if fails, skip whole batch for efficiency
        if torch.rand(1, device=imgs.device).item() >= self.jitter_prob:
            return imgs
        B = imgs.shape[0]
        out = imgs
        # Try torchvision functional for faithful ColorJitter; fallback to manual tensor ops
        try:
            import torchvision.transforms.functional as TF
            has_tf = True
        except Exception:
            has_tf = False

        augmented = []
        for i in range(B):
            x = out[i]  # [3,S,S] in [0,1]
            bf = 1.0 + (torch.rand(1).item() * 2 - 1) * self.jitter_brightness
            cf = 1.0 + (torch.rand(1).item() * 2 - 1) * self.jitter_contrast
            sf = 1.0 + (torch.rand(1).item() * 2 - 1) * self.jitter_saturation
            # clamp factors to reasonable range
            bf = max(0.1, bf)
            cf = max(0.1, cf)
            sf = max(0.0, sf)
            if has_tf:
                try:
                    x = TF.adjust_brightness(x, bf)
                    x = TF.adjust_contrast(x, cf)
                    x = TF.adjust_saturation(x, sf)
                except Exception:
                    # manual fallback if TF fails on this shape/dtype
                    x = (x * bf).clamp(0, 1)
                    x = ((x - 0.5) * cf + 0.5).clamp(0, 1)
                    gray = 0.2989 * x[0:1] + 0.5870 * x[1:2] + 0.1140 * x[2:3]
                    gray = gray.expand_as(x)
                    x = (x * sf + gray * (1 - sf)).clamp(0, 1)
            else:
                x = (x * bf).clamp(0, 1)
                x = ((x - 0.5) * cf + 0.5).clamp(0, 1)
                gray = 0.2989 * x[0:1] + 0.5870 * x[1:2] + 0.1140 * x[2:3]
                gray = gray.expand_as(x)
                x = (x * sf + gray * (1 - sf)).clamp(0, 1)
            if self.jitter_noise_std > 0:
                x = (x + torch.randn_like(x) * self.jitter_noise_std).clamp(0, 1)
            augmented.append(x)
        return torch.stack(augmented, dim=0)

    def _bbox_jitter(self, bboxes, S):
        """Training-gated bbox jitter ±15% uniform scale/translate, clamped to [0,S]."""
        if not self.training or self.bbox_jitter <= 0:
            return bboxes
        # per-sample coin flip: only jitter subset to keep some clean prompts
        B = bboxes.shape[0]
        out = bboxes.clone()
        j = self.bbox_jitter
        for i in range(B):
            if torch.rand(1, device=bboxes.device).item() >= self.jitter_prob:
                continue
            x0, y0, x1, y1 = out[i]
            w = (x1 - x0).clamp_min(1e-4)
            h = (y1 - y0).clamp_min(1e-4)
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            # scale jitter uniform [1-j, 1+j]
            scale_w = 1.0 + (torch.rand(1, device=bboxes.device).item() * 2 - 1) * j
            scale_h = 1.0 + (torch.rand(1, device=bboxes.device).item() * 2 - 1) * j
            shift_x = (torch.rand(1, device=bboxes.device).item() * 2 - 1) * j * w
            shift_y = (torch.rand(1, device=bboxes.device).item() * 2 - 1) * j * h
            w_new = (w * scale_w).clamp_min(4.0)
            h_new = (h * scale_h).clamp_min(4.0)
            # clamp center so box stays inside [0,S]
            cx_new = (cx + shift_x).clamp(w_new / 2, S - w_new / 2)
            cy_new = (cy + shift_y).clamp(h_new / 2, S - h_new / 2)
            x0n = cx_new - w_new * 0.5
            y0n = cy_new - h_new * 0.5
            x1n = cx_new + w_new * 0.5
            y1n = cy_new + h_new * 0.5
            out[i, 0] = x0n.clamp(0, S)
            out[i, 1] = y0n.clamp(0, S)
            out[i, 2] = x1n.clamp(0, S)
            out[i, 3] = y1n.clamp(0, S)
        return out

    def forward(self, imgs, bboxes):
        B, S = imgs.shape[0], imgs.shape[-1]
        # in-model augmentation (training only, no GT warp)
        imgs = self._photometric_jitter(imgs)
        bboxes = self._bbox_jitter(bboxes, float(S))
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
        mass = self.head(adapted.transpose(1, 2).reshape(B, adapted.shape[-1], ps, ps))
        return {"density": mass}


def build_model(cfg):
    return DinoPromptV2(cfg)

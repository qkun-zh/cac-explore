"""N0028_scb_multires — SCB-lite residual exemplar gating + multi-res joint training.

Parent: N0027_norm_flip_swa (23.11M). Deltas:
- SCB-lite: pools 3 boxes on ps×ps token grid → Linear(384→384) → cosine sim vs tokens → softmax over 3 → e_ctx; tokens += γ·sigmoid(MLP([tok,e_ctx]))·e_ctx, γ=0 init ⇒ step-0 == parent
- Multi-res training is handled by engine (loaders[ep%2] alternation) and fsc147 bboxes3 emission; model is resolution-equivariant (1×1 conv head + Fourier prompt normalized by S)
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

class SCBLite(nn.Module):
    """Residual exemplar gating. γ=0 init ensures identity at step 0."""
    def __init__(self, dim=384, hidden=384):
        super().__init__()
        self.box_proj = nn.Linear(dim, dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        # temperature for softmax over boxes (sharpens assignment)
        self.sim_scale = 5.0

    def forward(self, tokens, bboxes3, S):
        # tokens: [B, N, C]  N=ps*ps, C=dim
        # bboxes3: [B, 3, 4] in S-space (x0,y0,x1,y1)
        if bboxes3 is None:
            return tokens
        B, N, C = tokens.shape
        ps = int(math.sqrt(N))
        if ps * ps != N:
            # fallback: no gating if shape mismatch
            return tokens
        # reshape to grid for pooling: [B, ps, ps, C]
        grid = tokens.view(B, ps, ps, C)
        box_feats = []
        for b in range(B):
            feats_k = []
            for k in range(3):
                x0, y0, x1, y1 = bboxes3[b, k].tolist()
                # map pixel box to token indices
                c0 = max(0, int(math.floor(x0 / PATCH)))
                c1 = min(ps, int(math.ceil(x1 / PATCH)))
                r0 = max(0, int(math.floor(y0 / PATCH)))
                r1 = min(ps, int(math.ceil(y1 / PATCH)))
                if c1 <= c0:
                    c1 = min(ps, c0 + 1)
                    c0 = max(0, c1 - 1)
                if r1 <= r0:
                    r1 = min(ps, r0 + 1)
                    r0 = max(0, r1 - 1)
                # average pool over region
                pooled = grid[b, r0:r1, c0:c1, :].mean(dim=(0, 1))  # [C]
                feats_k.append(pooled)
            feats_k = torch.stack(feats_k, dim=0)  # [3, C]
            box_feats.append(feats_k)
        box_feats = torch.stack(box_feats, dim=0)  # [B, 3, C]
        box_feats = self.box_proj(box_feats)  # [B, 3, C]

        # cosine similarity
        tok_norm = F.normalize(tokens, dim=-1)  # [B, N, C]
        box_norm = F.normalize(box_feats, dim=-1)  # [B, 3, C]
        # sim [B, N, 3]
        sim = torch.einsum('bnc,bkc->bnk', tok_norm, box_norm) * self.sim_scale
        w = F.softmax(sim, dim=-1)  # [B, N, 3]
        e_ctx = torch.einsum('bnk,bkc->bnc', w, box_feats)  # [B, N, C]

        gate = torch.sigmoid(self.gate_mlp(torch.cat([tokens, e_ctx], dim=-1)))  # [B, N, 1]
        # residual with gamma
        out = tokens + self.gamma * gate * e_ctx
        return out

class DinoSCBMultires(nn.Module):
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
        self.scb = SCBLite(dim=ch, hidden=384)
        self.adapter = nn.Sequential(nn.Linear(ch, dim), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(dim, ch))
        self.head = nn.Sequential(nn.Conv2d(ch, 128, 1), nn.GELU(), nn.Dropout(drop),
                                  nn.Conv2d(128, 1, 1))

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

    def forward(self, imgs, bboxes, bboxes3=None):
        # imgs: [B,3,S,S] in /255 space; bboxes: [B,4] first box; bboxes3: [B,3,4] all boxes (optional)
        B, S = imgs.shape[0], imgs.shape[-1]
        # infer S for bboxes3 handling: if bboxes3 is not None, its coords are in S-space of this imgs
        imgs = (imgs - self.in_mean) / self.in_std
        taps = self.backbone(imgs)
        ps = S // self.patch
        f6, f11 = taps[0].float(), taps[1].float()
        if f6.ndim == 3:
            f6 = f6.transpose(1, 2).reshape(f6.shape[0], f6.shape[2], ps, ps)
            f11 = f11.transpose(1, 2).reshape(f11.shape[0], f11.shape[2], ps, ps)
        gate = torch.softmax(self.layer_logits, dim=0)
        z6 = self.t6_proj(f6.flatten(2).transpose(1, 2))
        z11 = self.t11_proj(f11.flatten(2).transpose(1, 2))
        tokens = gate[0] * z6 + gate[1] * z11  # [B, N, C]
        # SCB-lite gating (handles bboxes3=None → identity; gamma=0 → identity at init)
        # bboxes3 may be None during smoke or old eval
        if bboxes3 is not None and not isinstance(bboxes3, torch.Tensor):
            # handle case where engine passes unexpected type
            bboxes3 = None
        tokens = self.scb(tokens, bboxes3, S)
        prompt = self.prompt_enc(bboxes, S)
        adapted = self.adapter(torch.cat([prompt[:, None, :], tokens], dim=1))[:, 1:]
        mass = self.head(adapted.transpose(1, 2).reshape(B, adapted.shape[-1], ps, ps))
        return {"density": mass}

def build_model(cfg):
    return DinoSCBMultires(cfg)

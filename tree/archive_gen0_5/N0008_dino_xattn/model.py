"""N0008_dino_xattn — frozen DINOv2-S tokens + exemplar-conditioned cross-attention basis mixture."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "vit_small_patch14_reg4_dinov2.lvd142m"


class ExemplarTokenV2(nn.Module):
    def __init__(self, ch=384, dim=256, freqs=8, hidden=256):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(freqs) * math.pi)
        self.appearance = nn.Linear(ch, dim)
        self.prompt_mlp = nn.Sequential(nn.Linear(4 * freqs * 2 + 1, hidden), nn.GELU(),
                                        nn.Linear(hidden, dim))
        self.fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())

    def forward(self, tokens, bboxes, size, ps):
        dev = tokens.device
        ys, xs = torch.meshgrid(torch.arange(ps, device=dev), torch.arange(ps, device=dev), indexing="ij")
        cx, cy = (xs + 0.5) * (size / ps), (ys + 0.5) * (size / ps)
        inside = ((cx[None] >= bboxes[:, 0, None, None]) & (cy[None] >= bboxes[:, 1, None, None]) &
                  (cx[None] <= bboxes[:, 2, None, None]) & (cy[None] <= bboxes[:, 3, None, None]))
        m = inside.reshape(inside.shape[0], -1).float()
        m = m / m.sum(-1, keepdim=True).clamp_min(1.0)
        app = self.appearance(torch.einsum("bp,bpc->bc", m.to(tokens.dtype), tokens))
        b = bboxes / float(size)
        w = (b[:, 2] - b[:, 0]).clamp_min(1e-4)
        h = (b[:, 3] - b[:, 1]).clamp_min(1e-4)
        cxywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, w, h], dim=1)
        ang = cxywh[..., None] * self.freqs
        fourier = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)
        log_area = torch.log(w * h).unsqueeze(1).clamp(-13.8, 0.0)
        scale = self.prompt_mlp(torch.cat([fourier, log_area], dim=1))
        return self.fuse(torch.cat([app, scale], dim=1))


class DinoXAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        ch = 384
        dim = int(cfg.get("dec_dim", 256))
        K = int(cfg.get("queries", 8))
        drop = float(cfg.get("dropout", 0.1))
        self.backbone = timm.create_model(cfg.get("backbone", BACKBONE), pretrained=True,
                                          num_classes=0, dynamic_img_size=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.patch = int(self.backbone.patch_embed.patch_size[0])
        self.queries = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.mem_proj = nn.Linear(ch, dim)
        self.extok = ExemplarTokenV2(ch=ch, dim=dim)
        layer = nn.TransformerDecoderLayer(dim, nhead=4, dim_feedforward=512,
                                           batch_first=True, norm_first=True, dropout=drop)
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(cfg.get("dec_layers", 2)))
        self.basis = nn.Linear(dim, ch)
        nn.init.normal_(self.basis.weight, std=0.01)
        nn.init.zeros_(self.basis.bias)
        self.mix = nn.Linear(dim, 1)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, imgs, bboxes):
        B, S = imgs.shape[0], imgs.shape[-1]
        with torch.no_grad():
            toks = self.backbone.forward_features(imgs)
        ps = S // self.patch
        tokens = toks[:, -ps * ps:, :].float()
        extok = self.extok(tokens, bboxes, S, ps)
        mem = self.mem_proj(tokens) + extok[:, None, :]
        q = torch.cat([self.queries[None].expand(B, -1, -1), extok[:, None, :]], dim=1)
        out = self.decoder(q, mem)
        basis_maps = self.basis(out[:, :-1])
        wq = torch.softmax(self.mix(out[:, :-1]).squeeze(-1), dim=-1)
        maps = torch.einsum("bkc,bpc->bkp", basis_maps, F.normalize(tokens, dim=-1))
        maps = maps.reshape(B, -1, ps, ps)
        dens = (wq[..., None, None] * maps).sum(1, keepdim=True)
        return {"density": dens}


def build_model(cfg):
    return DinoXAttn(cfg)

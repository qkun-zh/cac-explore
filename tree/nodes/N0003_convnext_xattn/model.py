"""N0003_convnext_xattn — frozen ConvNeXt-Nano FPN + exemplar-conditioned cross-attention density bases."""
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "convnext_nano.in12k"


class FPN(nn.Module):
    def __init__(self, c3, c4, c5, ch=128):
        super().__init__()
        self.l3 = nn.Conv2d(c3, ch, 1)
        self.l4 = nn.Conv2d(c4, ch, 1)
        self.l5 = nn.Conv2d(c5, ch, 1)
        self.smooth = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x3, x4, x5):
        p5 = self.l5(x5)
        p4 = self.l4(x4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.l3(x3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        return self.smooth(p3)


class XAttnCountHead(nn.Module):
    def __init__(self, ch=128, dim=256, queries=8, layers=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.mem_proj = nn.Linear(ch, dim)
        self.ex_proj = nn.Linear(ch, dim)
        dec_layer = nn.TransformerDecoderLayer(dim, nhead=4, dim_feedforward=512,
                                               batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=layers)
        self.basis = nn.Linear(dim, ch)
        nn.init.normal_(self.basis.weight, std=0.01)
        nn.init.zeros_(self.basis.bias)
        self.mix = nn.Linear(dim, 1)

    def forward(self, fmap, bboxes):
        B, C, h, w = fmap.shape
        ys, xs = torch.meshgrid(torch.arange(h, device=fmap.device),
                                torch.arange(w, device=fmap.device), indexing="ij")
        cx, cy = (xs + 0.5) * 8.0, (ys + 0.5) * 8.0
        inside = ((cx[None] >= bboxes[:, 0, None, None]) & (cy[None] >= bboxes[:, 1, None, None]) &
                  (cx[None] <= bboxes[:, 2, None, None]) & (cy[None] <= bboxes[:, 3, None, None]))
        m = inside.reshape(B, -1).float()
        m = m / m.sum(-1, keepdim=True).clamp_min(1.0)
        feat = fmap.flatten(2).transpose(1, 2)
        ex_tok = self.ex_proj(torch.einsum("bp,bpc->bc", m.to(feat.dtype), feat))
        q = torch.cat([self.queries[None].expand(B, -1, -1), ex_tok[:, None, :]], dim=1)
        mem = self.mem_proj(F.avg_pool2d(fmap, 2).flatten(2).transpose(1, 2)) + ex_tok[:, None, :]
        out = self.decoder(q, mem)
        basis_maps = self.basis(out[:, :-1])
        wq = torch.softmax(self.mix(out[:, :-1]).squeeze(-1), dim=-1)
        maps = torch.einsum("bkc,bchw->bkhw", basis_maps, fmap)
        return (wq[..., None, None] * maps).sum(1, keepdim=True)


class ConvNeXtXAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        ch = int(cfg.get("fpn_ch", 128))
        self.backbone = timm.create_model(BACKBONE, pretrained=True, features_only=True,
                                          out_indices=(1, 2, 3))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        c3, c4, c5 = self.backbone.feature_info.channels()
        self.fpn = FPN(c3, c4, c5, ch)
        self.head = XAttnCountHead(ch=ch, dim=int(cfg.get("dec_dim", 256)),
                                   queries=int(cfg.get("queries", 8)),
                                   layers=int(cfg.get("dec_layers", 2)))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, imgs, bboxes):
        with torch.no_grad():
            x3, x4, x5 = self.backbone(imgs)
        fmap = self.fpn(x3.float(), x4.float(), x5.float())
        return {"density": self.head(fmap, bboxes)}


def build_model(cfg):
    return ConvNeXtXAttn(cfg)

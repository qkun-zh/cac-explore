"""OPE prototype extractor — vendored from LOCA (ICCV 2023, MIT license).
Source: https://github.com/djukicn/loca (models/ope.py, mlp.py, positional_encoding.py)

Separates exemplar SHAPE queries (w/h -> learned embedding) from APPEARANCE queries
(RoI-pooled features), then iteratively adapts them into object prototypes via
cross-attention against image-wide features. Output prototypes serve as conv kernels
for response maps.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align


class PositionalEncodingsFixed(nn.Module):
    def __init__(self, emb_dim, temperature=10000):
        super().__init__()
        self.emb_dim = emb_dim
        self.temperature = temperature

    def _1d_pos_enc(self, mask, dim):
        temp = torch.arange(self.emb_dim // 2).float().to(mask.device)
        temp = self.temperature ** (2 * (temp.div(2, rounding_mode="floor")) / self.emb_dim)
        enc = (~mask).cumsum(dim).float().unsqueeze(-1) / temp
        enc = torch.stack([enc[..., 0::2].sin(), enc[..., 1::2].cos()], dim=-1).flatten(-2)
        return enc

    def forward(self, bs, h, w, device):
        mask = torch.zeros(bs, h, w, dtype=torch.bool, requires_grad=False, device=device)
        x = self._1d_pos_enc(mask, dim=2)
        y = self._1d_pos_enc(mask, dim=1)
        return torch.cat([y, x], dim=3).permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, activation):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class IterativeAdaptationLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, layer_norm_eps, mlp_factor, norm_first, activation, zero_shot=False):
        super().__init__()
        self.norm_first = norm_first
        self.zero_shot = zero_shot
        if not zero_shot:
            self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
            self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
            self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout3 = nn.Dropout(dropout)
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(self, tgt, appearance, memory, pos_emb, query_pos_emb,
                tgt_mask=None, memory_mask=None, tgt_kpm=None, mem_kpm=None):
        if self.norm_first:
            if not self.zero_shot:
                tn = self.norm1(tgt)
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tn, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance, attn_mask=tgt_mask, key_padding_mask=tgt_kpm)[0])
            tn = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tn, query_pos_emb),
                key=memory + pos_emb, value=memory,
                attn_mask=memory_mask, key_padding_mask=mem_kpm)[0])
            tn = self.norm3(tgt)
            tgt = tgt + self.dropout3(self.mlp(tn))
        else:
            if not self.zero_shot:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb), value=appearance,
                    attn_mask=tgt_mask, key_padding_mask=tgt_kpm)[0]))
            tgt = self.norm2(tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt, query_pos_emb),
                key=memory + pos_emb, value=memory,
                attn_mask=memory_mask, key_padding_mask=mem_kpm)[0]))
            tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        return tgt


class IterativeAdaptationModule(nn.Module):
    def __init__(self, num_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                 mlp_factor, norm_first, activation, norm=True, zero_shot=False):
        super().__init__()
        self.layers = nn.ModuleList([
            IterativeAdaptationLayer(emb_dim, num_heads, dropout, layer_norm_eps,
                                     mlp_factor, norm_first, activation, zero_shot)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(self, tgt, appearance, memory, pos_emb, query_pos_emb,
                tgt_mask=None, memory_mask=None, tgt_kpm=None, mem_kpm=None):
        out = tgt
        outputs = []
        for layer in self.layers:
            out = layer(out, appearance, memory, pos_emb, query_pos_emb,
                        tgt_mask, memory_mask, tgt_kpm, mem_kpm)
            outputs.append(self.norm(out))
        return torch.stack(outputs)


class OPEModule(nn.Module):
    """Object Prototype Extraction (LOCA). f_e [B,d,H,W], bboxes [B,n,4] in f_e pixel coords."""
    def __init__(self, num_iterative_steps, emb_dim, kernel_dim, num_objects, num_heads,
                 reduction, layer_norm_eps=1e-5, mlp_factor=4, norm_first=False,
                 activation=nn.ReLU, norm=True, mlp_dropout=0.0):
        super().__init__()
        self.num_iterative_steps = num_iterative_steps
        self.kernel_dim = kernel_dim
        self.num_objects = num_objects
        self.emb_dim = emb_dim
        self.reduction = reduction
        self.iterative_adaptation = IterativeAdaptationModule(
            num_layers=num_iterative_steps, emb_dim=emb_dim, num_heads=num_heads,
            dropout=0.0, layer_norm_eps=layer_norm_eps, mlp_factor=mlp_factor,
            norm_first=norm_first, activation=activation, norm=norm, zero_shot=False)
        self.shape_or_objectness = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, kernel_dim ** 2 * emb_dim))
        self.pos_emb = PositionalEncodingsFixed(emb_dim)

    def forward(self, f_e, pos_emb, bboxes):
        bs, _, h, w = f_e.size()
        box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2, device=bboxes.device)
        box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
        box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]
        shape_q = self.shape_or_objectness(box_hw).reshape(
            bs, -1, self.kernel_dim ** 2, self.emb_dim).flatten(1, 2).transpose(0, 1)
        bboxes_roi = torch.cat([
            torch.arange(bs, device=bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1),
            bboxes.flatten(0, 1)], dim=1)
        appearance = roi_align(
            f_e, boxes=bboxes_roi, output_size=self.kernel_dim,
            spatial_scale=1.0 / self.reduction, aligned=True
        ).permute(0, 2, 3, 1).reshape(bs, self.num_objects * self.kernel_dim ** 2, -1).transpose(0, 1)
        qpe = self.pos_emb(bs, self.kernel_dim, self.kernel_dim, f_e.device
                           ).flatten(2).permute(2, 0, 1).repeat(self.num_objects, 1, 1)
        memory = f_e.flatten(2).permute(2, 0, 1)
        all_prototypes = self.iterative_adaptation(shape_q, appearance, memory, pos_emb, qpe)
        return all_prototypes  # [L, k^2*n, B, d]


def ope_response_maps(f_e, prototypes_last, kernel_dim, num_objects):
    """Depth-wise correlation of prototypes against feature map -> response maps [B, d, H, W]."""
    bs, d, H, W = f_e.shape
    outs = []
    for i in range(prototypes_last.size(0)):
        protos = prototypes_last[i].permute(1, 0, 2).reshape(
            bs, num_objects, kernel_dim, kernel_dim, -1).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]
        rm = F.conv2d(
            torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
            protos, bias=None, padding=kernel_dim // 2, groups=protos.size(0))
        outs.append(rm.view(bs, num_objects, d, H, W).max(dim=1)[0])
    return outs[-1]

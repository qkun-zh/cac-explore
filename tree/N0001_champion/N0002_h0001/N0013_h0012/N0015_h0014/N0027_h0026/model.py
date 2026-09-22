"""Champion architecture — N0054_xscale_exemplar lineage (canonical seed root).

Frozen DINOv3-ConvNeXt-Tiny + pluggable CountingHead: FineFuser → ExemplarEncoder
(transformer, XScale coarse multi-scale exemplar summary) → cross-attn Condenser →
DensityDecoder, plus GCA global-count aux. Every evolution step must build ON this
interface: `build_model(cfg)` → forward(imgs, bboxes[, bboxes3]) → {"density", "n_aux"}.
Only `out["density"]` feeds the loss.

The frozen-backbone / head-only pluggable regime is the invariant of this project;
see tree/N0001_champion/idea.md and docs/research_direction.md.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def _get(cfg, k, d=None):
    return cfg[k] if isinstance(cfg, dict) and k in cfg else (getattr(cfg, k, d) if hasattr(cfg, k) else d)


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel
        try:
            from cac.hub import hf_token
            tok = hf_token()
        except Exception:
            tok = None
        name = _get(cfg, "hf_model", "facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
        self.net = AutoModel.from_pretrained(name, token=tok, trust_remote_code=True)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        dims = _get(cfg, "backbone_dims", (192, 384))
        self.out_channels = list(dims)
        self.hs_map = (2, 3)

    @torch.no_grad()
    def forward_feature_map(self, x):
        hs = self.net(pixel_values=x, output_hidden_states=True).hidden_states
        return [hs[i] for i in self.hs_map]

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()
        return self


class FineFuser(nn.Module):
    """Component: fuse coarse(mid) + mid(fine) backbone features -> 1/4-res dense feature.
    Optional DDCA = parallel dilated dw branch (use_ddca). Self-contained; no external coupling."""
    def __init__(self, ch_coarse, ch_mid, d_fine=128, use_ddca=True):
        super().__init__()
        self.use_ddca = use_ddca
        self.top = nn.Sequential(nn.Conv2d(ch_coarse, d_fine, 1), nn.GroupNorm(8, d_fine))
        self.lat = nn.Sequential(nn.Conv2d(ch_mid, d_fine, 1), nn.GroupNorm(8, d_fine))
        self.fuse = nn.Sequential(nn.Conv2d(2 * d_fine, d_fine, 3, padding=1), nn.GroupNorm(8, d_fine), nn.GELU())
        self.refine = nn.Conv2d(d_fine, d_fine, 3, padding=1, groups=d_fine)
        if use_ddca:
            self.ctx = nn.Conv2d(d_fine, d_fine, 3, padding=2, dilation=2, groups=d_fine)
            nn.init.zeros_(self.ctx.weight)
            if self.ctx.bias is not None:
                nn.init.zeros_(self.ctx.bias)

    def forward(self, h2, h3):
        top = F.interpolate(self.top(h3), scale_factor=2, mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([self.lat(h2), top], 1))
        if self.use_ddca:
            f = F.gelu(self.refine(f) + f + self.ctx(f))
        else:
            f = F.gelu(self.refine(f) + f)
        return F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)


class ExemplarEncoder(nn.Module):
    def __init__(self, in_dim=384, d_model=256, n_layers=2, n_heads=4, roi_size=7, use_xscale=False, xs=3):
        super().__init__()
        self.r = roi_size
        self.use_xscale = use_xscale
        self.proj = nn.Linear(in_dim, d_model)
        self.shape_mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.attn = nn.Linear(d_model, 1)
        if use_xscale:
            self.xs = xs
            self.xproj = nn.Linear(in_dim, d_model)

    def forward(self, feat, bboxes, img_size):
        B, C, H, W = feat.shape
        K = bboxes.shape[1]
        s = W / float(img_size)
        idx = torch.arange(B, device=bboxes.device, dtype=bboxes.dtype).view(B, 1, 1).expand(B, K, 1)
        rois = torch.cat([idx, bboxes * s], -1).reshape(B * K, 5)
        roi = roi_align(feat, rois, output_size=(self.r, self.r))
        tok = self.proj(roi.flatten(2).transpose(1, 2))
        wh = (bboxes[:, :, 2:4] - bboxes[:, :, :2]).clamp_min(1.)
        tok = (tok.view(B, K, self.r * self.r, -1) + self.shape_mlp(wh).unsqueeze(2)).reshape(B * K, self.r * self.r, -1)
        tok = self.tr(tok)
        a = self.attn(tok).softmax(1)
        out = (tok * a).sum(1).view(B, K, -1)
        if self.use_xscale:
            roi2 = roi_align(feat, rois, output_size=(self.xs, self.xs))
            coarse = F.adaptive_avg_pool2d(roi2, (1, 1)).squeeze(-1).squeeze(-1)
            out = out + self.xproj(coarse).view(B, K, -1)
        return out


class Condenser(nn.Module):
    def __init__(self, d_in=128, d_sim=256, n_heads=4, ff=512, d_out=64):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_sim)
        self.attn = nn.MultiheadAttention(d_sim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_sim)
        self.norm2 = nn.LayerNorm(d_sim)
        self.ffn = nn.Sequential(nn.Linear(d_sim, ff), nn.GELU(), nn.Linear(ff, d_sim))
        self.out = nn.Linear(d_sim, d_out)

    def forward(self, tok, e):
        tok = self.proj_in(tok)
        a, _ = self.attn(self.norm1(tok), e, e, need_weights=False)
        q = self.norm1(tok + a)
        return self.out(self.norm2(q + self.ffn(q)))


class DensityDecoder(nn.Module):
    def __init__(self, in_ch, hidden=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=2, dilation=2), nn.GroupNorm(4, hidden // 2), nn.GELU())
        self.head = nn.Conv2d(hidden // 2, 1, 1)
        for m in [self.block[0], self.block[3]]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return F.softplus(self.head(self.block(x)))


class CountingHead(nn.Module):
    """Pluggable trunk: bundles FineFuser + ExemplarEncoder + Condenser + DensityDecoder
    as ONE module. Emits density and exposes fine/e_mean as shared interfaces for GCA, etc."""
    def __init__(self, cfg):
        super().__init__()
        D = _get(cfg, "d_fine", 128)
        dims = _get(cfg, "backbone_dims", (192, 384))
        use_ddca = _get(cfg, "use_ddca", True)
        self.fuser = FineFuser(dims[1], dims[0], d_fine=D, use_ddca=use_ddca)
        self.exemplar = ExemplarEncoder(in_dim=dims[1], d_model=_get(cfg, "embed_dim", 256),
                                        n_layers=_get(cfg, "exemplar_layers", 2), roi_size=_get(cfg, "roi_size", 7),
                                        use_xscale=_get(cfg, "use_xscale", False), xs=_get(cfg, "xscale_size", 3))
        self.cond = Condenser(d_in=D, d_sim=_get(cfg, "embed_dim", 256), d_out=_get(cfg, "cond_dim", 64))
        self.decoder = DensityDecoder(in_ch=D + _get(cfg, "cond_dim", 64), hidden=2 * D)
        self.S = _get(cfg, "input_size", 384)
        # H0001 switch (non-module flag: no RNG consumption, safe before SimPrior wiring).
        self.use_simprior = _get(cfg, "use_simprior", False)
        # H0012 switch (non-module flag: no RNG consumption).
        self.use_cellcal = _get(cfg, "use_cellcal", False)
        # H0014 switch (non-module flag: no RNG consumption).
        self.use_peakcal = _get(cfg, "use_peakcal", False)
        # H0026 switch (non-module flag: no RNG consumption).
        self.use_geme = _get(cfg, "use_geme", False)

    def forward(self, h2, h3, bboxes_in):
        B = bboxes_in.shape[0]
        fine = self.fuser(h2, h3)
        Hf = Wf = self.S // 4
        fmap = fine.permute(0, 2, 3, 1).flatten(1, 2)
        e = self.exemplar(h3, bboxes_in, self.S)
        cond = self.cond(fmap, e)
        cond_map = cond.transpose(1, 2).reshape(B, -1, Hf, Wf)
        # H0026 (use_geme): per-image geometry magnitude factor on SimPrior evidence.
        # Flag off -> geme_f stays None and every executed line is the parent's.
        geme_f = None
        if self.use_geme and getattr(self, "gemecal", None) is not None:
            geme_f = self.gemecal(bboxes_in, self.S)
        # H0001 (use_simprior): additive dense similarity-prior residual into cond.
        # Flag off -> line below is skipped and forward is the parent's byte-for-byte.
        if self.use_simprior:
            if self.use_cellcal:
                if self.use_peakcal:
                    cond_map = cond_map + self.simprior(fine, e, use_cellcal=True, w_c=self.cellcal.w_c, use_peakcal=True, w_p=self.peakcal.w_p, geme_f=geme_f)
                else:
                    cond_map = cond_map + self.simprior(fine, e, use_cellcal=True, w_c=self.cellcal.w_c, geme_f=geme_f)
            else:
                if self.use_peakcal:
                    cond_map = cond_map + self.simprior(fine, e, use_peakcal=True, w_p=self.peakcal.w_p, geme_f=geme_f)
                else:
                    cond_map = cond_map + self.simprior(fine, e, geme_f=geme_f)
        dens = self.decoder(torch.cat([fine, cond_map], 1))
        e_mean = e.mean(dim=1)
        return dens, fine, e_mean


class GCA(nn.Module):
    """Independent pluggable aux: global count head. Reads GAP(fine) + e_mean only."""
    def __init__(self, D, d_model):
        super().__init__()
        self.gca = nn.Sequential(nn.Linear(D + d_model, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.gca[-1].weight)
        nn.init.zeros_(self.gca[-1].bias)

    def forward(self, fine, e_mean, Hf, Wf):
        gap = fine.mean(dim=(2, 3))
        n_aux = F.softplus(self.gca(torch.cat([gap, e_mean], 1))).squeeze(1)
        bias = (n_aux / float(Hf * Wf)).view(-1, 1, 1, 1) * 0.02
        return n_aux, bias


class SimPrior(nn.Module):
    """H0001 dense exemplar-similarity prior readout (`use_simprior`).

    Explicit per-cell cosine match evidence over the K=3 pooled exemplar
    embeddings, compressed through a zero-init 1x1 projection added
    residually into `cond`. Pure additive readout at the decoder interface:
    decoder in_ch stays 192, no parent module shape/init is touched, `e`
    and `fine` are read-only (never gated/rescaled). Zero-init `out` makes
    step-0 forward numerically identical to the parent.
    """
    def __init__(self, d_fine=128, d_model=256, d_proj=64, cond_dim=64, n_ev=2):
        super().__init__()
        self.qproj = nn.Conv2d(d_fine, d_proj, 1, bias=False)   # 128 -> 64
        self.kproj = nn.Linear(d_model, d_proj, bias=False)     # 256 -> 64
        self.temp = nn.Parameter(torch.tensor(0.07))            # learnable temperature (divisor)
        self.out = nn.Conv2d(n_ev, cond_dim, 1)                 # 2 -> 64, ZERO-INIT
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, fine, e, use_cellcal=False, w_c=None, use_peakcal=False, w_p=None, geme_f=None):
        # fine: (B,128,96,96); e: (B,K=3,256)
        q = F.normalize(self.qproj(fine), dim=1)                # (B,64,96,96)
        p = F.normalize(self.kproj(e), dim=-1)                  # (B,K,64)
        S = torch.einsum('bchw,bkc->bkhw', q, p)                # (B,K,96,96) cosine in [-1,1]
        W = (S / self.temp.detach().clamp_min(1e-3)).softmax(dim=1)  # softmax over K
        top1 = S.max(dim=1, keepdim=True).values                # (B,1,96,96)
        cons = (W * S).sum(dim=1, keepdim=True)                 # (B,1,96,96) softmax-weighted consensus
        ev = torch.cat([top1, cons], dim=1)                     # (B,2,96,96)
        if use_cellcal:
            m = fine.detach().mean(dim=1, keepdim=True).clamp_min(0)   # (B,1,96,96)
            mbar = m.mean(dim=(2, 3), keepdim=True)                    # (B,1,1,1)
            mhat = m / (mbar + 1e-6)                                   # mean 1.0 by construction
            gain = torch.exp(w_c * torch.log1p(mhat))                  # (B,1,96,96); w_c=0 -> all ones
            ev = ev * gain                                             # broadcast over the 2 evidence channels
        if use_peakcal:
            m = fine.detach().mean(dim=1, keepdim=True).clamp_min(0)   # (B,1,96,96)
            mbar = m.mean(dim=(2, 3), keepdim=True)                    # (B,1,1,1)
            mhat = (m / (mbar + 1e-6)).clamp(0, 10)                    # mean ~1 by construction
            loc = F.avg_pool2d(mhat, 3, stride=1, padding=1)           # (B,1,96,96) local mean
            peak = (mhat / (loc + 1e-4)).clamp(0, 5)                   # >1 peaked center, ~=1 diffuse flat
            elev = (mhat - 1).clamp(0, 5)                              # elevated-only (ReLU via clamp)
            flat = torch.exp(-((peak - 1).square()) / 0.125).nan_to_num(0.0, 1.0, 1.0)  # sigma=0.25 fixed -> 2s^2=0.125; ~1 iff peak~=1
            f = (elev * flat).clamp(0, 5)                              # (B,1,96,96) pure-data diffuse field
            arg = (w_p * f).clamp(-3, 3)
            gain = torch.exp(arg).clamp(1 - 0.3, 1).nan_to_num(1.0, 1.0, 0.7)  # D=0.3 FIXED; NaN falls back to identity
            ev = ev * gain                                             # broadcast over the 2 evidence channels
        if geme_f is not None:
            ev = ev * geme_f.view(-1, 1, 1, 1)                    # H0026 post-gain, pre-out; f==1 at step-0
        return self.out(ev)                                     # (B,64,96,96), all-zeros at init


class CellCal(nn.Module):
    """H0012 per-cell self-normalized energy gain field (`use_cellcal`).

    Scalar-only pluggable module: ONE zero-init scalar `w_c`. The gain math
    itself lives in `SimPrior.forward` (gated on the head flag
    `use_cellcal`); this module only owns `w_c` so the head flag stays a
    non-module switch like `use_simprior`. Zero-init draws no RNG, so
    append-only construction keeps parent init draws identical.
    """
    def __init__(self):
        super().__init__()
        self.w_c = nn.Parameter(torch.zeros(()))


class PeakCal(nn.Module):
    """H0014 energy-peakiness-gated per-cell discount (`use_peakcal`).

    Scalar-only pluggable module: ONE zero-init scalar `w_p`. The gain math
    itself lives in `SimPrior.forward` (gated on the head flag
    `use_peakcal`); this module only owns `w_p` so the head flag stays a
    non-module switch like `use_simprior`/`use_cellcal`. Zero-init draws no
    RNG, so append-only construction keeps parent init draws identical. NO
    normalization layers (train/eval agree exactly).
    """
    def __init__(self):
        super().__init__()
        self.w_p = nn.Parameter(torch.zeros(()))


class GemeCal(nn.Module):
    """H0026 geometry magnitude embedding scale: zero-init w_g -> f = (me/32)^w_g.
    ME = mean_K(S^2 / (w_k * h_k)) from annotation boxes (CACViT geometry).
    torch.zeros(()) draws no RNG; step-0 w_g=0 => f=1 exact.
    """
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.zeros(()))

    def forward(self, bboxes, img_size):
        wh = (bboxes[:, :, 2:4] - bboxes[:, :, :2]).clamp_min(1.0)
        area = (wh[..., 0] * wh[..., 1]).clamp_min(1.0)          # (B, K)
        me = (float(img_size) * float(img_size) / area).mean(dim=1)  # (B,) capacity prior
        me = me.clamp(1.0, 1.0e4)
        f = torch.exp(self.w_g * (torch.log(me) - math.log(32.0))).clamp(0.25, 4.0)
        return f                                                  # (B,), == 1 at init


class Counter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.S = _get(cfg, "input_size", 384)
        dims = _get(cfg, "backbone_dims", (192, 384))
        D = _get(cfg, "d_fine", 128)
        self.use_gca = _get(cfg, "use_gca", True)
        self.backbone = Backbone(cfg)
        self.head = CountingHead(cfg)
        if self.use_gca:
            self.gca = GCA(D, _get(cfg, "embed_dim", 256))
        # H0001: SimPrior is constructed AFTER every parent module (append-only
        # RNG order, AGENTS rule 13) and attached to the head, so all parent
        # init draws keep their exact RNG positions; only SimPrior's own
        # qproj/kproj init draws are appended. Forward gating lives in
        # CountingHead.forward via self.use_simprior.
        self.head.simprior = SimPrior(d_fine=D, d_model=_get(cfg, "embed_dim", 256),
                                      cond_dim=_get(cfg, "cond_dim", 64))
        # H0012: CellCal scalar is constructed LAST, after the SimPrior attach
        # (append-only RNG order, AGENTS rule 13). Zero-init draws no RNG, so
        # parent init draws keep their exact RNG positions. Forward gating
        # lives in CountingHead.forward via self.use_cellcal.
        self.head.cellcal = CellCal()
        # H0014: PeakCal scalar is constructed LAST, after the CellCal attach
        # (append-only RNG order, AGENTS rule 13). Zero-init draws no RNG, so
        # parent init draws keep their exact RNG positions. Forward gating
        # lives in CountingHead.forward via self.use_peakcal.
        self.head.peakcal = PeakCal()
        # H0026: GemeCal scalar is constructed AFTER every parent module
        # (append-only RNG order, AGENTS rule 13). torch.zeros(()) draws no
        # RNG, so parent init draws keep their exact RNG positions. Forward
        # gating lives in CountingHead.forward via self.use_geme.
        if self.head.use_geme:
            self.head.gemecal = GemeCal()
        # Temp-pin hygiene: when use_peakcal is on, pin simprior temp
        # (value-identical detach in forward closes the free-temp path).
        if self.head.use_peakcal:
            self.head.simprior.temp.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, imgs, bboxes, bboxes3=None):
        if bboxes.dim() == 2:
            bboxes = bboxes.unsqueeze(1)
        bboxes_in = bboxes3 if bboxes3 is not None else bboxes
        h2, h3 = self.backbone.forward_feature_map(imgs)
        dens, fine, e_mean = self.head(h2, h3, bboxes_in)
        B = imgs.shape[0]
        Hf = Wf = self.S // 4
        out = {"density": dens}
        if self.use_gca:
            n_aux, bias = self.gca(fine, e_mean, Hf, Wf)
            out["density"] = dens + bias
            out["n_aux"] = n_aux
        return out


def build_model(cfg):
    return Counter(cfg)

"""Champion architecture — N0054_xscale_exemplar lineage (canonical seed root).

Frozen DINOv3-ConvNeXt-Tiny + pluggable CountingHead: FineFuser → ExemplarEncoder
(transformer, XScale coarse multi-scale exemplar summary) → cross-attn Condenser →
DensityDecoder, plus GCA global-count aux. Every evolution step must build ON this
interface: `build_model(cfg)` → forward(imgs, bboxes[, bboxes3]) → {"density", "n_aux"}.
Only `out["density"]` feeds the loss.

The frozen-backbone / head-only pluggable regime is the invariant of this project;
see tree/N0001_champion/idea.md and docs/research_direction.md.
"""
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
        return F.softplus(self.forward_logits(x))

    def forward_logits(self, x):
        return self.head(self.block(x))


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
        # H0008 switch (non-module flag: no RNG consumption).
        self.use_hires = _get(cfg, "use_hires", False)

    def forward(self, h2, h3, bboxes_in):
        B = bboxes_in.shape[0]
        fine = self.fuser(h2, h3)
        Hf = Wf = self.S // 4
        fmap = fine.permute(0, 2, 3, 1).flatten(1, 2)
        e = self.exemplar(h3, bboxes_in, self.S)
        cond = self.cond(fmap, e)
        cond_map = cond.transpose(1, 2).reshape(B, -1, Hf, Wf)
        # H0001 (use_simprior): additive dense similarity-prior residual into cond.
        # Flag off -> line below is skipped and forward is the parent's byte-for-byte.
        if self.use_simprior:
            cond_map = cond_map + self.simprior(fine, e)
        feat = torch.cat([fine, cond_map], 1)
        logits = self.decoder.forward_logits(feat)
        # H0008 (use_hires): zero-init post-decoder detail residual on detached
        # logits + detached frozen-h2 guide. Flag off -> softplus(logits) which
        # is op-identical to the parent decoder forward (byte-for-byte).
        if self.use_hires:
            r = self.hires(logits.detach(), h2.detach())
            dens = F.softplus(logits + r)
        else:
            dens = F.softplus(logits)
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

    def forward(self, fine, e):
        # fine: (B,128,96,96); e: (B,K=3,256)
        q = F.normalize(self.qproj(fine), dim=1)                # (B,64,96,96)
        p = F.normalize(self.kproj(e), dim=-1)                  # (B,K,64)
        S = torch.einsum('bchw,bkc->bkhw', q, p)                # (B,K,96,96) cosine in [-1,1]
        W = (S / self.temp.clamp_min(1e-3)).softmax(dim=1)      # softmax over K
        top1 = S.max(dim=1, keepdim=True).values                # (B,1,96,96)
        cons = (W * S).sum(dim=1, keepdim=True)                 # (B,1,96,96) softmax-weighted consensus
        ev = torch.cat([top1, cons], dim=1)                     # (B,2,96,96)
        return self.out(ev)                                     # (B,64,96,96), all-zeros at init


class HiResDetail(nn.Module):
    """H0008 post-decoder detail residual (`use_hires`).

    Zero-init additive high-frequency correction on the pre-softplus decoder
    logits, guided by read-only frozen h2 context. Inputs are both
    stop-gradient detached at routing (bare logits L + h2 guide), so no second
    gradient path touches the confirmed simprior readout or any shared
    projection. Zero-init `head` makes step-0 forward numerically identical to
    the parent. Bilinear upsample only (no transposed conv). No dropout.
    Params: guide 4680 + ref1 5472 + ref2 5256 + head 25 = 15,433.
    """
    def __init__(self, h2_ch=192, guide_ch=24):
        super().__init__()
        self.guide = nn.Sequential(
            nn.Conv2d(h2_ch, guide_ch, 1), nn.GroupNorm(8, guide_ch))
        self.ref1 = nn.Sequential(
            nn.Conv2d(guide_ch + 1, guide_ch, 3, padding=1),
            nn.GroupNorm(8, guide_ch), nn.GELU())
        self.ref2 = nn.Sequential(
            nn.Conv2d(guide_ch, guide_ch, 3, padding=1),
            nn.GroupNorm(8, guide_ch), nn.GELU())
        self.head = nn.Conv2d(guide_ch, 1, 1)  # ZERO-INIT
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, logits, h2):
        # logits: (B,1,96,96) detached; h2: (B,192,48,48) detached/frozen.
        g = self.guide(h2)
        g = F.interpolate(g, scale_factor=2, mode="bilinear", align_corners=False)
        j = torch.cat([logits, g], dim=1)  # (B,25,96,96)
        y = self.ref1(j)
        y = self.ref2(y)
        return self.head(y)  # (B,1,96,96), all-zeros at init


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
        # H0008: HiResDetail is constructed AFTER every parent module
        # (append-only RNG order, AGENTS rule 13), attached to the head, so
        # all parent init draws keep their exact RNG positions; only the new
        # guide/ref init draws are appended. Forward gating lives in
        # CountingHead.forward via self.use_hires.
        self.head.hires = HiResDetail(h2_ch=dims[0])

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
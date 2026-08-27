"""N0033_salf_moe_frozen — frozen DINOv3 ConvNeXt Tiny + SALF-lite + DR-MoE-lite.
Backbone FROZEN. Engine calls build_model(cfg_dict) -> returns Counter with forward(imgs,bboxes[,bboxes3]) -> {"density":...}
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.ops import roi_align

def _get(cfg, k, d=None):
    return cfg[k] if isinstance(cfg, dict) and k in cfg else (getattr(cfg, k, d) if hasattr(cfg, k) else d)

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel
        # hf_token optional
        try:
            from cac_d.common import hf_token
            tok = hf_token()
        except:
            tok = None
        model_name = _get(cfg, "hf_model", "facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
        self.net = AutoModel.from_pretrained(model_name, token=tok, trust_remote_code=True)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        dims = _get(cfg, "backbone_dims", (96,192,384,768))
        self.out_channels = list(dims)
    @torch.no_grad()
    def forward_feature_map(self, x):
        hs = self.net(pixel_values=x, output_hidden_states=True).hidden_states
        return [hs[1], hs[2], hs[3], hs[4]]
    def train(self, mode=True):
        super().train(mode)
        self.net.eval()
        return self

class ExemplarEncoder(nn.Module):
    def __init__(self, in_dim=768, d_model=256, n_layers=2, n_heads=4, roi_size=7):
        super().__init__()
        self.r = roi_size
        self.proj = nn.Linear(in_dim, d_model)
        self.shape_mlp = nn.Sequential(nn.Linear(2,64), nn.ReLU(), nn.Linear(64,d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.attn = nn.Linear(d_model,1)
    def forward(self, feat, bboxes, img_size):
        B,C,H,W = feat.shape
        K = bboxes.shape[1]
        s = W / float(img_size)
        idx = torch.arange(B, device=bboxes.device, dtype=bboxes.dtype).view(B,1,1).expand(B,K,1)
        rois = torch.cat([idx, bboxes * s], -1).reshape(B*K,5)
        roi = roi_align(feat, rois, output_size=(self.r,self.r))
        tok = self.proj(roi.flatten(2).transpose(1,2))
        wh = (bboxes[:,:,2:4] - bboxes[:,:,:2]).clamp_min(1.)
        tok = (tok.view(B, K, self.r*self.r, -1) + self.shape_mlp(wh).unsqueeze(2)).reshape(B*K, self.r*self.r, -1)
        tok = self.tr(tok)
        a = self.attn(tok).softmax(1)
        return (tok * a).sum(1).view(B, K, -1)

class SALF(nn.Module):
    def __init__(self, chs, d=128):
        super().__init__()
        self.lats = nn.ModuleList([nn.Sequential(nn.Conv2d(c,d,1), nn.GroupNorm(8,d)) for c in chs])
        self.gate = nn.Sequential(nn.Conv2d(4*d,128,1), nn.GroupNorm(8,128), nn.GELU(), nn.Conv2d(128,4,1))
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
    def forward(self, feats, target_hw=(48,48)):
        outs=[]
        for i, f in enumerate(feats):
            f = self.lats[i](f)
            if f.shape[-2:] != target_hw:
                f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            outs.append(f)
        cat = torch.cat(outs,1)
        logits = self.gate(cat)
        w = F.softmax(logits, dim=1)
        stacked = torch.stack(outs, dim=1)
        fused = (stacked * w.unsqueeze(2)).sum(1)
        return fused, w

class DRMoE(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.router = nn.Sequential(nn.Conv2d(d,32,1), nn.GroupNorm(4,32), nn.GELU(), nn.Conv2d(32,3,1))
        nn.init.zeros_(self.router[-1].weight); nn.init.zeros_(self.router[-1].bias)
        self.e1 = nn.Sequential(nn.Conv2d(d,d,3,padding=1), nn.GroupNorm(8,d), nn.GELU())
        self.e2 = nn.Sequential(nn.Conv2d(d,d,3,padding=2,dilation=2,groups=d), nn.GroupNorm(8,d), nn.GELU(), nn.Conv2d(d,d,1), nn.GroupNorm(8,d))
        self.e3 = nn.Sequential(nn.Conv2d(d,d,7,padding=3,groups=d), nn.GroupNorm(8,d), nn.GELU(), nn.Conv2d(d,d,1), nn.GroupNorm(8,d))
    def forward(self, x):
        logits = self.router(x)
        w = F.softmax(logits, dim=1)
        o1 = self.e1(x); o2 = self.e2(x); o3 = self.e3(x)
        out = w[:,0:1]*o1 + w[:,1:2]*o2 + w[:,2:3]*o3
        out = F.gelu(out + x)
        return out, w, logits

class Condenser(nn.Module):
    def __init__(self, d_in=128, d_sim=256, n_heads=4, ff=512, d_out=64):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_sim)
        self.attn = nn.MultiheadAttention(d_sim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_sim); self.norm2 = nn.LayerNorm(d_sim)
        self.ffn = nn.Sequential(nn.Linear(d_sim,ff), nn.GELU(), nn.Linear(ff,d_sim))
        self.out = nn.Linear(d_sim, d_out)
    def forward(self, tok, e):
        tok = self.proj_in(tok)
        a,_ = self.attn(self.norm1(tok), e, e, need_weights=False)
        q = self.norm1(tok + a)
        return self.out(self.norm2(q + self.ffn(q)))

class DensityDecoder(nn.Module):
    def __init__(self, in_ch, hidden=256):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch,hidden,3,padding=1), nn.GroupNorm(8,hidden), nn.GELU(),
                                   nn.Conv2d(hidden,hidden//2,3,padding=2,dilation=2), nn.GroupNorm(4,hidden//2), nn.GELU())
        self.head = nn.Conv2d(hidden//2,1,1)
        for m in [self.block[0], self.block[3]]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(self.head.bias)
    def forward(self, x):
        return F.softplus(self.head(self.block(x)))

class Counter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        self.S = _get(cfg, "input_size", _get(cfg, "image_size", 384))
        dims = _get(cfg, "backbone_dims", (96,192,384,768))
        D = _get(cfg, "d_fine", 128)
        self.backbone = Backbone(cfg)
        self.salf = SALF(list(dims), d=D)
        self.moe = DRMoE(d=D)
        d_model = _get(cfg, "embed_dim", 256)
        n_layers = _get(cfg, "exemplar_layers", 2)
        roi_size = _get(cfg, "roi_size", 7)
        self.exemplar = ExemplarEncoder(in_dim=dims[-1], d_model=d_model, n_layers=n_layers, roi_size=roi_size)
        cond_dim = _get(cfg, "cond_dim", 64)
        self.cond = Condenser(d_in=D, d_sim=d_model, d_out=cond_dim)
        self.density_head = DensityDecoder(in_ch=D+cond_dim, hidden=2*D)
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    def forward(self, imgs, bboxes, bboxes3=None):
        # bboxes [B,4], bboxes3 [B,3,4] optional — use single bbox expanded to 3 for exemplar encoder
        if bboxes.dim()==2:
            bboxes = bboxes.unsqueeze(1)
        if bboxes3 is not None:
            bboxes_in = bboxes3
        else:
            # expand single box to 1 exemplar (encoder handles K=1)
            bboxes_in = bboxes
            # if only 1 box, repeat to 3? keep 1 to avoid shape issues — encoder supports any K
        feats = self.backbone.forward_feature_map(imgs)
        H8 = self.S // 8
        fused, w_salf = self.salf(feats, target_hw=(H8,H8))
        routed, w_moe, _ = self.moe(fused)
        B = routed.shape[0]
        Hf = Wf = H8*2
        fine = F.interpolate(routed, size=(Hf,Wf), mode="bilinear", align_corners=False)
        fmap = fine.permute(0,2,3,1).flatten(1,2)
        e = self.exemplar(feats[-1], bboxes_in, self.S)
        cond = self.cond(fmap, e)
        dens = self.density_head(torch.cat([fine, cond.transpose(1,2).reshape(B,-1,Hf,Wf)],1))
        return {"density": dens}

def build_model(cfg):
    return Counter(cfg)

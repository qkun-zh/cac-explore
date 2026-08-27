"""N0033_salf_moe_frozen — frozen DINOv3 ConvNeXt Tiny + SALF-lite + DR-MoE-lite.
Backbone FROZEN (eval, no grad). Only ~1.4M extra vs champion.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.ops import roi_align

# ── Frozen backbone exposing 4 stages ──
class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel
        from cac_d.common import hf_token
        self.net = AutoModel.from_pretrained(cfg.hf_model, token=hf_token(), trust_remote_code=True)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.out_channels = list(cfg.backbone_dims)  # [96,192,384,768]
    @torch.no_grad()
    def forward_feature_map(self, x):
        hs = self.net(pixel_values=x, output_hidden_states=True).hidden_states
        # hs[0]=image, hs[1]=s4, hs[2]=s8, hs[3]=s16, hs[4]=s32
        return [hs[1], hs[2], hs[3], hs[4]]
    def train(self, mode=True):
        super().train(mode)
        self.eval()
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
        tok = (tok.view(B,K,self.r*self.r,-1) + self.shape_mlp(wh).unsqueeze(2)).reshape(B*K,self.r*self.r,-1)
        tok = self.tr(tok)
        a = self.attn(tok).softmax(1)
        return (tok*a).sum(1).view(B,K,-1)

class SALF(nn.Module):
    """4-stage → H/8 fused 128ch"""
    def __init__(self, chs, d=128):
        super().__init__()
        self.lats = nn.ModuleList([nn.Sequential(nn.Conv2d(c,d,1), nn.GroupNorm(8,d)) for c in chs])
        self.gate = nn.Sequential(nn.Conv2d(4*d,128,1), nn.GroupNorm(8,128), nn.GELU(),
                                  nn.Conv2d(128,4,1))
        # init gate bias 0 → uniform softmax step0
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
    def forward(self, feats, target_hw=(48,48)):
        # feats: [s4 96x96, s8 48x48, s16 24x24, s32 12x12] @384
        outs=[]
        for i, f in enumerate(feats):
            f = self.lats[i](f)
            if f.shape[-2:] != target_hw:
                f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            outs.append(f)
        cat = torch.cat(outs,1)  # 512ch
        logits = self.gate(cat)  # [B,4,H,W]
        w = F.softmax(logits, dim=1)  # per-location
        stacked = torch.stack(outs, dim=1)  # [B,4,128,H,W]
        fused = (stacked * w.unsqueeze(2)).sum(1)  # [B,128,H,W]
        return fused, w

class DRMoE(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.router = nn.Sequential(nn.Conv2d(d,32,1), nn.GroupNorm(4,32), nn.GELU(),
                                    nn.Conv2d(32,3,1))
        nn.init.zeros_(self.router[-1].weight); nn.init.zeros_(self.router[-1].bias)
        self.e1 = nn.Sequential(nn.Conv2d(d,d,3,padding=1), nn.GroupNorm(8,d), nn.GELU())
        self.e2 = nn.Sequential(nn.Conv2d(d,d,3,padding=2,dilation=2,groups=d), nn.GroupNorm(8,d), nn.GELU(),
                                nn.Conv2d(d,d,1), nn.GroupNorm(8,d))
        self.e3 = nn.Sequential(nn.Conv2d(d,d,7,padding=3,groups=d), nn.GroupNorm(8,d), nn.GELU(),
                                nn.Conv2d(d,d,1), nn.GroupNorm(8,d))
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
        self.cfg=cfg; self.S=cfg.image_size
        self.backbone = Backbone(cfg)
        D=cfg.d_fine
        self.salf = SALF(list(cfg.backbone_dims), d=D)
        self.moe = DRMoE(d=D)
        self.exemplar = ExemplarEncoder(in_dim=cfg.backbone_dims[-1], d_model=cfg.embed_dim, n_layers=cfg.exemplar_layers, roi_size=cfg.roi_size)
        self.cond = Condenser(d_in=D, d_sim=cfg.embed_dim, d_out=cfg.cond_dim)
        self.density = DensityDecoder(in_ch=D+cfg.cond_dim, hidden=2*D)
        self.balance_weight = float(getattr(cfg,'balance_weight',0.01))
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    def forward(self, x, bboxes, points=None):
        # x [B,3,S,S], bboxes [B,K,4]
        feats = self.backbone.forward_feature_map(x)  # 4 stages
        H8 = self.S // 8  # 48 @384
        fused, w_salf = self.salf(feats, target_hw=(H8,H8))  # [B,128,48,48]
        routed, w_moe, _ = self.moe(fused)  # [B,128,48,48]
        B = routed.shape[0]
        Hf = Wf = H8*2  # 96
        fine = F.interpolate(routed, size=(Hf,Wf), mode="bilinear", align_corners=False)
        fmap = fine.permute(0,2,3,1).flatten(1,2)
        e = self.exemplar(feats[-1], bboxes, self.S)
        cond = self.cond(fmap, e)
        dens = self.density(torch.cat([fine, cond.transpose(1,2).reshape(B,-1,Hf,Wf)],1))
        counts = dens.sum((1,2,3))
        if points is None:
            return {"pred_counts":counts, "density":dens, "w_salf":w_salf, "w_moe":w_moe}
        # losses
        from cac_d.models.losses.losses import gaussian_density, adaptive_gaussian_density, bayesian_density_loss
        import torch.nn.functional as F2
        c=self.cfg
        kw=dict(k=c.gauss_knn, smin=c.sigma_min, smax=c.sigma_max, beta=c.sigma_beta)
        if c.density_loss=="mse":
            gt_d = gaussian_density(points, B, Hf, Wf, self.S, sigma=c.gauss_sigma)
            loss_den = F2.mse_loss(dens, gt_d)
        elif c.density_loss=="ada_mse":
            gt_d = adaptive_gaussian_density(points, B, Hf, Wf, self.S, **kw)
            loss_den = F2.mse_loss(dens, gt_d)
        elif c.density_loss=="bl":
            loss_den = bayesian_density_loss(dens, points, Hf, Wf, self.S, **kw)
        else:
            raise ValueError(c.density_loss)
        N = torch.tensor([len(q) for q in points], device=dens.device, dtype=torch.float32)
        loss_cnt = F2.smooth_l1_loss((counts+1).log(), (N+1).log())
        # balance loss
        w_mean = w_moe.mean(dim=(0,2,3))  # [3]
        cv = w_mean.std() / (w_mean.mean().clamp_min(1e-6))
        loss_bal = (cv**2) * self.balance_weight
        loss = c.density_weight*loss_den + c.cnt_weight*loss_cnt + loss_bal
        return {"loss":loss, "pred_counts":counts.detach(), "density":dens, "loss_den":loss_den.detach(), "loss_cnt":loss_cnt.detach(), "loss_bal":loss_bal.detach(), "w_salf":w_salf.detach(), "w_moe":w_moe.detach()}

def build_model(cfg):
    return Counter(cfg)

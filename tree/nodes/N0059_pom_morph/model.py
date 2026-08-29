"""N0059_pom_morph — frozen ConvNeXt Tiny + pluggable CountingHead + PoM-PolyMorpher Exemplar (POM).
Pluggable base = N0054, with use_ddca=False (DDCA harmful, +1.8) and use_gca=True (the winning aux).
N0059 REPLACES the per-exemplar aggregation operator (proj 384->256 + 2xTransformerEncoderLayer self-
attention + attention-pool over 49 ROI patch tokens) with 2x PoM-PolyMorpher blocks (arXiv:2604.06129
eq.3) at MATCHED capacity (D=352, k=2) over all 49 tokens. Each block keeps a token-averaged 2nd-order
moment state H with a per-token nonlinear gate, and a norm_first residual path structurally identical to
N0054's TransformerEncoderLayer. Keeps the single fused-prototype interface the load-bearing Condenser
consumes (the allowed exemplar-embedding interface, §5.14) and attn-pool/roi/shape_mlp/xscale EXACTLY
as N0054, ~0.0M delta to head, single-switch use_pom. use_pom=False restores the exact N0054 attention
path (bit-identical, single-switch identity proven by param match).
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.ops import roi_align
def _get(cfg,k,d=None):
    return cfg[k] if isinstance(cfg,dict) and k in cfg else (getattr(cfg,k,d) if hasattr(cfg,k) else d)

class Backbone(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from transformers import AutoModel
        try:
            from cac_d.common import hf_token
            tok=hf_token()
        except:
            tok=None
        name=_get(cfg,"hf_model","facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
        self.net=AutoModel.from_pretrained(name, token=tok, trust_remote_code=True)
        self.net.eval()
        for p in self.net.parameters(): p.requires_grad_(False)
        dims=_get(cfg,"backbone_dims",(192,384))
        self.out_channels=list(dims)
        self.hs_map=(2,3)
    @torch.no_grad()
    def forward_feature_map(self,x):
        hs=self.net(pixel_values=x, output_hidden_states=True).hidden_states
        return [hs[i] for i in self.hs_map]
    def train(self,mode=True):
        super().train(mode)
        self.net.eval()
        return self

class FineFuser(nn.Module):
    """Component: fuse coarse(mid) + mid(fine) backbone features -> 1/4-res dense feature.
    Optional DDCA = parallel dilated dw branch (use_ddca). Self-contained branch; no external coupling.
    """
    def __init__(self,ch_coarse,ch_mid,d_fine=128,use_ddca=True):
        super().__init__()
        self.use_ddca=use_ddca
        self.top=nn.Sequential(nn.Conv2d(ch_coarse,d_fine,1), nn.GroupNorm(8,d_fine))
        self.lat=nn.Sequential(nn.Conv2d(ch_mid,d_fine,1), nn.GroupNorm(8,d_fine))
        self.fuse=nn.Sequential(nn.Conv2d(2*d_fine,d_fine,3,padding=1), nn.GroupNorm(8,d_fine), nn.GELU())
        self.refine=nn.Conv2d(d_fine,d_fine,3,padding=1,groups=d_fine)
        if use_ddca:
            self.ctx=nn.Conv2d(d_fine,d_fine,3,padding=2,dilation=2,groups=d_fine)
            nn.init.zeros_(self.ctx.weight)
            if self.ctx.bias is not None:
                nn.init.zeros_(self.ctx.bias)
    def forward(self,h2,h3):
        top=F.interpolate(self.top(h3),scale_factor=2,mode="bilinear",align_corners=False)
        f=self.fuse(torch.cat([self.lat(h2),top],1))
        if self.use_ddca:
            f=F.gelu(self.refine(f) + f + self.ctx(f))
        else:
            f=F.gelu(self.refine(f) + f)
        return F.interpolate(f,scale_factor=2,mode="bilinear",align_corners=False)

class PolyMorpher(nn.Module):
    """PoM polynomial-mixer aggregation block (arXiv:2604.06129 eq.3). Token-averaged 2nd-order moment
    state H(X) + per-token nonlinear gate sigma(W_s X), W_o mixing, norm_first residual path identical
    in structure to the N0054 TransformerEncoderLayer it replaces (norm -> sub-layer -> residual, norm
    -> sub-layer -> residual)."""
    def __init__(self,d_model,D=352,k=2,ff=1024):
        super().__init__()
        self.D=D
        self.norm1=nn.LayerNorm(d_model)
        self.W_h=nn.Linear(d_model,D)
        self.alpha=nn.Parameter(torch.zeros(D,k))
        nn.init.uniform_(self.alpha,-1e-3,1e-3)
        self.W_s=nn.Linear(d_model,D)
        self.W_o=nn.Linear(D,d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.ff=nn.Sequential(nn.Linear(d_model,ff), nn.GELU(), nn.Linear(ff,d_model))
    def forward(self,x):
        xn=self.norm1(x)
        h=F.gelu(self.W_h(xn))                                       # (B*K,49,D)
        H=(self.alpha[:,0].view(1,1,self.D)*h**1
           +self.alpha[:,1].view(1,1,self.D)*h**2).mean(dim=1)       # (B*K,D)  mean over 49 tokens
        gate_v=torch.sigmoid(self.W_s(xn))                           # (B*K,49,D)
        po=gate_v*H.unsqueeze(1)                                     # (B*K,49,D)
        po_parts=self.W_o(po)                                        # (B*K,49,256)
        x=x+po_parts
        x=x+self.ff(self.norm2(x))
        return x

class ExemplarEncoder(nn.Module):
    def __init__(self,in_dim=384,d_model=256,n_layers=2,n_heads=4,roi_size=7,use_xscale=False,xs=3,use_pom=False,POM_D=352,POM_K=2,POM_BLOCKS=2):
        super().__init__()
        self.r=roi_size
        self.use_xscale=use_xscale
        self.use_pom=use_pom
        self.proj=nn.Linear(in_dim,d_model)
        self.shape_mlp=nn.Sequential(nn.Linear(2,64), nn.ReLU(), nn.Linear(64,d_model))
        if use_pom:
            self.pom=nn.Sequential(*[PolyMorpher(d_model,POM_D,k=POM_K) for _ in range(POM_BLOCKS)])
        else:
            layer=nn.TransformerEncoderLayer(d_model,n_heads,d_model*4,dropout=0.0,batch_first=True,norm_first=True)
            self.tr=nn.TransformerEncoder(layer,n_layers,enable_nested_tensor=False)
        self.attn=nn.Linear(d_model,1)
        if use_xscale:
            self.xs=xs
            self.xproj=nn.Linear(in_dim,d_model)
    def forward(self,feat,bboxes,img_size):
        B,C,H,W=feat.shape
        K=bboxes.shape[1]
        s=W/float(img_size)
        idx=torch.arange(B,device=bboxes.device,dtype=bboxes.dtype).view(B,1,1).expand(B,K,1)
        rois=torch.cat([idx,bboxes*s],-1).reshape(B*K,5)
        roi=roi_align(feat,rois,output_size=(self.r,self.r))
        tok=self.proj(roi.flatten(2).transpose(1,2))
        wh=(bboxes[:,:,2:4]-bboxes[:,:,:2]).clamp_min(1.)
        tok=(tok.view(B,K,self.r*self.r,-1)+self.shape_mlp(wh).unsqueeze(2)).reshape(B*K,self.r*self.r,-1)
        tok=self.pom(tok) if self.use_pom else self.tr(tok)
        a=self.attn(tok).softmax(1)
        out=(tok*a).sum(1).view(B,K,-1)
        if self.use_xscale:
            # coarse global summary per exemplar: pool at 2nd (smaller) scale -> GAP -> proj -> add
            roi2=roi_align(feat,rois,output_size=(self.xs,self.xs))
            coarse=F.adaptive_avg_pool2d(roi2,(1,1)).squeeze(-1).squeeze(-1)  # (B*K,in_dim)
            out=out+self.xproj(coarse).view(B,K,-1)
        return out

class Condenser(nn.Module):
    def __init__(self,d_in=128,d_sim=256,n_heads=4,ff=512,d_out=64):
        super().__init__()
        self.proj_in=nn.Linear(d_in,d_sim)
        self.attn=nn.MultiheadAttention(d_sim,n_heads,batch_first=True)
        self.norm1=nn.LayerNorm(d_sim);self.norm2=nn.LayerNorm(d_sim)
        self.ffn=nn.Sequential(nn.Linear(d_sim,ff),nn.GELU(),nn.Linear(ff,d_sim))
        self.out=nn.Linear(d_sim,d_out)
    def forward(self,tok,e):
        tok=self.proj_in(tok)
        a,_=self.attn(self.norm1(tok),e,e,need_weights=False)
        q=self.norm1(tok+a)
        return self.out(self.norm2(q+self.ffn(q)))

class DensityDecoder(nn.Module):
    def __init__(self,in_ch,hidden=256):
        super().__init__()
        self.block=nn.Sequential(nn.Conv2d(in_ch,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.GELU(),nn.Conv2d(hidden,hidden//2,3,padding=2,dilation=2),nn.GroupNorm(4,hidden//2),nn.GELU())
        self.head=nn.Conv2d(hidden//2,1,1)
        for m in [self.block[0],self.block[3]]: nn.init.kaiming_normal_(m.weight,nonlinearity="relu")
        nn.init.zeros_(self.head.bias)
    def forward(self,x): return F.softplus(self.head(self.block(x)))

class CountingHead(nn.Module):
    """PLUGGABLE TRUNK: bundles FineFuser + ExemplarEncoder + Condenser + DensityDecoder as ONE module.
    Emits density + exposes fine and e_mean as shared-interface outputs so external components
    (GCA, ...) can attach without touching trunk internals.
    """
    def __init__(self,cfg):
        super().__init__()
        D=_get(cfg,"d_fine",128)
        dims=_get(cfg,"backbone_dims",(192,384))
        use_ddca=_get(cfg,"use_ddca",True)
        self.fuser=FineFuser(dims[1],dims[0],d_fine=D,use_ddca=use_ddca)
        self.exemplar=ExemplarEncoder(in_dim=dims[1],d_model=_get(cfg,"embed_dim",256),n_layers=_get(cfg,"exemplar_layers",2),roi_size=_get(cfg,"roi_size",7),use_xscale=_get(cfg,"use_xscale",False),xs=_get(cfg,"xscale_size",3),use_pom=_get(cfg,"use_pom",False),POM_D=_get(cfg,"POM_D",352),POM_K=_get(cfg,"POM_K",2),POM_BLOCKS=_get(cfg,"POM_BLOCKS",2))
        self.cond=Condenser(d_in=D,d_sim=_get(cfg,"embed_dim",256),d_out=_get(cfg,"cond_dim",64))
        self.decoder=DensityDecoder(in_ch=D+_get(cfg,"cond_dim",64),hidden=2*D)
        self.S=_get(cfg,"input_size",384)
    def forward(self,h2,h3,bboxes_in):
        B=bboxes_in.shape[0]
        fine=self.fuser(h2,h3)
        Hf=Wf=self.S//4
        fmap=fine.permute(0,2,3,1).flatten(1,2)
        e=self.exemplar(h3,bboxes_in,self.S)
        cond=self.cond(fmap,e)
        dens=self.decoder(torch.cat([fine,cond.transpose(1,2).reshape(B,-1,Hf,Wf)],1))
        e_mean=e.mean(dim=1)  # shared interface for GCA
        return dens, fine, e_mean
    def train(self,mode=True):
        super().train(mode)
        return self

class GCA(nn.Module):
    """Independent pluggable aux: global count head. Reads GAP(fine) + e_mean only."""
    def __init__(self,D,d_model):
        super().__init__()
        self.gca=nn.Sequential(nn.Linear(D+d_model,64), nn.GELU(), nn.Linear(64,1))
        nn.init.zeros_(self.gca[-1].weight); nn.init.zeros_(self.gca[-1].bias)
    def forward(self,fine,e_mean,Hf,Wf):
        gap=fine.mean(dim=(2,3))
        n_aux=F.softplus(self.gca(torch.cat([gap,e_mean],1))).squeeze(1)
        bias=(n_aux/float(Hf*Wf)).view(-1,1,1,1)*0.02
        return n_aux, bias

class Counter(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.S=_get(cfg,"input_size",384)
        dims=_get(cfg,"backbone_dims",(192,384))
        D=_get(cfg,"d_fine",128)
        self.use_gca=_get(cfg,"use_gca",True)
        self.backbone=Backbone(cfg)
        self.head=CountingHead(cfg)
        if self.use_gca:
            self.gca=GCA(D,_get(cfg,"embed_dim",256))
    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    def forward(self,imgs,bboxes,bboxes3=None):
        if bboxes.dim()==2: bboxes=bboxes.unsqueeze(1)
        bboxes_in=bboxes3 if bboxes3 is not None else bboxes
        h2,h3=self.backbone.forward_feature_map(imgs)
        dens,fine,e_mean=self.head(h2,h3,bboxes_in)
        B=imgs.shape[0]
        Hf=Wf=self.S//4
        out={"density":dens}
        if self.use_gca:
            n_aux,bias=self.gca(fine,e_mean,Hf,Wf)
            out["density"]=dens+bias
            out["n_aux"]=n_aux
        return out
def build_model(cfg): return Counter(cfg)

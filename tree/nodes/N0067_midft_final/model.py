"""N0067_midft_final — GCA+XScale on FINAL readout with mid-late FT.
Backbone: DINOv3-ConvNeXt-Tiny hs_map (3,4) dims (384,768), stages 2,3 FT @0.1x LR.
Head adapters scaled to 768ch exemplar/fuser. Tests final vs intermediate.
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
        self.hs_map=tuple(_get(cfg,"hs_map",(3,4)))
        self.tune_stages=list(_get(cfg,"tune_stages",[2,3]))
        self.backbone_lr=float(_get(cfg,"backbone_lr",1e-4))
        for p in self.net.parameters(): p.requires_grad_(False)
        for idx in self.tune_stages:
            try:
                for p in self.net.model.stages[idx].parameters():
                    p.requires_grad_(True)
            except Exception:
                pass
        dims_map={1:96,2:192,3:384,4:768}
        dims=tuple(dims_map[i] for i in self.hs_map)
        self.out_channels=list(dims)
        self._dims=dims
    def forward_feature_map(self,x):
        hs=self.net(pixel_values=x, output_hidden_states=True).hidden_states
        return [hs[i] for i in self.hs_map]
    def train(self,mode=True):
        super().train(mode)
        try:
            self.net.train(mode)
            for i, stage in enumerate(self.net.model.stages):
                if i in self.tune_stages:
                    stage.train(mode)
                else:
                    stage.eval()
        except Exception:
            self.net.eval() if not self.tune_stages else self.net.train(mode)
        return self

class FineFuser(nn.Module):
    def __init__(self,ch_coarse,ch_mid,d_fine=128,use_ddca=False):
        super().__init__()
        self.use_ddca=use_ddca
        self.top=nn.Sequential(nn.Conv2d(ch_coarse,d_fine,1), nn.GroupNorm(8,d_fine))
        self.lat=nn.Sequential(nn.Conv2d(ch_mid,d_fine,1), nn.GroupNorm(8,d_fine))
        self.fuse=nn.Sequential(nn.Conv2d(2*d_fine,d_fine,3,padding=1), nn.GroupNorm(8,d_fine), nn.GELU())
        self.refine=nn.Conv2d(d_fine,d_fine,3,padding=1,groups=d_fine)
        if use_ddca:
            self.ctx=nn.Conv2d(d_fine,d_fine,3,padding=2,dilation=2,groups=d_fine)
            nn.init.zeros_(self.ctx.weight)
            if self.ctx.bias is not None: nn.init.zeros_(self.ctx.bias)
    def forward(self,h2,h3):
        top=F.interpolate(self.top(h3),scale_factor=2,mode="bilinear",align_corners=False)
        f=self.fuse(torch.cat([self.lat(h2),top],1))
        if self.use_ddca:
            f=F.gelu(self.refine(f) + f + self.ctx(f))
        else:
            f=F.gelu(self.refine(f) + f)
        # adaptive to S//4=96: fuse output is at h2 res (24 for final, 48 for intermediate) -> interpolate to 96
        Hf = h2.shape[-1] * 4 if h2.shape[-1] == 24 else h2.shape[-1] * 2  # 24->96, 48->96
        # generic: target is 4x hs[3] (12->48->96) or 2x hs[2] whichever yields 96; simply interpolate to 96
        return F.interpolate(f,size=(96,96),mode="bilinear",align_corners=False)

class ExemplarEncoder(nn.Module):
    def __init__(self,in_dim=768,d_model=256,n_layers=2,n_heads=4,roi_size=7,use_xscale=True,xs=3):
        super().__init__()
        self.r=roi_size
        self.use_xscale=use_xscale
        self.proj=nn.Linear(in_dim,d_model)
        self.shape_mlp=nn.Sequential(nn.Linear(2,64), nn.ReLU(), nn.Linear(64,d_model))
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
        tok=self.tr(tok)
        a=self.attn(tok).softmax(1)
        out=(tok*a).sum(1).view(B,K,-1)
        if self.use_xscale:
            roi2=roi_align(feat,rois,output_size=(self.xs,self.xs))
            coarse=F.adaptive_avg_pool2d(roi2,(1,1)).squeeze(-1).squeeze(-1)
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
    def __init__(self,cfg):
        super().__init__()
        D=_get(cfg,"d_fine",128)
        dims=_get(cfg,"backbone_dims",(384,768))
        use_ddca=_get(cfg,"use_ddca",False)
        self.fuser=FineFuser(dims[1],dims[0],d_fine=D,use_ddca=use_ddca)
        self.exemplar=ExemplarEncoder(in_dim=dims[1],d_model=_get(cfg,"embed_dim",256),n_layers=_get(cfg,"exemplar_layers",2),roi_size=_get(cfg,"roi_size",7),use_xscale=_get(cfg,"use_xscale",True),xs=_get(cfg,"xscale_size",3))
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
        e_mean=e.mean(dim=1)
        return dens, fine, e_mean

class GCA(nn.Module):
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
        dims=_get(cfg,"backbone_dims",(384,768))
        D=_get(cfg,"d_fine",128)
        self.use_gca=_get(cfg,"use_gca",True)
        self.backbone=Backbone(cfg)
        self.head=CountingHead(cfg)
        if self.use_gca:
            self.gca=GCA(D,_get(cfg,"embed_dim",256))
    def param_groups(self, lr, weight_decay):
        bb_lr=float(self.cfg.get("backbone_lr", lr*0.1))
        bb_params=[]; head_params=[]
        for n,p in self.named_parameters():
            if not p.requires_grad: continue
            if n.startswith("backbone."):
                bb_params.append(p)
            else:
                head_params.append(p)
        groups=[]
        if bb_params: groups.append({"params": bb_params, "lr": bb_lr, "weight_decay": weight_decay})
        if head_params: groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
        return groups if groups else [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr, "weight_decay": weight_decay}]
    def train(self,mode=True):
        super().train(mode)
        self.backbone.train(mode)
        return self
    def forward(self,imgs,bboxes,bboxes3=None):
        if bboxes.dim()==2: bboxes=bboxes.unsqueeze(1)
        bboxes_in=bboxes3 if bboxes3 is not None else bboxes
        h2,h3=self.backbone.forward_feature_map(imgs)
        dens,fine,e_mean=self.head(h2,h3,bboxes_in)
        Hf=Wf=self.S//4
        out={"density":dens}
        if self.use_gca:
            n_aux,bias=self.gca(fine,e_mean,Hf,Wf)
            out["density"]=dens+bias
            out["n_aux"]=n_aux
        return out
def build_model(cfg): return Counter(cfg)

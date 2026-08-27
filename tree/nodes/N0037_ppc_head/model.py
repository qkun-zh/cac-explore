"""N0037_ppc_head — frozen ConvNeXt Tiny + FineFuser + PPC factorized N·p head."""
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
    def __init__(self,ch_coarse,ch_mid,d_fine=128):
        super().__init__()
        self.top=nn.Sequential(nn.Conv2d(ch_coarse,d_fine,1), nn.GroupNorm(8,d_fine))
        self.lat=nn.Sequential(nn.Conv2d(ch_mid,d_fine,1), nn.GroupNorm(8,d_fine))
        self.fuse=nn.Sequential(nn.Conv2d(2*d_fine,d_fine,3,padding=1), nn.GroupNorm(8,d_fine), nn.GELU())
        self.refine=nn.Conv2d(d_fine,d_fine,3,padding=1,groups=d_fine)
    def forward(self,h2,h3):
        top=F.interpolate(self.top(h3),scale_factor=2,mode="bilinear",align_corners=False)
        f=self.fuse(torch.cat([self.lat(h2),top],1))
        f=F.gelu(self.refine(f)+f)
        return F.interpolate(f,scale_factor=2,mode="bilinear",align_corners=False)
class ExemplarEncoder(nn.Module):
    def __init__(self,in_dim=384,d_model=256,n_layers=2,n_heads=4,roi_size=7):
        super().__init__()
        self.r=roi_size
        self.proj=nn.Linear(in_dim,d_model)
        self.shape_mlp=nn.Sequential(nn.Linear(2,64), nn.ReLU(), nn.Linear(64,d_model))
        layer=nn.TransformerEncoderLayer(d_model,n_heads,d_model*4,dropout=0.0,batch_first=True,norm_first=True)
        self.tr=nn.TransformerEncoder(layer,n_layers,enable_nested_tensor=False)
        self.attn=nn.Linear(d_model,1)
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
        return (tok*a).sum(1).view(B,K,-1)
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

class PPCHead(nn.Module):
    def __init__(self, d_fine=128, cond_dim=64, d_model=256):
        super().__init__()
        # shape branch: cond 64 -> 32 ->1
        self.shape=nn.Sequential(nn.Conv2d(cond_dim,32,3,padding=1), nn.GroupNorm(8,32), nn.GELU(),
                                 nn.Conv2d(32,1,1))
        # count branch: GAP fine 128 + e_mean 256 -> 384->64->1
        self.count_mlp=nn.Sequential(nn.Linear(d_fine+d_model,64), nn.GELU(), nn.Linear(64,1))
        nn.init.zeros_(self.shape[-1].bias)
        nn.init.zeros_(self.count_mlp[-1].bias)
    def forward(self, cond_map, fine, e):
        # cond_map [B,64,Hf,Wf], fine [B,128,Hf,Wf], e [B,K,256]
        s = F.softplus(self.shape(cond_map))  # [B,1,Hf,Wf] positive
        # normalize to sum 1 per image (avoid div by zero)
        s_sum = s.flatten(2).sum(2).clamp_min(1e-6).view(s.shape[0],1,1,1)
        p = s / s_sum  # sum 1
        gap = fine.mean(dim=(2,3))  # [B,128]
        e_mean = e.mean(dim=1)  # [B,256]
        n = F.softplus(self.count_mlp(torch.cat([gap, e_mean],1))).squeeze(1)  # [B]
        # density = N * p
        dens = p * n.view(-1,1,1,1)
        return dens, p, n

class Counter(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.S=_get(cfg,"input_size",384)
        dims=_get(cfg,"backbone_dims",(192,384))
        D=_get(cfg,"d_fine",128)
        self.backbone=Backbone(cfg)
        self.fuser=FineFuser(dims[1],dims[0],d_fine=D)
        self.exemplar=ExemplarEncoder(in_dim=dims[1],d_model=_get(cfg,"embed_dim",256),n_layers=_get(cfg,"exemplar_layers",2),roi_size=_get(cfg,"roi_size",7))
        self.cond=Condenser(d_in=D,d_sim=_get(cfg,"embed_dim",256),d_out=_get(cfg,"cond_dim",64))
        self.head=PPCHead(d_fine=D, cond_dim=_get(cfg,"cond_dim",64), d_model=_get(cfg,"embed_dim",256))
    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    def forward(self,imgs,bboxes,bboxes3=None):
        if bboxes.dim()==2: bboxes=bboxes.unsqueeze(1)
        bboxes_in=bboxes3 if bboxes3 is not None else bboxes
        h2,h3=self.backbone.forward_feature_map(imgs)
        fine=self.fuser(h2,h3)
        B=imgs.shape[0]
        Hf=Wf=self.S//4
        fmap=fine.permute(0,2,3,1).flatten(1,2)
        e=self.exemplar(h3,bboxes_in,self.S)
        cond=self.cond(fmap,e)
        cond_map=cond.transpose(1,2).reshape(B,-1,Hf,Wf)
        dens,_,_=self.head(cond_map, fine, e)
        return {"density":dens}
def build_model(cfg): return Counter(cfg)

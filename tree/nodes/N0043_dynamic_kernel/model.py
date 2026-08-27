"""N0043_dynamic_kernel — N0036 GCA+DDCA + SCDC dynamic depthwise kernel, no Condenser."""
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

class FineFuserDDCA(nn.Module):
    def __init__(self,ch_coarse,ch_mid,d_fine=128):
        super().__init__()
        self.top=nn.Sequential(nn.Conv2d(ch_coarse,d_fine,1), nn.GroupNorm(8,d_fine))
        self.lat=nn.Sequential(nn.Conv2d(ch_mid,d_fine,1), nn.GroupNorm(8,d_fine))
        self.fuse=nn.Sequential(nn.Conv2d(2*d_fine,d_fine,3,padding=1), nn.GroupNorm(8,d_fine), nn.GELU())
        self.refine=nn.Conv2d(d_fine,d_fine,3,padding=1,groups=d_fine)
        self.ctx=nn.Conv2d(d_fine,d_fine,3,padding=2,dilation=2,groups=d_fine)
        nn.init.zeros_(self.ctx.weight)
        if self.ctx.bias is not None: nn.init.zeros_(self.ctx.bias)
    def forward(self,h2,h3):
        top=F.interpolate(self.top(h3),scale_factor=2,mode="bilinear",align_corners=False)
        f=self.fuse(torch.cat([self.lat(h2),top],1))
        f=F.gelu(self.refine(f)+f+self.ctx(f))
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

class DensityDecoder(nn.Module):
    def __init__(self,in_ch,hidden=256):
        super().__init__()
        self.block=nn.Sequential(nn.Conv2d(in_ch,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.GELU(),nn.Conv2d(hidden,hidden//2,3,padding=2,dilation=2),nn.GroupNorm(4,hidden//2),nn.GELU())
        self.head=nn.Conv2d(hidden//2,1,1)
        for m in [self.block[0],self.block[3]]: nn.init.kaiming_normal_(m.weight,nonlinearity="relu")
        nn.init.zeros_(self.head.bias)
    def forward(self,x): return F.softplus(self.head(self.block(x)))

class Counter(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.S=_get(cfg,"input_size",384)
        dims=_get(cfg,"backbone_dims",(192,384))
        D=_get(cfg,"d_fine",128)
        d_model=_get(cfg,"embed_dim",256)
        self.backbone=Backbone(cfg)
        self.fuser=FineFuserDDCA(dims[1],dims[0],d_fine=D)
        self.exemplar=ExemplarEncoder(in_dim=dims[1],d_model=d_model,n_layers=_get(cfg,"exemplar_layers",2),roi_size=_get(cfg,"roi_size",7))
        # SCDC: e_mean -> depthwise 3x3 kernel
        self.kernel_gen=nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, D*9))
        nn.init.zeros_(self.kernel_gen[-1].weight); nn.init.zeros_(self.kernel_gen[-1].bias)
        self.decoder=DensityDecoder(in_ch=D, hidden=2*D)
        self.gca=nn.Sequential(nn.Linear(D+d_model,64), nn.GELU(), nn.Linear(64,1))
        nn.init.zeros_(self.gca[-1].weight); nn.init.zeros_(self.gca[-1].bias)
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
        e=self.exemplar(h3,bboxes_in,self.S)
        e_mean=e.mean(dim=1)
        # dynamic depthwise kernel
        k=self.kernel_gen(e_mean).view(B, D, 3, 3)
        k=F.softmax(k.view(B,D,9), dim=-1).view(B,D,3,3)
        # apply depthwise per sample (grouped conv workaround)
        # fine: [B,D,Hf,Wf], k: [B,D,3,3] -> loop over B or use conv trick
        # Use unfold + einsum for efficiency and gradient
        # unfold fine to [B, D*9, Hf*Wf]
        fine_pad=F.pad(fine, (1,1,1,1), mode='replicate')
        patches=F.unfold(fine_pad, kernel_size=3)  # Wait fine already padded? Use direct unfold
        # Better: F.unfold(fine, 3, padding=1) -> [B, D*9, Hf*Wf]
        patches=F.unfold(fine, kernel_size=3, padding=1)  # [B, D*9, L]
        k_flat=k.view(B, D, 9)  # [B,D,9]
        # Need to apply per-channel: patches view [B,D,9,L] -> sum over 9 weighted by k
        patches=patches.view(B, D, 9, Hf*Wf)
        filtered=(patches * k_flat.unsqueeze(-1)).sum(dim=2)  # [B,D,L]
        filtered=filtered.view(B,D,Hf,Wf)
        # residual blend, zero-init ensures filtered ~ average, blend 0.5
        feat=fine*0.5 + filtered*0.5
        dens=self.decoder(feat)
        gap=fine.mean(dim=(2,3))
        gca_in=torch.cat([gap,e_mean],1)
        n_aux=F.softplus(self.gca(gca_in)).squeeze(1)
        bias=(n_aux/float(Hf*Wf)).view(B,1,1,1)*0.02
        dens=dens+bias
        return {"density":dens,"n_aux":n_aux}
def build_model(cfg): return Counter(cfg)

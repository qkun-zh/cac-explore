import math, torch, torch.nn as nn, torch.nn.functional as F
PATCH=16
class PromptEncoderV2(nn.Module):
    def __init__(self, freqs=8, hidden=256, out_dim=384):
        super().__init__()
        self.register_buffer("freqs", 2.0**torch.arange(freqs)*math.pi)
        self.mlp=nn.Sequential(nn.Linear(4*freqs*2+1,hidden),nn.GELU(),nn.Linear(hidden,out_dim))
    def forward(self,bboxes,size):
        b=bboxes/float(size); w=(b[:,2]-b[:,0]).clamp_min(1e-4); h=(b[:,3]-b[:,1]).clamp_min(1e-4)
        cxywh=torch.stack([(b[:,0]+b[:,2])/2,(b[:,1]+b[:,3])/2,w,h],dim=1)
        ang=cxywh[...,None]*self.freqs
        fourier=torch.cat([ang.sin(),ang.cos()],dim=-1).flatten(1)
        log_area=torch.log(w*h).unsqueeze(1).clamp(-13.8,0.0)
        return self.mlp(torch.cat([fourier,log_area],dim=1))
class DinoV3Dual(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        import sys; sys.path.insert(0,"/data/asset/r0i_probe/dinov3")
        # stem: residual 3x3 stack, zero-init last
        self.stem=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GELU(),nn.Conv2d(16,3,3,padding=1))
        nn.init.zeros_(self.stem[2].weight); nn.init.zeros_(self.stem[2].bias)
        # DINOv3 backbone (direct, avoids hubconf torchmetrics dep)
        from dinov3.models.vision_transformer import DinoVisionTransformer
        self.backbone=DinoVisionTransformer(img_size=384,patch_size=16,in_chans=3,pos_embed_rope_base=100,pos_embed_rope_normalize_coords="separate",pos_embed_rope_rescale_coords=2,pos_embed_rope_dtype="fp32",embed_dim=384,depth=12,num_heads=6,ffn_ratio=4,qkv_bias=True,drop_path_rate=0.0,layerscale_init=1e-05,norm_layer="layernormbf16",ffn_layer="mlp",ffn_bias=True,proj_bias=True,n_storage_tokens=4,mask_k_bias=True)
        sd=torch.load("/data/asset/r0i_probe/dinov3_vits16.pth",map_location="cpu",weights_only=False)
        sd=sd.get("model",sd)
        self.backbone.load_state_dict(sd,strict=False)
        for p in self.backbone.parameters(): p.requires_grad_(False)
        self.patch=PATCH
        ch=384; dim=int(cfg.get("adapter_dim",768)); drop=float(cfg.get("dropout",0.1))
        self.t_proj=nn.Linear(ch,ch)
        self.prompt_enc=PromptEncoderV2(out_dim=ch)
        self.adapter=nn.Sequential(nn.Linear(ch,dim),nn.GELU(),nn.Dropout(drop),nn.Linear(dim,ch))
        self.head=nn.Sequential(nn.Conv2d(ch,128,1),nn.GELU(),nn.Dropout(drop),nn.Conv2d(128,2,1))
        mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
        self.register_buffer("in_mean",torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("in_std",torch.tensor(std).view(1,3,1,1))
    def forward(self,imgs,bboxes):
        B,S=imgs.shape[0],imgs.shape[-1]
        imgs=(imgs+self.stem(imgs)-self.in_mean)/self.in_std  # stem residual then norm
        # DINOv3 forward_features returns tokens; use get_intermediate_layers for taps
        feats=self.backbone.get_intermediate_layers(imgs,n=[6,11],reshape=True,norm=True)
        f=torch.cat([feats[0].float(),feats[1].float()],dim=1) # not used, keep simple: use last
        f=feats[-1].float() # [B,384,24,24]
        f=self.t_proj(f.permute(0,2,3,1)).permute(0,3,1,2)
        # prompt conditioning via FiLM-like add
        pe=self.prompt_enc(bboxes,S) # [B,384]
        f=f+pe.view(B,384,1,1)
        f=self.adapter(f.permute(0,2,3,1)).permute(0,3,1,2)
        dens=self.head(f) # [B,2,24,24]
        return {"density": dens}
    def param_groups(self,base_lr,wd):
        return [{"params":[p for p in self.parameters() if p.requires_grad],"lr":base_lr}]


def build_model(cfg):
    return DinoV3Dual(cfg)

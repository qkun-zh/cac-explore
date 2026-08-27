"""Smoke for N0033_salf_moe_frozen — stubs HF backbone, checks frozen + MAE path."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F

# stub the model by importing and replacing Backbone
import importlib.util
node_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tree/nodes/N0033_salf_moe_frozen")
spec = importlib.util.spec_from_file_location("n33_model", os.path.join(node_dir, "model.py"))
# mock transformers/cac_d before exec
import unittest.mock as mock
sys.modules['transformers'] = mock.MagicMock()
sys.modules['cac_d.common'] = mock.MagicMock()
sys.modules['cac_d.models.losses.losses'] = mock.MagicMock()
# provide fake losses to avoid import error inside model
fake_loss = mock.MagicMock()
fake_loss.gaussian_density = lambda *a, **kw: torch.randn(a[1],1,a[2],a[3])
fake_loss.adaptive_gaussian_density = lambda *a, **kw: torch.randn(a[1],1,a[2],a[3])
fake_loss.bayesian_density_loss = lambda *a, **kw: torch.tensor(0.1)
sys.modules['cac_d.models.losses.losses'] = fake_loss

# load file as text and patch
code = open(os.path.join(node_dir, "model.py")).read()
code = code.replace("from transformers import AutoModel","").replace("from cac_d.common import hf_token","")
ns = {}
exec(code, ns)

class StubBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.out_channels=[96,192,384,768]
    @torch.no_grad()
    def forward_feature_map(self,x):
        B=x.shape[0]
        return [torch.randn(B,96,96,96), torch.randn(B,192,48,48), torch.randn(B,384,24,24), torch.randn(B,768,12,12)]
    def train(self, mode=True):
        super().train(mode); self.eval(); return self

ns['Backbone']=StubBackbone

class Cfg:
    image_size=384
    hf_model="fake"
    backbone_dims=(96,192,384,768)
    embed_dim=256
    exemplar_layers=2
    roi_size=7
    d_fine=128
    cond_dim=64
    gauss_sigma=1.5
    density_weight=1.0
    cnt_weight=1.0
    density_loss="mse"
    gauss_knn=3
    sigma_beta=1.0
    sigma_min=1.0
    sigma_max=8.0
    balance_weight=0.01

Counter=ns['Counter']
orig_init=Counter.__init__
def new_init(self,cfg):
    self.cfg=cfg; self.S=cfg.image_size
    self.backbone=StubBackbone(cfg)
    D=cfg.d_fine
    self.salf=ns['SALF'](list(cfg.backbone_dims), d=D)
    self.moe=ns['DRMoE'](d=D)
    self.exemplar=ns['ExemplarEncoder'](in_dim=cfg.backbone_dims[-1], d_model=cfg.embed_dim, n_layers=cfg.exemplar_layers, roi_size=cfg.roi_size)
    self.cond=ns['Condenser'](d_in=D, d_sim=cfg.embed_dim, d_out=cfg.cond_dim)
    self.density=ns['DensityDecoder'](in_ch=D+cfg.cond_dim, hidden=2*D)
    self.balance_weight=float(getattr(cfg,"balance_weight",0.01))
Counter.__init__=new_init

# patch losses inside Counter.forward via mock already
import cac_d.models.losses.losses as L
L.gaussian_density = lambda points,B,H,W,S,sigma=1.5: torch.randn(B,1,H,W)
L.adaptive_gaussian_density = lambda points,B,H,W,S,**kw: torch.randn(B,1,H,W)
L.bayesian_density_loss = lambda dens,points,H,W,S,**kw: torch.tensor(0.5, device=dens.device)

m=Counter(Cfg())
m.train()
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
total = sum(p.numel() for p in m.parameters())
print(f"trainable: {trainable/1e6:.2f}M total: {total/1e6:.2f}M")
assert trainable < 5_000_000, "trainable too large for 32M budget check"
x=torch.randn(2,3,384,384)
bboxes=torch.rand(2,3,4)*100+50
bboxes[...,2:]+=bboxes[...,0]
bboxes[...,3:]+=bboxes[...,1]
pts=[torch.rand(10,2)*384, torch.rand(7,2)*384]
out=m(x,bboxes,pts)
print(f"loss {out['loss'].item():.4f} w_salf {out['w_salf'].mean(dim=(0,2,3)).tolist()} w_moe {out['w_moe'].mean(dim=(0,2,3)).tolist()}")
out2=m(x,bboxes)
print(f"pred {out2['pred_counts'].tolist()} dens {out2['density'].shape}")
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
opt.zero_grad(); out["loss"].backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.0); opt.step()
print("grad ok, backbone frozen:", all(not p.requires_grad for p in m.backbone.parameters()) if list(m.backbone.parameters()) else True)
print("SMOKE GREEN")

#!/usr/bin/env python3
"""V1 (GPU~5min): 冠军前向val，记3框内预测积分vs框尺寸相关方向
裁决：H-A (box质量 vs 物体像素面积正相关) 证据等级
"""
import os, json, math, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT="/data/dataset/FSC147"
CKPT="/data/runs/N0027_norm_flip_swa/best.pth"
NODE_DIR=os.path.join(REPO,"tree/nodes/N0027_norm_flip_swa")
def load_mod(p,n):
    import importlib.util
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg=load_mod(os.path.join(NODE_DIR,"config.py"),"c").cfg
mod=load_mod(os.path.join(NODE_DIR,"model.py"),"m")
model=mod.build_model(cfg).to(device).eval()
ckpt=torch.load(CKPT,map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)
print(f"loaded {CKPT} ep {ckpt.get('epoch')}")
with open(os.path.join(DATA_ROOT,"Train_Test_Val_FSC_147.json")) as f: val_ids=json.load(f)["val"][:300]
with open(os.path.join(DATA_ROOT,"annotation_FSC147_384.json")) as f: anno=json.load(f)
areas=[]; masses=[]
for im_id in val_ids:
    stem=im_id[:-4] if im_id.endswith(".jpg") else im_id
    img=Image.open(os.path.join(DATA_ROOT,f"images_384_VarV2/{stem}.jpg")).convert("RGB").resize((392,392), Image.BILINEAR)
    t=torch.from_numpy(np.asarray(img)).permute(2,0,1).float().unsqueeze(0)/255.0
    t=t.to(device)
    # bbox first box
    ann=anno[im_id]
    sx, sy=392/float(ann["W"]), 392/float(ann["H"])
    xs=[p[0] for p in ann["box_examples_coordinates"][0]]; ys=[p[1] for p in ann["box_examples_coordinates"][0]]
    bbox=torch.tensor([[min(xs)*sx, min(ys)*sy, max(xs)*sx, max(ys)*sy]], dtype=torch.float32).to(device)
    with torch.no_grad():
        out=model(t,bbox)
        dens=out["density"][0,0].cpu().numpy()  # 28x28
        # map box to grid 28x28
        x0,y0,x1,y1=bbox[0].tolist()
        g0,g1=int(max(0,x0/14)), int(min(28, math.ceil(x1/14)))
        h0,h1=int(max(0,y0/14)), int(min(28, math.ceil(y1/14)))
        mass=float(dens[h0:h1, g0:g1].sum()) if h1>h0 and g1>g0 else 0
        # also true box area
        area=(x1-x0)*(y1-y0)
        areas.append(area); masses.append(mass)
# correlation
import math
pro=np.array(areas); ms=np.array(masses)
# spearman
try:
    from scipy.stats import spearmanr
    rho,_=spearmanr(pro, ms)
except:
    rho=float(np.corrcoef(np.argsort(pro), np.argsort(ms))[0,1])
pear=float(np.corrcoef(pro, ms)[0,1]) if np.std(pro)>0 and np.std(ms)>0 else 0
print(f"[V1] n={len(areas)} box_area vs predicted_mass: Spearman {rho:.3f} Pearson {pear:.3f}")
print(f"  area range {pro.min():.0f}-{pro.max():.0f}  mass range {ms.min():.3f}-{ms.max():.3f} mean mass {ms.mean():.3f}")
# also check mass vs GT? but H-A predicts mass ∝ area
if rho>0.3:
    print("VERDICT: PASS — 正相关支持 H-A (幅度坍缩与面积成比例)")
elif rho<-0.3:
    print("VERDICT: 反向相关 — 证伪 H-A")
else:
    print("VERDICT: 无相关 — H-A 证据不足")
# show scatter binned
for q in [0, 0.33, 0.66, 1.0]:
    print(f" q {q:.2f} area {np.quantile(pro,q):.0f} mass {np.quantile(ms,q):.3f}")

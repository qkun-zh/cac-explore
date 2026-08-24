#!/usr/bin/env python3
"""V4 (CPU免费): 尾部图占用按横带统计梯度 — 透视校正够不够格
若尾部密集图 f 在图像下半部系统性偏高，则透视(近大远小)是主导非平稳源
裁决：透视校正是否值得成为新谱系种子
"""
import os, json, numpy as np, torch, torch.nn.functional as F
DATA_ROOT=os.environ.get("OIR_DATA","/data/dataset/FSC147")
GRID=28
def occ_soft(dens, thr=0.02):
    peak=float(dens.max()) if dens.size else 0
    if peak<1e-12: return np.zeros((GRID,GRID))
    occ=(dens>peak*thr).astype(np.float32)
    return F.adaptive_avg_pool2d(torch.from_numpy(occ)[None,None], (GRID,GRID))[0,0].numpy()
ids=json.load(open(os.path.join(DATA_ROOT,"Train_Test_Val_FSC_147.json")))["val"]
# tail images N>=300
tails=[]
for im_id in ids:
    stem=im_id[:-4] if im_id.endswith(".jpg") else im_id
    d=np.load(os.path.join(DATA_ROOT,f"gt_density_map_adaptive_384_VarV2/{stem}.npy"))
    if d.ndim==3: d=d[0]
    if float(d.sum())>=300:
        tails.append((im_id,d))
print(f"[V4] tail n={len(tails)} (N>=300)")
# for each tail image compute f per horizontal band (top third / middle / bottom third)
bands=[(0,9),(9,19),(19,28)]  # rows in 28 grid
for thr in [0.02]:
    grads=[]
    for im_id,d in tails:
        soft=occ_soft(d,thr)
        f_bands=[float(soft[r0:r1,:].mean()) for r0,r1 in bands]
        # gradient bottom - top
        grads.append(f_bands[2]-f_bands[0])
    print(f"thr={thr:.3f}  bottom-top median grad {np.median(grads):.3f} mean {np.mean(grads):.3f}  p(bottom>top) {np.mean([g>0 for g in grads]):.2f}")
    print(f"  band med f: top {np.median([occ_soft(d,thr)[0:9,:].mean() for _,d in tails]):.3f} mid {np.median([occ_soft(d,thr)[9:19,:].mean() for _,d in tails]):.3f} bottom {np.median([occ_soft(d,thr)[19:28,:].mean() for _,d in tails]):.3f}")
# also bulk comparison
bulks=[]
for im_id in ids[:500]:
    stem=im_id[:-4] if im_id.endswith(".jpg") else im_id
    d=np.load(os.path.join(DATA_ROOT,f"gt_density_map_adaptive_384_VarV2/{stem}.npy"))
    if d.ndim==3: d=d[0]
    if float(d.sum())<50:
        bulks.append((im_id,d))
grads_b=[]
for im_id,d in bulks[:38]:  # match tail n
    soft=occ_soft(d,0.02)
    grads_b.append(float(soft[19:28,:].mean()-soft[0:9,:].mean()))
print(f"bulk bottom-top median grad {np.median(grads_b):.3f} (vs tail {np.median(grads):.3f})")
if np.median(grads)>0.08:
    print("VERDICT: 透视信号强 — 值得做分带归一化/透视校正谱系")
else:
    print("VERDICT: 透视信号弱 — 非平稳源不是透视主导")

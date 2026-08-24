#!/usr/bin/env python3
"""V3 最优先 (CPU~15min): GT占用图算val全量f̂分布按桶分层
PASS IF bulk中位f̂<0.5 且 tail[300,∞)落入[0.45,0.75]
裁决：渗流路由能否作为独立资产移植到现有冠军
"""
import os, json, math, numpy as np, torch, torch.nn.functional as F
DATA_ROOT=os.environ.get("OIR_DATA","/data/dataset/FSC147")
GRID=28
def occ_soft(dens, thr):
    peak=float(dens.max()) if dens.size else 0
    if peak<1e-12: return np.zeros((GRID,GRID))
    occ=(dens>peak*thr).astype(np.float32)
    return F.adaptive_avg_pool2d(torch.from_numpy(occ)[None,None], (GRID,GRID))[0,0].numpy()
ids=json.load(open(os.path.join(DATA_ROOT,"Train_Test_Val_FSC_147.json")))["val"]
print(f"[V3] n={len(ids)}  thr scan 0.005-0.12")
for thr in [0.01,0.02,0.05,0.08]:
    f_list=[]
    buckets={ (0,50):[],(50,150):[],(150,300):[],(300,600):[],(600,3000):[] }
    for im_id in ids:
        stem=im_id[:-4] if im_id.endswith(".jpg") else im_id
        d=np.load(os.path.join(DATA_ROOT,f"gt_density_map_adaptive_384_VarV2/{stem}.npy"))
        if d.ndim==3: d=d[0]
        f=float(occ_soft(d,thr).mean())
        f_list.append(f)
        gt=float(d.sum())
        for lo,hi in buckets:
            if lo<=gt<hi:
                buckets[(lo,hi)].append(f); break
        # also tail 300 inf for pass criterion
    bulk=np.median(buckets[(0,50)])
    tail_vals=buckets[(300,600)]+buckets[(600,3000)]
    tail_med=np.median(tail_vals) if tail_vals else float('nan')
    passes=(bulk<0.5) and (0.45<=tail_med<=0.75)
    print(f"thr={thr:.3f} bulk[0,50) med f={bulk:.3f}  tail[300,inf) med f={tail_med:.3f}  PASS? {passes}")
    print(f"  per bucket med f: " + ", ".join([f"[{lo},{hi}):{np.median(v):.2f}(n={len(v)})" for (lo,hi),v in buckets.items() if v]))
# also report percolation interpretation
print("\n[V3] 渗流阈值 θc=0.676 判据: bulk<0.5→亚临界, tail 0.45-0.75→临界窗内")

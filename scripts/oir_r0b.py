#!/usr/bin/env python3
"""R0-B oracle 上界 — CPU 20min

GT密度图 → oracle最优阈值 → f → 反演 (r 取真值/框估计)
PASS IF val Spearman>0.9 且 [300,∞)桶 rel-err中位 <40%
FAIL ⇒ 整个估计器类出局
"""
import os, json, math, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import glob

DATA_ROOT="/data/dataset/FSC147"
VAL_IDS_JSON=os.path.join(DATA_ROOT,"Train_Test_Val_FSC_147.json")
ANNO_JSON=os.path.join(DATA_ROOT,"annotation_FSC147_384.json")

def boxes_r_est(ann, S):
    # E[r2] from 3 boxes: r = sqrt(w*h/pi) ? Actually disc radius vs box: box area ~ (2r)^2 =4r2 => r= sqrt(area)/2
    # Use mean box area
    areas=[]
    for corners in ann["box_examples_coordinates"][:3]:
        xs=[p[0] for p in corners]; ys=[p[1] for p in corners]
        w=(max(xs)-min(xs))*S/float(ann["W"])
        h=(max(ys)-min(ys))*S/float(ann["H"])
        areas.append(w*h)
    mean_area=float(np.mean(areas))
    # disc: area = pi r2 => r2 = area/pi  ; but box vs disc: if box tightly encloses disc, box area = (2r)^2=4r2 => r2=area/4
    # Which mapping? Use box area -> r2 = area /4  (square) ; then pi r2 = pi*area/4
    # We'll test both; default to area/4 => more conservative.
    # Actually exemplar box is object bounding box, not disc. Approx r ~ sqrt(area)/2.
    r2_est=mean_area/4.0
    # also compute pi*Er2 for formula
    return r2_est, mean_area

def gt_occupancy_from_density(dens, thr_frac=0.02, S=392):
    # dens: [S,S] float (image-sized density map, sum = GT count)
    # threshold at thr_frac * peak ? Use peak height
    peak=float(dens.max()) if dens.size>0 else 0
    if peak<1e-9:
        return np.zeros_like(dens, dtype=float)
    thr=peak*thr_frac
    occ=(dens > thr).astype(float)  # binary per-pixel occupancy
    # average pool to 28x28 grid (14px cell) -> soft label t_c = mean occupancy per cell
    # Use torch for pooling
    t=torch.from_numpy(occ).unsqueeze(0).unsqueeze(0).float()  # [1,1,S,S]
    pooled=F.avg_pool2d(t, kernel_size=14, stride=14)  # [1,1,28,28]
    # Actually S=392 => 28 cells, kernel 14 stride 14 gives 28x28
    soft=pooled[0,0].numpy()  # [28,28] in [0,1]
    return soft

def evaluate_thresholds(thr_list, S=392):
    with open(VAL_IDS_JSON) as f:
        ids=json.load(f)["val"]
    with open(ANNO_JSON) as f:
        anno=json.load(f)
    # pre-load GT counts and density maps paths
    gts=[]
    r2s=[]
    # we will for each thr compute Nhat per image
    results={thr: [] for thr in thr_list}
    for im_id in ids:
        stem=im_id[:-4] if im_id.endswith(".jpg") else im_id
        dens=np.load(os.path.join(DATA_ROOT,f"gt_density_map_adaptive_384_VarV2/{stem}.npy"))
        if dens.ndim==3:
            dens=dens[0]
        # density map is same size as image? It's 384 VarV2 but we resize? Actually files are 384_VarV2 sized; but we assume S=392 resizing would change density?
        # For oracle, we use dens as stored (sum = GT count) directly without resizing; coverage f derived from thresholded dens pooled to 28x28 via same logic.
        gt=float(dens.sum())
        gts.append(gt)
        ann=anno[im_id]
        r2,_=boxes_r_est(ann, 392)
        r2s.append(r2)
        # for each thr compute soft occupancy and f
        for thr in thr_list:
            soft=gt_occupancy_from_density(dens, thr_frac=thr, S=dens.shape[0])
            f=float(soft.mean())
            f=np.clip(f, 1e-6, 1-1e-6)
            # inversion
            lam_hat= -math.log(1-f) / (math.pi*max(r2,1e-6))
            A=392*392 # use fixed A for inversion (as spec)
            Nhat=lam_hat*A
            results[thr].append(Nhat)
    # compute metrics per thr
    best=None
    print(f"[R0-B] thr scan over {len(thr_list)} values, n={len(ids)} val images")
    print(f"{'thr':>6} {'spear':>6} {'med_rel%':>8} {'p50[300,inf)':>12} {'MAE':>7} {'RMSE':>7}")
    from scipy.stats import spearmanr
    import math as m
    for thr in thr_list:
        preds=np.array(results[thr])
        gts_a=np.array(gts)
        # spearman
        try:
            rho,_=spearmanr(preds, gts_a)
        except:
            rho=float(np.corrcoef(preds, gts_a)[0,1])
        # rel err median overall and in buckets
        rel=np.abs(preds-gts_a)/np.maximum(gts_a,1.0)
        med_rel=np.median(rel)*100
        # bucket [300,inf)
        mask=gts_a>=300
        if mask.sum()>0:
            med_rel_300=np.median(rel[mask])*100
            mae_300=np.mean(np.abs(preds[mask]-gts_a[mask]))
        else:
            med_rel_300=float('nan'); mae_300=float('nan')
        mae=float(np.mean(np.abs(preds-gts_a)))
        rmse=float(np.sqrt(np.mean((preds-gts_a)**2)))
        print(f"{thr:6.3f} {rho:6.3f} {med_rel:8.1f}% {med_rel_300:12.1f}% {mae:7.1f} {rmse:7.1f}")
        # track best by spearman
        if best is None or rho>best[0]:
            best=(rho, thr, med_rel, med_rel_300, mae, rmse)
    print(f"\nBest thr by Spearman: {best[1]:.3f} rho={best[0]:.3f} med_rel={best[2]:.1f}% tail300 med_rel={best[3]:.1f}% MAE={best[4]:.1f}")
    # PASS criterion: Spearman>0.9 AND tail [300,inf) median rel-err <40%
    rho_best,_,_,tail_med,_,_=best
    # also need to report per ST0 buckets table
    # compute ST0 intervals: [0,50)/[50,150)/[150,300)/[300,600)/[600,2200)
    preds_best=np.array(results[best[1]])
    gts_a=np.array(gts)
    print("\nBucket table at best thr:")
    print(f"{'GT interval':>12} {'n':>4} {'median GT/pred':>14} {'median rel%':>11} {'MAE':>7}")
    buckets=[(0,50),(50,150),(150,300),(300,600),(600,3000)]
    for lo,hi in buckets:
        mask=(gts_a>=lo)&(gts_a<hi)
        n=int(mask.sum())
        if n==0:
            continue
        gt_sub=gts_a[mask]; pr_sub=preds_best[mask]
        # median GT/pred (compression ratio)
        ratios=gt_sub/np.maximum(pr_sub,1e-6)
        med_ratio=np.median(ratios)
        rel=np.abs(pr_sub-gt_sub)/np.maximum(gt_sub,1.0)
        med_rel=np.median(rel)*100
        mae=np.mean(np.abs(pr_sub-gt_sub))
        print(f"[{lo:3d},{hi:4d}) {n:4d} {med_ratio:14.2f} {med_rel:11.1f}% {mae:7.1f}")
    print(f"\n[R0-B] criterion: Spearman>0.9 ? {rho_best>0.9} (got {rho_best:.3f}) ; tail [300,inf) med_rel<40% ? {tail_med<40} (got {tail_med:.1f}%)")
    if rho_best>0.9 and tail_med<40:
        print("VERDICT: PASS — oracle upper bound sufficient, proceed to R0-C")
    else:
        print("VERDICT: FAIL — oracle cannot reach required accuracy, kill OIR estimator class (per §1.5f2: deep sat CV 10-34% already accounts for part of 40% allowance)")
        # also explain if fail due to r estimation noise vs formula bias
        print("  Diagnosis: check if fail is due to r2 from 3 boxes (CV~0.224 → 26% noise) vs coverage bias vs density threshold artifact")

if __name__=="__main__":
    # thr scan 0.5% to 10% peak
    thr_list=[0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.07,0.10]
    evaluate_thresholds(thr_list)

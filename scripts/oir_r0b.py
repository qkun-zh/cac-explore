#!/usr/bin/env python3
"""OIR R0-B oracle 上界（修正版）— CPU ~20min，需数据集

GT 密度图 → 阈值扫描 → 软占用 f̂ → 反演。修正点：
  1. 变长宽比安全：直接 adaptive_avg_pool2d 到 28×28（f 是面积占比，仿射不变）
  2. grain 面积不再拍脑袋：扫描填充比 c∈{1/π,0.5,0.75,1.0}，Ā=c·mean(框面积@S空间)
  3. 用 R0-A 验证过的偏差校正反演：per-image κ 从 3 框尺寸离散估计（clip [1,3]）
PASS IF val Spearman>0.9 且 [300,∞) 桶 rel-err 中位 <40%（GOD §4 锁死）
"""
import os, json, math
import numpy as np
import torch
import torch.nn.functional as F

DATA_ROOT = os.environ.get("OIR_DATA", "/data/dataset/FSC147")
GRID = 28


def load_val():
    with open(os.path.join(DATA_ROOT, "Train_Test_Val_FSC_147.json")) as f:
        ids = json.load(f)["val"]
    with open(os.path.join(DATA_ROOT, "annotation_FSC147_384.json")) as f:
        anno = json.load(f)
    return ids, anno


def box_stats(ann, S=392):
    """3 框 → S 空间面积列表；返回 (mean_area, kappa_est)"""
    areas = []
    for corners in ann["box_examples_coordinates"][:3]:
        xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
        w = (max(xs) - min(xs)) * S / float(ann["W"])
        h = (max(ys) - min(ys)) * S / float(ann["H"])
        areas.append(max(w * h, 1e-6))
    m = float(np.mean(areas))
    if len(areas) >= 2 and m > 0:
        # E[A²]/E[A]² 作为面积离散代理（κ 的面积版，≥1）
        kap = float(np.mean(np.square(areas))) / m ** 2
    else:
        kap = 1.0
    return m, min(max(kap, 1.0), 3.0)


def occupancy_soft(dens, thr_frac):
    """密度图 → 阈值二值化 → 自适应池化到 GRID×GRID 软占用"""
    peak = float(dens.max()) if dens.size else 0.0
    if peak < 1e-12:
        return np.full((GRID, GRID), 1e-6)
    occ = (dens > peak * thr_frac).astype(np.float32)
    t = torch.from_numpy(occ)[None, None]
    soft = F.adaptive_avg_pool2d(t, (GRID, GRID))[0, 0].numpy()
    return np.clip(soft, 1e-6, 1 - 1e-6)


def invert_corrected(f, grain_area, kappa):
    """解 μ：1−f = e^{−μ}(1+½(κ−1)μ²)，μ=λ·Ā ⇒ λ̂=μ/Ā，N̂=λ̂·A_S"""
    target = max(1e-12, 1 - f)
    g = kappa - 1
    lo, hi = 1e-9, 50.0
    if g <= 1e-9:
        mu = -math.log(target)
    else:
        h = lambda mu: math.exp(-mu) * (1 + 0.5 * g * mu * mu) - target
        if h(hi) > 0:
            mu = hi
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if h(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            mu = 0.5 * (lo + hi)
    lam_hat = mu / grain_area
    return lam_hat * 392 * 392


def spearman(a, b):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except Exception:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])


def main():
    thr_list = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    fills = [1.0 / math.pi, 0.5, 0.75, 1.0]
    ids, anno = load_val()
    print(f"[R0-B] n={len(ids)} thr×{len(thr_list)} fill×{len(fills)} (+校正臂)")

    gts, areas, kaps = [], [], []
    dens_all = []
    for im_id in ids:
        stem = im_id[:-4] if im_id.endswith(".jpg") else im_id
        d = np.load(os.path.join(DATA_ROOT, "gt_density_map_adaptive_384_VarV2", f"{stem}.npy"))
        if d.ndim == 3:
            d = d[0]
        d = d.astype(np.float32)
        dens_all.append(d)
        gts.append(float(d.sum()))
        a, k = box_stats(anno[im_id])
        areas.append(a); kaps.append(k)

    # 预计算每个阈值的软占用均值 f（每图每阈值一次）
    Fm = {}  # (thr) -> [f_per_img]
    for thr in thr_list:
        Fm[thr] = []
        for d in dens_all:
            Fm[thr].append(float(occupancy_soft(d, thr).mean()))
        print(f"  thr={thr:.3f} scanned")

    A_S = 392.0 * 392.0
    results = []  # (label, preds, extra)
    for thr in thr_list:
        for c in fills:
            preds = [invert_corrected(min(f, 1 - 1e-6), c * a, k)
                     for f, a, k in zip(Fm[thr], areas, kaps)]
            results.append((f"thr={thr:.3f} c={c:.3f} corr", np.array(preds)))
    # 天真反演对照臂（无校正、c=1/π）
    results.append(("thr=0.020 c=0.318 NAIVE",
                    np.array([-math.log(max(1e-6, 1 - min(f, 1 - 1e-6))) / (math.pi / math.pi * math.pi * a / math.pi) * A_S
                              for f, a in zip(Fm[0.02], areas)])))

    def summarize(label, p):
        p = np.asarray(p, dtype=float)
        g = np.array(gts)
        rho = spearman(p, g)
        rel = np.abs(p - g) / np.maximum(g, 1.0)
        tail = rel[g >= 300]
        med_tail = float(np.median(tail)) * 100 if len(tail) else float("nan")
        mae = float(np.mean(np.abs(p - g)))
        rmse = float(math.sqrt(np.mean((p - g) ** 2)))
        return rho, med_tail, float(np.median(rel)) * 100, mae, rmse

    print(f"\n{'config':<26} {'spear':>6} {'tail300med%':>11} {'allmed%':>8} {'MAE':>7} {'RMSE':>7}")
    scored = []
    for label, p in results:
        rho, mt, mall, mae, rmse = summarize(label, p)
        scored.append((rho, mt, label, p))
        print(f"{label:<26} {rho:6.3f} {mt:11.1f} {mall:8.1f} {mae:7.1f} {rmse:7.1f}")

    scored.sort(key=lambda t: (-t[0], t[1]))
    rho_best, tail_best, label_best, p_best = scored[0]
    print(f"\nBest: {label_best}  Spearman={rho_best:.3f} tail_med={tail_best:.1f}%")

    g = np.array(gts)
    print("\n分桶表 @best（对照 GOD §0）:")
    print(f"{'GT 区间':>12} {'n':>4} {'med(GT/pred)':>12} {'med rel%':>9}")
    for lo, hi in [(0, 50), (50, 150), (150, 300), (300, 600), (600, 3000)]:
        m = (g >= lo) & (g < hi)
        if not m.any():
            continue
        ratio = np.median(g[m] / np.maximum(p_best[m], 1e-6))
        mr = float(np.median(np.abs(p_best[m] - g[m]) / g[m])) * 100
        print(f"[{lo:4d},{hi:4d}) {int(m.sum()):4d} {ratio:12.2f} {mr:9.1f}")

    ok_rho = rho_best > 0.9
    ok_tail = tail_best < 40.0
    print(f"\n[R0-B] Spearman>0.9? {ok_rho} ({rho_best:.3f})   tail[300,∞) med<40%? {ok_tail} ({tail_best:.1f}%)")
    print("VERDICT:", "PASS — 进入 R0-C" if (ok_rho and ok_tail) else "FAIL — oracle 上界不达标，OIR 估计器类出局")


if __name__ == "__main__":
    main()

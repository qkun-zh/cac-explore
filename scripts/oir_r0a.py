#!/usr/bin/env python3
"""OIR R0-A 合成泊松圆盘自洽 sanity（修正版）— 纯 CPU，无需数据集

三层验证：
  A1 固定半径 round-trip：公式闭合性（<0.1%，纯浮点误差）
  A2 随机半径天真反演：量化 Jensen 偏差，对照 §1.5a 一阶公式 λ̂/λ≈1−½(κ−1)(−ln(1−f))
     —— 这不是 bug，是物理；预期偏差，不判 FAIL
  A3 偏差校正反演（二阶泰勒）：1−f ≈ e^{−μ}(1+½(κ−1)μ²)，bisection 解 μ
     —— PASS IF 校正后中位 rel-err <2%（cv≤0.30 全场景）
"""
import math
import numpy as np

A_SIDE = 392.0
AREA = A_SIDE * A_SIDE


def draw_radii(rng, N, r_mean, cv_r):
    if cv_r == 0:
        return np.full(N, float(r_mean))
    sigma = math.sqrt(math.log(1 + cv_r ** 2))
    mu = math.log(r_mean) - 0.5 * sigma ** 2
    return rng.lognormal(mu, sigma, size=N)


def kappa_of(cv_r):
    """lognormal: κ=E[R⁴]/E[R²]²=exp(4σ²)，CV(R)=sqrt(e^{σ²}−1)"""
    if cv_r == 0:
        return 1.0
    sigma2 = math.log(1 + cv_r ** 2)
    return math.exp(4 * sigma2)


def naive_invert(f, Er2):
    return -math.log(max(1e-12, 1 - f)) / (math.pi * Er2)


def corrected_invert(f, Er2, kappa):
    """解 μ=λπE[R²]： 1−f = e^{−μ}(1+½(κ−1)μ²)，二阶泰勒 E[e^{−X}]≈e^{−E[X]}(1+Var(X)/2)。
    返回 λ̂ = μ/(πE[R²])。f 很低时退化为天真公式。"""
    target = max(1e-12, 1 - f)
    g = kappa - 1
    if g <= 0:
        return naive_invert(f, Er2)
    lo, hi = 1e-9, 50.0  # μ 上界 50 ⇒ f≈1−e^{−50}(1+..) 远超任何实际场景

    def h(mu):
        return math.exp(-mu) * (1 + 0.5 * g * mu * mu) - target

    if h(hi) > 0:  # 目标覆盖不可达（深饱和），钳到 hi
        return hi / (math.pi * Er2)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if h(mid) > 0:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    return mu / (math.pi * Er2)


def main():
    rng = np.random.default_rng(7)
    print("[R0-A1] 固定半径 round-trip（公式闭合性）")
    worst = 0.0
    for N, r in [(20, 10), (100, 10), (300, 10), (600, 10), (1200, 10), (300, 6), (800, 6)]:
        lam = N / AREA
        f = 1 - math.exp(-lam * math.pi * r * r)
        lam_hat = naive_invert(f, r * r)
        rel = abs(lam_hat * AREA - N) / N
        worst = max(worst, rel)
        print(f"  N={N:5d} r={r:2d} f={f:.4f} rel={rel*100:.6f}%")
    ok1 = worst < 1e-6
    print(f"  A1 {'PASS' if ok1 else 'FAIL'} (worst {worst*100:.6f}% < 1e-4%)\n")

    print("[R0-A2] 随机半径天真反演 vs §1.5a 一阶偏差公式（预期偏差≠FAIL）")
    print(f"  {'N':>5} {'r':>3} {'cv':>5} {'f':>6} {'kappa':>6} {'实测偏差%':>8} {'一阶预测%':>9}")
    rows = []
    for N, r, cv in [(100, 10, 0.224), (300, 10, 0.224), (600, 10, 0.224),
                     (300, 10, 0.30), (600, 20, 0.224), (800, 6, 0.224), (1200, 10, 0.30)]:
        radii = draw_radii(rng, N, r, cv)
        lam = N / AREA
        f = 1 - float(np.mean(np.exp(-lam * math.pi * radii ** 2)))
        Er2 = float(np.mean(radii ** 2))
        kap_true = float(np.mean(radii ** 4)) / Er2 ** 2
        lam_hat = naive_invert(f, Er2)
        bias_emp = lam_hat / lam - 1.0
        kap_pop = kappa_of(cv)
        bias_pred = -0.5 * (kap_pop - 1) * (-math.log(1 - min(f, 0.9999)))
        rows.append((N, r, cv, f, kap_true, bias_emp))
        print(f"  {N:5d} {r:3d} {cv:5.3f} {f:6.3f} {kap_true:6.3f} {bias_emp*100:8.2f} {bias_pred*100:9.2f}")

    print("\n[R0-A3] 偏差校正反演（bisection 二阶解）— PASS IF 中位 rel-err <2%")
    rels_n, rels_c = [], []
    for N, r, cv in [(50, 8, 0.0), (100, 10, 0.1), (150, 12, 0.15), (200, 10, 0.224),
                     (300, 10, 0.224), (400, 9, 0.25), (500, 7, 0.30), (600, 10, 0.224),
                     (700, 11, 0.224), (800, 6, 0.224), (900, 13, 0.20), (1000, 10, 0.25),
                     (1200, 10, 0.30), (60, 20, 0.224), (250, 5, 0.30)]:
        radii = draw_radii(rng, N, r, cv)
        lam = N / AREA
        f = 1 - float(np.mean(np.exp(-lam * math.pi * radii ** 2)))
        Er2 = float(np.mean(radii ** 2))
        kap_true = float(np.mean(radii ** 4)) / Er2 ** 2
        n_hat = naive_invert(f, Er2) * AREA
        c_hat = corrected_invert(f, Er2, kap_true) * AREA
        rels_n.append(abs(n_hat - N) / N)
        rels_c.append(abs(c_hat - N) / N)
    med_n, med_c = np.median(rels_n), np.median(rels_c)
    p95_c = np.percentile(rels_c, 95)
    print(f"  天真反演   中位 {med_n*100:.2f}%   p95 {np.percentile(rels_n,95)*100:.2f}%")
    print(f"  校正反演   中位 {med_c*100:.2f}%   p95 {p95_c*100:.2f}%")
    ok3 = med_c < 0.02
    print(f"\n[A1] {'PASS' if ok1 else 'FAIL'}   [A3 校正后中位<2%] {'PASS' if ok3 else 'FAIL'}")
    if ok1 and ok3:
        print("VERDICT: PASS — 公式闭合且校正反演可用；进入 R0-B")
    elif ok1:
        print("VERDICT: FAIL — 校正反演残差过大：检查深饱和场景（f→1 时二阶泰勒失真）或改用 Weil 三方程")
    else:
        print("VERDICT: FAIL — 公式本身不闭合，实现有 bug")


if __name__ == "__main__":
    main()

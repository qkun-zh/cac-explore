#!/usr/bin/env python3
"""Offline unit tests for Wave1 readout (no GPU, no data)."""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/cdino_run")
from src.readout import (
    adaptive_threshold,
    kde_threshold,
    readout_count,
    restore_peak_mass,
    _peak_mask,
    _nms_peaks,
)

def make_sparse(n=12, H=64, W=64, sigma=1.5, mass=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    ys = torch.randint(2, H - 2, (n,), generator=g)
    xs = torch.randint(2, W - 2, (n,), generator=g)
    d = torch.zeros(H, W)
    yy = torch.arange(H).float().unsqueeze(1)
    xx = torch.arange(W).float().unsqueeze(0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        bump = torch.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2)))
        bump = bump / bump.sum() * mass
        d += bump
    return d

def make_dense_merged(n=40, H=64, W=64, sigma=2.0, mass=0.4, seed=1):
    # crowded grid-like, each blob mass < 1 → undercount if pure integral
    d = torch.zeros(H, W)
    yy = torch.arange(H).float().unsqueeze(1)
    xx = torch.arange(W).float().unsqueeze(0)
    g = torch.Generator().manual_seed(seed)
    step = int((H * W / n) ** 0.5)
    step = max(step, 4)
    k = 0
    for y in range(4, H - 4, step):
        for x in range(4, W - 4, step):
            if k >= n:
                break
            bump = torch.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2)))
            bump = bump / bump.sum() * mass
            d += bump
            k += 1
    return d, k

def almost(a, b, tol=1e-4):
    return abs(a - b) <= tol

fails = []

def check(name, cond, detail=""):
    if cond:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name} {detail}")
        fails.append(name)

# density readout == pure integral
d = make_sparse(10)
c = readout_count(d, count_readout="density")
check("density_integral", almost(c, float(d.sum()), 1e-3), f"{c} vs {d.sum()}")

# background zeros / empty
z = torch.zeros(16, 16)
check("empty_zero", readout_count(z, "hybrid") == 0.0)

# sparse unit-mass peaks: hybrid without restore ≈ n
sp = make_sparse(15, mass=1.0, sigma=1.5)
hy = readout_count(sp, count_readout="hybrid", thresh_mode="hybrid", mass_restore="none")
pk = readout_count(sp, count_readout="peaks", thresh_mode="hybrid")
check("sparse_hybrid_near_n", 10 <= hy <= 22, f"hy={hy}")
check("sparse_peaks_near_n", 8 <= pk <= 20, f"pk={pk}")

# dense undercount recovery
dense, n_true = make_dense_merged(n=40, mass=0.4)
pure = readout_count(dense, "density")
restored = readout_count(
    dense, count_readout="hybrid", thresh_mode="hybrid",
    mass_restore="peak", mass_radius=5, mass_target=1.0,
)
peaks_only = readout_count(dense, "peaks", thresh_mode="hybrid")
check("dense_pure_undercounts", pure < n_true * 0.7, f"pure={pure} n={n_true}")
check("dense_restore_helps", restored > pure * 1.1, f"restored={restored} pure={pure}")
check("dense_restore_toward_n", restored >= pure and restored <= n_true * 1.5,
      f"restored={restored} n={n_true}")
check("dense_peaks_reasonable", peaks_only >= 10, f"peaks={peaks_only}")

# mass restore actually adds non-negative mass
coords, cnt = _nms_peaks(dense, _peak_mask(dense, 5), 3, 2, 0.0)
r = restore_peak_mass(dense, coords, radius=5, target_mass=1.0, strength=1.0)
check("restore_nonneg", float(r.min()) >= -1e-6)
check("restore_increases", float(r.sum()) >= float(dense.sum()) - 1e-4,
      f"{r.sum()} vs {dense.sum()}")

# adaptive threshold modes
vals = sp[sp > 0]
t_fix = adaptive_threshold(sp, "fixed", fixed=0.01)
t_pct = adaptive_threshold(sp, "percentile", pct=70)
t_kde = adaptive_threshold(sp, "kde")
t_hyb = adaptive_threshold(sp, "hybrid", fixed=0.005, pct=70)
check("thresh_fixed", almost(t_fix, 0.01))
check("thresh_pct_pos", t_pct > 0, f"{t_pct}")
check("thresh_kde_pos", t_kde > 0, f"{t_kde}")
check("thresh_hyb_ge_fixed", t_hyb >= 0.005 - 1e-9, f"{t_hyb}")

# kde on constant
const = torch.ones(8, 8) * 0.3
tk = kde_threshold(const[const > 0])
check("kde_constant", tk > 0, f"{tk}")

# hybrid never negative; dense safety with many peaks
check("hybrid_nonneg", readout_count(sp, "hybrid") >= 0)

# peak_weight increases count
c0 = readout_count(dense, "hybrid", mass_restore="none", peak_weight=0.0)
c1 = readout_count(dense, "hybrid", mass_restore="none", peak_weight=0.5)
check("peak_weight_adds", c1 >= c0 - 1e-6, f"c0={c0} c1={c1}")

# identical call deterministic
a = readout_count(sp, "hybrid", mass_restore="peak")
b = readout_count(sp, "hybrid", mass_restore="peak")
check("deterministic", a == b, f"{a} {b}")

# finite garbage handling
bad = torch.full((8, 8), float("nan"))
check("nan_safe", readout_count(bad, "hybrid") >= 0)

print("----")
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("ALL_PASS")

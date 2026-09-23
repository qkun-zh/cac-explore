"""Training-free count readout: adaptive threshold + peak/mass recovery.

Pure inference-side post-processing on a normalized density map.
No learned parameters; hyperparameters are fixed defaults (stable region).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _pos_values(d: torch.Tensor) -> torch.Tensor:
    return d[d > 0]


def kde_threshold(vals: torch.Tensor, nbins: int = 48) -> float:
    """Mode-valley threshold via 1D histogram KDE on positive activations.

    Returns a value near the low shoulder of the main mode so weak
    background mass is suppressed without wiping object peaks.
    """
    if vals.numel() == 0:
        return 0.0
    lo = float(vals.min())
    hi = float(vals.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return lo
    hist, edges = torch.histogram(vals, bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) * 0.5
    hist = hist.float()
    # bandwidth: Silverman-like on bin scale
    std = float(vals.std(unbiased=False))
    n = int(vals.numel())
    bin_w = (hi - lo) / nbins
    h = 1.06 * std * (n ** (-0.2)) if std > 0 and n > 1 else bin_w
    h = max(h, bin_w)
    # KDE at bin centers using histogram weights
    # dens(c) = sum_j hist_j * exp(-0.5 ((c-mu_j)/h)^2)
    z = (centers.unsqueeze(0) - centers.unsqueeze(1)) / h  # (nbins, nbins)
    kern = torch.exp(-0.5 * z * z)
    dens = kern @ hist  # (nbins,)
    mode_i = int(torch.argmax(dens))
    # walk left from mode until density drops below 25% of mode → valley/shoulder
    peak = float(dens[mode_i])
    thr_i = mode_i
    floor = 0.25 * peak
    while thr_i > 0 and float(dens[thr_i]) > floor:
        thr_i -= 1
    thr = float(centers[thr_i])
    # never below a mild floor of overall low positives
    mild = float(torch.quantile(vals, 0.20))
    return max(thr, mild, lo)


def adaptive_threshold(
    density: torch.Tensor,
    mode: str = "hybrid",
    fixed: Optional[float] = None,
    pct: float = 70.0,
) -> float:
    """Scalar threshold for sparsifying a non-negative density map."""
    vals = _pos_values(density)
    if vals.numel() == 0:
        return 0.0
    mode = (mode or "fixed").lower()
    if mode == "fixed":
        return float(fixed) if fixed is not None else 0.0
    if mode == "percentile":
        q = float(torch.clamp(torch.tensor(pct), 0.0, 99.0))
        return float(torch.quantile(vals, q / 100.0))
    if mode == "kde":
        return kde_threshold(vals)
    if mode == "hybrid":
        t_fixed = float(fixed) if fixed is not None else 0.0
        t_kde = kde_threshold(vals)
        # mild percentile as extra guard against single huge mode
        t_p = float(torch.quantile(vals, min(max(pct, 50.0), 95.0) / 100.0))
        return max(t_fixed, 0.5 * t_kde + 0.5 * t_p)
    raise ValueError(f"unknown thresh mode: {mode}")


def _peak_mask(d: torch.Tensor, k: int) -> torch.Tensor:
    if k % 2 == 0:
        k += 1
    k = max(k, 3)
    pad = k // 2
    x = d.unsqueeze(0).unsqueeze(0)
    mx = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
    return (x == mx).squeeze(0).squeeze(0) & (d > 0)


def _nms_peaks(
    d: torch.Tensor,
    peak: torch.Tensor,
    d_min: int,
    d_merge: int,
    thresh: float,
) -> Tuple[torch.Tensor, int]:
    """Greedy distance NMS on peak coordinates. Returns kept coords (M,2) and count."""
    ys, xs = torch.nonzero(peak & (d >= thresh), as_tuple=True)
    if ys.numel() == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=d.device), 0
    scores = d[ys, xs]
    order = torch.argsort(scores, descending=True)
    ys, xs, scores = ys[order], xs[order], scores[order]
    kept_y, kept_x = [], []
    r = max(int(d_min), 1)
    rm = max(int(d_merge), 0)
    H, W = d.shape
    for y, x in zip(ys.tolist(), xs.tolist()):
        ok = True
        for ky, kx in zip(kept_y, kept_x):
            if (y - ky) ** 2 + (x - kx) ** 2 <= rm * rm:
                ok = False
                break
        if ok:
            # also enforce min distance softly (drop closer weaker peaks)
            kept_y.append(y)
            kept_x.append(x)
    # second pass: drop peaks within d_min of a stronger kept peak
    final_y, final_x = [], []
    for i, (y, x) in enumerate(zip(kept_y, kept_x)):
        good = True
        for j, (ky, kx) in enumerate(zip(final_y, final_x)):
            if (y - ky) ** 2 + (x - kx) ** 2 <= r * r:
                good = False
                break
        if good:
            final_y.append(y)
            final_x.append(x)
    if not final_y:
        return torch.zeros(0, 2, dtype=torch.long, device=d.device), 0
    coords = torch.stack(
        [torch.tensor(final_y, dtype=torch.long, device=d.device),
         torch.tensor(final_x, dtype=torch.long, device=d.device)],
        dim=1,
    )
    return coords, coords.shape[0]


def restore_peak_mass(
    density: torch.Tensor,
    coords: torch.Tensor,
    radius: int,
    target_mass: float = 1.0,
    strength: float = 1.0,
) -> torch.Tensor:
    """Add missing unit mass around peaks whose local integral < target.

    Addresses dense undercount from merged blobs (ellipse unit-mass failure).
    """
    if coords.numel() == 0 or strength <= 0:
        return density
    out = density.clone()
    H, W = out.shape
    r = max(int(radius), 1)
    ys = coords[:, 0]
    xs = coords[:, 1]
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        patch = out[y0:y1, x0:x1]
        local = float(patch.sum())
        if local >= target_mass * 0.999:
            continue
        deficit = (target_mass - local) * strength
        if deficit <= 0:
            continue
        # gaussian bump centered on peak
        yy = torch.arange(y0, y1, device=out.device, dtype=out.dtype).unsqueeze(1)
        xx = torch.arange(x0, x1, device=out.device, dtype=out.dtype).unsqueeze(0)
        sig = max(r / 2.0, 0.75)
        bump = torch.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * sig * sig)))
        s = float(bump.sum())
        if s <= 0:
            continue
        out[y0:y1, x0:x1] = patch + bump * (deficit / s)
    return out


def readout_count(
    density: torch.Tensor,
    count_readout: str = "density",
    thresh_mode: str = "hybrid",
    thresh_pct: float = 70.0,
    fixed_thresh: Optional[float] = None,
    peak_kernel: int = 5,
    peak_dmin: int = 3,
    peak_dmerge: int = 2,
    mass_restore: str = "none",
    mass_radius: int = 4,
    mass_target: float = 1.0,
    mass_strength: float = 1.0,
    peak_weight: float = 0.0,
) -> float:
    """Convert density map → count.

    Modes:
      density  : pure integral (baseline)
      peaks    : NMS peak count only
      hybrid   : sparsify → optional mass restore → max(integral, peak-aware)
    """
    if density is None:
        return 0.0
    # histogram/quantile are CPU-only on this torch build
    d = density.detach()
    if d.is_cuda:
        d = d.cpu()
    d = d.float()
    if not torch.isfinite(d).all():
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    mode = (count_readout or "density").lower()
    if mode == "density":
        if mass_restore != "none":
            thr0 = adaptive_threshold(d, mode=thresh_mode, fixed=fixed_thresh, pct=thresh_pct)
            ds0 = d.clone()
            ds0[ds0 < thr0] = 0.0
            peak0 = _peak_mask(ds0, peak_kernel)
            coords0, n0 = _nms_peaks(ds0, peak0, peak_dmin, peak_dmerge, thr0)
            ds0 = restore_peak_mass(
                ds0, coords0, radius=mass_radius,
                target_mass=mass_target, strength=mass_strength,
            )
            return float(ds0.clamp_min(0).sum().item())
        return float(d.clamp_min(0).sum().item())

    # fixed+0 means: keep map as-is (filter_background already applied upstream)
    if (thresh_mode or "").lower() == "fixed" and (fixed_thresh is None or float(fixed_thresh) <= 0.0):
        thresh = 0.0
    else:
        thresh = adaptive_threshold(d, mode=thresh_mode, fixed=fixed_thresh, pct=thresh_pct)
    ds = d.clone()
    if thresh > 0:
        ds[ds < thresh] = 0.0

    peak = _peak_mask(ds, peak_kernel)
    coords, n_peaks = _nms_peaks(ds, peak, peak_dmin, peak_dmerge, thresh)
    dens_mass = float(ds.clamp_min(0).sum().item())
    # pre-restore mean mass/peak: crowd detector (blobs merged → <<1)
    mean_pre = dens_mass / float(n_peaks) if n_peaks > 0 else 1.0

    if mass_restore != "none" and n_peaks > 0:
        ds = restore_peak_mass(
            ds, coords, radius=mass_radius,
            target_mass=mass_target, strength=mass_strength,
        )
        dens_mass = float(ds.clamp_min(0).sum().item())

    if mode == "peaks":
        return float(n_peaks)

    # hybrid: peak bonus gated by PRE-restore under-mass (crowd regime only)
    count = dens_mass
    if peak_weight and n_peaks > 0 and mean_pre < 0.85:
        gate = (0.85 - mean_pre) / 0.85
        count = dens_mass + float(peak_weight) * float(n_peaks) * gate
    # dense safety: many peaks + under-unit mass before restore
    if n_peaks >= 8 and mean_pre < 0.7:
        count = max(count, float(n_peaks) * 0.85)
    return float(max(count, 0.0))

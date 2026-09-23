"""Density post-process pipeline (reduce exemplars → ROI-norm → hard filter)."""
import torch
import torch.nn.functional as F
import torchvision.ops as ops

from src.helpers import (
    rescale_tensor,
    resize_conv_maps,
    ellipse_coverage,
)
from .state import device


def _otsu_threshold(t):
    """#22: parameter-free Otsu valley on clamp_min(0) map; 256 bins."""
    x = t.detach().float().clamp_min(0).reshape(-1)
    if x.numel() == 0:
        return 0.0
    xmin = float(x.min().item())
    xmax = float(x.max().item())
    if xmax <= xmin:
        return xmax
    hist = torch.histc(x, bins=256, min=xmin, max=xmax)
    if float(hist.sum().item()) <= 0:
        return xmax
    step = (xmax - xmin) / 256.0
    centers = xmin + (torch.arange(256, device=x.device, dtype=x.dtype) + 0.5) * step
    w0 = torch.cumsum(hist, dim=0)
    w1 = hist.sum() - w0
    s0 = torch.cumsum(hist * centers, dim=0)
    s1 = (hist * centers).sum() - s0
    valid0 = w0 > 0
    valid1 = w1 > 0
    m0 = torch.where(valid0, s0 / w0.clamp_min(1.0), torch.zeros_like(s0))
    m1 = torch.where(valid1, s1 / w1.clamp_min(1.0), torch.zeros_like(s1))
    between = w0 * w1 * (m0 - m1) ** 2
    between = torch.where(valid0 & valid1, between, torch.zeros_like(between))
    idx = int(torch.argmax(between).item())
    # keep threshold as bin lower edge (first bin kept starts at centers[idx]-step/2)
    tau = xmin + float(idx) * step
    if tau < 0:
        tau = 0.0
    return float(tau)


def _scale_bbox(bbox, ratio):
    """Map annotation bbox from image coords to grid coords via resize ratios."""
    return torch.tensor([
        bbox[0] * ratio[1], bbox[1] * ratio[0],
        bbox[2] * ratio[1], bbox[3] * ratio[0]
    ]).int()


def _roi_align_2d(map2d, scaled_bbox):
    """ROI-align a single-channel 2D map in its own box; returns pooled (1,1,h,w)."""
    osize = (
        max(1, int(scaled_bbox[3] - scaled_bbox[1])),
        max(1, int(scaled_bbox[2] - scaled_bbox[0])),
    )
    return ops.roi_align(
        map2d.unsqueeze(0).unsqueeze(0),
        [scaled_bbox.unsqueeze(0).float().to(device)],
        output_size=osize, spatial_scale=1.0,
    )


def _roi_norm_coeff(map2d, bboxes, resize_ratios, config):
    """Champion global ROI-norm: mean over boxes of ellipse (or plain) pool / scaling_coeff."""
    pooled_vals = [
        _roi_align_2d(map2d, _scale_bbox(bbox, ratio))
        for bbox, ratio in zip(bboxes, resize_ratios)
    ]
    if not pooled_vals:
        return torch.tensor(1.0, device=map2d.device)
    if config.ellipse_normalization:
        acc = sum(
            (p[0, 0] * ellipse_coverage(p.shape[-2], p.shape[-1]).to(device)).sum()
            for p in pooled_vals
        )
    else:
        acc = sum(p.sum() for p in pooled_vals)
    norm_coeff = acc / (len(pooled_vals) * config.scaling_coeff)
    if config.fixed_norm_coeff is not None:
        norm_coeff = config.fixed_norm_coeff
    return norm_coeff


def _abs_filter_thresh(pooled_feats, config):
    """Locked hard-filter threshold: (1 / max pooled spatial area) * fs."""
    area = max(f.shape[-2] * f.shape[-1] for f in pooled_feats)
    fs = float(getattr(config, "filter_thresh_scale", 1.0))
    if fs is None:
        fs = 1.0
    if fs <= 0:
        return -1.0, fs
    return (1.0 / area) * fs, fs


def _reduce_exemplar_maps(stacked, conv_maps, config, pooled_feats,
                          bboxes, resize_ratios, feats):
    """Reduce stacked (n_ex*c, H, W) maps to a single 2D density + norm_coeff.

    Paths (mutually exclusive attempt flags; champion = plain mean):
      - roi_norm_per_exemplar (#24, closed): per-exemplar minmax+ROI-norm then mean
      - per_exemplar_filter (#25, closed): per-exemplar abs cut then mean
      - mass_weight_exemplar (#26): ROI-mass-weighted mean of minmax maps
      - exemplar_reduce=max (H, closed): channel-mean then max
      - default: channel+exemplar mean (champion)
    Returns (output_2d, norm_coeff).
    """
    reduce = str(getattr(config, "exemplar_reduce", "mean")).lower()
    n_ex = len(conv_maps)
    c_per = conv_maps[0].shape[0] if n_ex else 0
    aligned = (
        n_ex > 0
        and len(bboxes) == len(resize_ratios)
        and stacked.shape[0] == n_ex * c_per
    )

    # ---- closed attempt #24: per-exemplar ROI-norm before mean ----
    _pex = (
        bool(getattr(config, "roi_norm_per_exemplar", False))
        and aligned
        and reduce != "max"
        and len(bboxes) > 0
    )
    if _pex:
        per_ex = stacked.view(n_ex, c_per, stacked.shape[-2], stacked.shape[-1])
        scale = float(config.scaling_coeff) or 1.0
        norm_list = []
        out_maps = []
        for i in range(n_ex):
            mi = per_ex[i].mean(dim=0)
            if config.use_minmax_norm:
                mi = rescale_tensor(mi)
            pooled = _roi_align_2d(mi, _scale_bbox(bboxes[i], resize_ratios[i]))
            if config.ellipse_normalization:
                n_raw = float(
                    (pooled[0, 0] * ellipse_coverage(pooled.shape[-2], pooled.shape[-1]).to(device)).sum().item()
                )
            else:
                n_raw = float(pooled.sum().item())
            n_i = float(config.fixed_norm_coeff) if config.fixed_norm_coeff is not None else n_raw / scale
            n_i = max(n_i, 1e-12)
            norm_list.append(n_i)
            out_maps.append(mi / n_i)
        output = torch.stack(out_maps, dim=0).mean(dim=0)
        print(
            f"PEX n_ex={n_ex} norms={[round(x, 4) for x in norm_list]} "
            f"mean_norm={float(sum(norm_list) / len(norm_list)):.4f} scale={scale}",
            flush=True,
        )
        # scaling absorbed into each n_i; shared /norm_coeff stays a no-op
        return output, 1.0

    # ---- closed attempt #25: per-exemplar abs hard-filter before mean ----
    _pef = (
        bool(getattr(config, "per_exemplar_filter", False))
        and aligned
        and reduce != "max"
        and n_ex > 0
    )
    if _pef:
        per_ex = stacked.view(n_ex, c_per, stacked.shape[-2], stacked.shape[-1])
        t_abs, fs = _abs_filter_thresh(pooled_feats, config)
        kept = []
        out_maps = []
        for i in range(n_ex):
            mi = per_ex[i].mean(dim=0)
            if config.use_minmax_norm:
                mi = rescale_tensor(mi)
            if config.filter_background and fs > 0:
                mi = mi.clone()
                mi[mi < t_abs] = 0
                kept.append(int((mi > 0).sum().item()))
            out_maps.append(mi)
        output = torch.stack(out_maps, dim=0).mean(dim=0)
        print(
            f"PEF n_ex={n_ex} thresh={t_abs:.6g} fs={fs} kept_per={kept}",
            flush=True,
        )
        # maps already minmax'd per exemplar; skip re-minmax; global ROI-norm follows
        return output, _roi_norm_coeff(output, bboxes, resize_ratios, config)

    # ---- #26: mass-weighted exemplar aggregation ----
    _mwe = (
        bool(getattr(config, "mass_weight_exemplar", False))
        and aligned
        and reduce == "mean"
        and n_ex > 0
        and len(bboxes) > 0
    )
    if _mwe:
        per_ex = stacked.view(n_ex, c_per, stacked.shape[-2], stacked.shape[-1])
        masses = []
        maps = []
        for i in range(n_ex):
            mi = per_ex[i].mean(dim=0)
            if config.use_minmax_norm:
                mi = rescale_tensor(mi)
            pooled = _roi_align_2d(mi, _scale_bbox(bboxes[i], resize_ratios[i]))
            if config.ellipse_normalization:
                m_i = float(
                    (pooled[0, 0] * ellipse_coverage(pooled.shape[-2], pooled.shape[-1]).to(device)).sum().item()
                )
            else:
                m_i = float(pooled.sum().item())
            masses.append(max(m_i, 0.0))
            maps.append(mi)
        m_sum = float(sum(masses))
        if m_sum < 1e-12:
            weights = [1.0 / n_ex] * n_ex
        else:
            weights = [m / m_sum for m in masses]
        stacked_w = torch.stack(
            [torch.as_tensor(w, dtype=maps[i].dtype, device=maps[i].device) * maps[i]
             for i, w in enumerate(weights)],
            dim=0,
        ).sum(dim=0)
        output = stacked_w
        if bool(getattr(config, "local_contrast", False)):
            # #31 / H0006: S / avg_pool(S) before final minmax; global z/cut unchanged.
            blurred = F.avg_pool2d(
                output.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2
            ).squeeze(0).squeeze(0)
            before = float(output.clamp_min(0).sum().item())
            output = output / (blurred + 1e-6)
            after = float(output.clamp_min(0).sum().item())
            print(
                f"LCONTRAST path=mwex soft_pre={before:.4f} soft_post={after:.4f} "
                f"max={float(output.max().item()):.4f} k=5",
                flush=True,
            )
        if config.use_minmax_norm:
            output = rescale_tensor(output)
        print(
            f"MWEx n_ex={n_ex} masses={[round(m, 4) for m in masses]} "
            f"weights={[round(w, 4) for w in weights]}",
            flush=True,
        )
        return output, _roi_norm_coeff(output, bboxes, resize_ratios, config)

    # ---- closed H: max across exemplars after channel-mean ----
    if reduce == "max" and n_ex > 0:
        per_ex = stacked.view(n_ex, c_per, stacked.shape[-2], stacked.shape[-1]).mean(dim=1)
        output = per_ex.max(dim=0).values
        if config.use_minmax_norm:
            output = rescale_tensor(output)
        return output, _roi_norm_coeff(output, bboxes, resize_ratios, config)

    # ---- champion: plain mean over stacked channels+exemplars ----
    output = stacked.mean(dim=0)
    if bool(getattr(config, "local_contrast", False)):
        # #31 / H0006: S / avg_pool(S) before minmax; global z/cut unchanged.
        blurred = F.avg_pool2d(
            output.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0).squeeze(0)
        before = float(output.clamp_min(0).sum().item())
        output = output / (blurred + 1e-6)
        after = float(output.clamp_min(0).sum().item())
        print(
            f"LCONTRAST soft_pre={before:.4f} soft_post={after:.4f} "
            f"max={float(output.max().item()):.4f} k=5",
            flush=True,
        )
    if config.use_minmax_norm:
        output = rescale_tensor(output)
    return output, _roi_norm_coeff(output, bboxes, resize_ratios, config)


def _apply_context_sim(output, feats, bboxes, config):
    """#17 (closed): TFCounter background-context fusion on ROI-normed map."""
    t_div = float(getattr(config, "context_t_div", 1.3) or 1.3)
    lam = float(getattr(config, "context_fusion_ratio", 0.5) or 0.0)
    fh, fw = int(feats.shape[-2]), int(feats.shape[-1])
    fg = torch.zeros((fh, fw), dtype=torch.bool, device=output.device)
    for b in bboxes:
        x1 = int(max(0, min(fw, float(b[0]))))
        y1 = int(max(0, min(fh, float(b[1]))))
        x2 = int(max(0, min(fw, float(b[2]))))
        y2 = int(max(0, min(fh, float(b[3]))))
        if x2 > x1 and y2 > y1:
            fg[y1:y2, x1:x2] = True
    fsim_ds = F.interpolate(
        output.unsqueeze(0).unsqueeze(0), size=(fh, fw),
        mode="bilinear", align_corners=False,
    ).squeeze(0).squeeze(0)
    fmax = float(fsim_ds.max().item()) if fsim_ds.numel() else 0.0
    T = fmax / max(t_div, 1e-6)
    if fmax > 0:
        fg = fg | (fsim_ds >= T)
    fg_ratio = float(fg.float().mean().item()) if fg.numel() else 1.0
    bg = ~fg
    applied = 0
    bsim_max = 0.0
    if bool(bg.any()) and lam != 0.0:
        bg_feat = feats[0][:, bg]
        bg_emb = bg_feat.mean(dim=1)
        bg_emb = bg_emb / bg_emb.norm().clamp_min(1e-12)
        test = feats[0] / feats[0].norm(dim=0, keepdim=True).clamp_min(1e-12)
        bsim = (bg_emb.view(-1, 1, 1) * test).sum(dim=0)
        bsim = F.interpolate(
            bsim.unsqueeze(0).unsqueeze(0), size=output.shape[-2:],
            mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)
        fsim_absmax = float(output.abs().max().item()) if output.numel() else 0.0
        bsim_absmax = float(bsim.abs().max().item()) if bsim.numel() else 0.0
        bsim_max = bsim_absmax
        if bsim_absmax > 1e-12 and fsim_absmax > 0:
            bsim = bsim * (fsim_absmax / bsim_absmax)
        if fg_ratio <= 0.5:
            output = output - lam * bsim
            applied = 1
    print(
        f"CTXSIM fg_ratio={fg_ratio:.4f} applied={applied} "
        f"lam={lam} Tdiv={t_div} bsim_absmax={bsim_max:.6g} "
        f"fsim_absmax={float(output.abs().max().item()):.6g}",
        flush=True,
    )
    return output


def _apply_hard_filter(output, norm_coeff, pooled_feats, config, prenorm_skipped):
    """Post-divide hard filter (champion fs=0.5 / #22 Otsu / #23 prenorm skip)."""
    if config.filter_background is not True:
        return output
    area = max(f.shape[-2] * f.shape[-1] for f in pooled_feats)
    scale = float(getattr(config, "filter_thresh_scale", 1.0))
    if scale is None:
        scale = 1.0
    ngate = float(getattr(config, "dense_norm_gate", 0.0) or 0.0)
    soft = float(output.clamp_min(0).sum().item())
    ncoeff = float(norm_coeff)
    fired = 0
    if ngate > 0 and ncoeff >= ngate:
        scale = float(getattr(config, "dense_fs", scale) or scale)
        fired = 1
    print(
        f"DFSCALL soft={soft:.2f} norm={ncoeff:.4f} scale={scale} fired={fired}",
        flush=True,
    )
    if prenorm_skipped:
        print(
            f"DFSCALL prenorm=1 soft={soft:.2f} norm={ncoeff:.4f} "
            f"scale={scale} fired={fired} (post-divide cut skipped)",
            flush=True,
        )
        return output
    if bool(getattr(config, "filter_otsu", False)):
        tau = _otsu_threshold(output)
        kept = float((output >= tau).float().sum().item())
        print(
            f"OTSU tau={tau:.6g} soft={soft:.2f} "
            f"kept={kept}/{output.numel()} norm={ncoeff:.4f}",
            flush=True,
        )
        output[output < tau] = 0
        return output
    if scale <= 0:
        return output
    thresh = (1.0 / area) * scale
    if bool(getattr(config, "thresh_expand", False)):
        # #28 / H0003: convex lift of kept cells at the locked cut (x -> x*(x/t)).
        # Same t as champion; only the value map above t changes shape.
        keep = output >= thresh
        expanded = output * (output / thresh)
        output = torch.where(keep, expanded, torch.zeros_like(output))
        soft2 = float(output.clamp_min(0).sum().item())
        print(
            f"THEXPD t={thresh:.6g} soft_pre={soft:.2f} soft_post={soft2:.2f} "
            f"ratio={soft2 / max(soft, 1e-12):.4f}",
            flush=True,
        )
        return output
    output[output < thresh] = 0
    return output


def _box_peak_residual(pre, post, bboxes, config):
    """#27: re-inject mass the hard filter wiped inside each exemplar box
    (residual = max(0, pre_box_sum - post_box_sum)) at the argmax cell of
    ``pre`` within that box. Parameter-free; orthogonal to MWEx aggregation."""
    if not bool(getattr(config, "box_peak_residual", False)):
        return post
    out = post.clone()
    height, width = out.shape[-2:]
    for bbox in bboxes:
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(width, int(bbox[2]))
        y2 = min(height, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        region_pre = pre[y1:y2, x1:x2].clamp_min(0)
        region_post = out[y1:y2, x1:x2].clamp_min(0)
        residual = float(region_pre.sum().item() - region_post.sum().item())
        if residual <= 0:
            continue
        peak = int(torch.argmax(region_pre.reshape(-1)).item())
        ry, rx = divmod(peak, region_pre.shape[-1])
        out[y1 + ry, x1 + rx] = out[y1 + ry, x1 + rx] + residual
    return out


def _box_cell_mask(bboxes, height, width, device):
    """Boolean mask of cells covered by any exemplar box (grid coords)."""
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    for bbox in bboxes:
        x1 = max(0, min(width, int(bbox[0])))
        y1 = max(0, min(height, int(bbox[1])))
        x2 = max(0, min(width, int(bbox[2])))
        y2 = max(0, min(height, int(bbox[3])))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def _bg_sub_integral(output, bboxes, config):
    """#29: subtract exterior median floor, restore pre-subtract box integral.

    Sparse overcount often comes from a diffuse exterior floor that the locked
    hard filter only partly removes. Dense undercount benefits when box mass is
    held fixed and peak cells re-scale after the floor drops out.
    """
    if not bool(getattr(config, "bg_sub_integral", False)):
        return output
    if not bboxes or output.numel() == 0:
        return output
    h, w = int(output.shape[-2]), int(output.shape[-1])
    in_box = _box_cell_mask(bboxes, h, w, output.device)
    exterior = ~in_box
    if not bool(exterior.any()) or not bool(in_box.any()):
        return output
    bg = float(output[exterior].median().item())
    box_before = float(output[in_box].clamp_min(0).sum().item())
    adjusted = (output - bg).clamp_min(0)
    box_after = float(adjusted[in_box].sum().item())
    if box_after > 1e-12 and box_before > 1e-12:
        adjusted = adjusted * (box_before / box_after)
    soft_before = float(output.clamp_min(0).sum().item())
    soft_after = float(adjusted.clamp_min(0).sum().item())
    print(
        f"BGSUB bg={bg:.6g} box0={box_before:.4f} box1={box_after:.4f} "
        f"soft0={soft_before:.2f} soft1={soft_after:.2f} "
        f"ratio={soft_after / max(soft_before, 1e-12):.4f}",
        flush=True,
    )
    return adjusted


def _tile_split_apply(output, norm_coeff, pooled_feats, bboxes, resize_ratios, config):
    """#30 / H0005: always-on 2x2 tile split (LC-A style count assembly).

    Each quadrant: local minmax contrast, ROI-norm from boxes whose center falls
    in the tile (empty quadrant keeps global z), locked (1/area)*fs cut; stitch.
    Lifts crowded low-contrast tiles that a single global minmax/z leaves dim.
    """
    h, w = int(output.shape[-2]), int(output.shape[-1])
    scaled_boxes = []
    for bbox, ratio in zip(bboxes, resize_ratios):
        sb = _scale_bbox(bbox, ratio)
        scaled_boxes.append((
            float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3]),
        ))
    area = max(f.shape[-2] * f.shape[-1] for f in pooled_feats)
    fs = float(getattr(config, "filter_thresh_scale", 1.0))
    if fs is None:
        fs = 1.0
    thresh = (1.0 / area) * fs if fs > 0 else -1.0
    ys = (0, h // 2, h)
    xs = (0, w // 2, w)
    n_global = max(float(norm_coeff), 1e-12)
    soft0 = float(output.clamp_min(0).sum().item())
    out = torch.zeros_like(output)
    n_tile_list = []
    soft_tile = []
    for yi in range(2):
        for xi in range(2):
            y0, y1 = ys[yi], ys[yi + 1]
            x0, x1 = xs[xi], xs[xi + 1]
            if y1 <= y0 or x1 <= x0:
                continue
            tile = output[y0:y1, x0:x1]
            tmin = float(tile.min().item())
            tmax = float(tile.max().item())
            if tmax > tmin + 1e-12:
                local = (tile - tmin) / (tmax - tmin)
            else:
                local = tile.clone()
            boxes_t = []
            for bx1, by1, bx2, by2 in scaled_boxes:
                cx = 0.5 * (bx1 + bx2)
                cy = 0.5 * (by1 + by2)
                if not (x0 <= cx < x1 and y0 <= cy < y1):
                    continue
                b = (
                    max(0.0, bx1 - x0), max(0.0, by1 - y0),
                    min(float(x1 - x0), bx2 - x0),
                    min(float(y1 - y0), by2 - y0),
                )
                if b[2] > b[0] and b[3] > b[1]:
                    boxes_t.append(torch.tensor(b, dtype=torch.float, device=local.device))
            if boxes_t:
                ones = [(1.0, 1.0)] * len(boxes_t)
                n_t = float(_roi_norm_coeff(local, boxes_t, ones, config).item())
                n_t = max(n_t, 1e-12)
            else:
                n_t = n_global
            tile_out = local / n_t
            if thresh > 0:
                tile_out = tile_out.clone()
                tile_out[tile_out < thresh] = 0
            out[y0:y1, x0:x1] = tile_out
            n_tile_list.append(round(n_t, 4))
            soft_tile.append(round(float(tile_out.clamp_min(0).sum().item()), 2))
    soft1 = float(out.clamp_min(0).sum().item())
    print(
        f"TILESPLIT n_tiles={len(n_tile_list)} z={n_tile_list} "
        f"soft_tile={soft_tile} soft0={soft0:.2f} soft1={soft1:.2f} "
        f"ratio={soft1 / max(soft0, 1e-12):.4f} thresh={thresh:.6g}",
        flush=True,
    )
    return out


def _boxwise_counts_apply(stacked, conv_maps, pooled_feats, bboxes, resize_ratios, config):
    """#34 / H0009: closed unit-mass path per annotation box, mean of filtered maps."""
    n_ex = len(conv_maps)
    c_per = conv_maps[0].shape[0] if n_ex else 0
    if n_ex == 0 or c_per == 0 or stacked.shape[0] != n_ex * c_per:
        return None
    area = max(f.shape[-2] * f.shape[-1] for f in pooled_feats)
    fs = float(getattr(config, "filter_thresh_scale", 1.0))
    if fs is None:
        fs = 1.0
    thresh = (1.0 / area) * fs if fs > 0 else -1.0
    maps = []
    z_list = []
    soft_list = []
    for i in range(n_ex):
        mi = stacked[i * c_per:(i + 1) * c_per].mean(dim=0)
        if config.use_minmax_norm:
            mi = rescale_tensor(mi)
        n_i = _roi_norm_coeff(mi, [bboxes[i]], [resize_ratios[i]], config)
        n_i = max(float(n_i.item()), 1e-12)
        mi = mi / n_i
        if thresh > 0:
            mi = mi.clone()
            mi[mi < thresh] = 0
        maps.append(mi)
        z_list.append(round(n_i, 4))
        soft_list.append(round(float(mi.clamp_min(0).sum().item()), 2))
    output = torch.stack(maps, dim=0).mean(dim=0)
    soft = float(output.clamp_min(0).sum().item())
    print(
        f"BOXWISE n_ex={n_ex} z={z_list} soft_box={soft_list} "
        f"soft_mean={soft:.2f} thresh={thresh:.6g}",
        flush=True,
    )
    return output


def post_process_density_map(conv_maps, pooled_feats, bboxes, output_sizes, config, feats=None):
    """Density post-process pipeline.

    Order (champion):
      resize → reduce exemplars → minmax → optional #23 pre-cut → /ROI-norm
      → optional #17 CTXSIM → optional #22/#11 hard filter.
    Attempt flags only reorder/replace steps; default flags reproduce champion.
    """
    if config.use_threshold:
        output, _ = resize_conv_maps(conv_maps)
        output = output.mean(dim=0)
        if config.use_minmax_norm:
            output = rescale_tensor(output)
        thresh = torch.median(output)
        output[output < thresh] = 0
        return output

    if not (config.use_roi_norm and config.roi_norm_after_mean):
        # legacy path (roi_norm_before_mean): mean/max only; no ROI tail here
        stacked, _ = resize_conv_maps(conv_maps)
        reduce = str(getattr(config, "exemplar_reduce", "mean")).lower()
        if reduce == "max" and len(conv_maps) > 0:
            n_ex = len(conv_maps)
            c_per = conv_maps[0].shape[0]
            per_ex = stacked.view(n_ex, c_per, stacked.shape[-2], stacked.shape[-1]).mean(dim=1)
            output = per_ex.max(dim=0).values
        else:
            output = stacked.mean(dim=0)
        if config.use_minmax_norm:
            output = rescale_tensor(output)
        return output

    stacked, resize_ratios = resize_conv_maps(conv_maps)

    # #34 / H0009: per-box closed path replaces shared MWEx z + global cut.
    if bool(getattr(config, "boxwise_counts", False)):
        bw = _boxwise_counts_apply(
            stacked, conv_maps, pooled_feats, bboxes, resize_ratios, config
        )
        if bw is not None:
            return bw

    output, norm_coeff = _reduce_exemplar_maps(
        stacked, conv_maps, config, pooled_feats, bboxes, resize_ratios, feats
    )

    # #30 / H0005: always-on 2x2 tile-split replaces the single global
    # /norm_coeff + hard cut with per-quadrant contrast + ROI-norm + cut.
    if bool(getattr(config, "tile_split", False)):
        return _tile_split_apply(
            output, norm_coeff, pooled_feats, bboxes, resize_ratios, config
        )

    # #23 pre-norm hard-filter (closed): cut before /norm_coeff, skip post-cut
    prenorm = (
        bool(getattr(config, "filter_prenorm", False))
        and config.filter_background is True
        and not bool(getattr(config, "filter_otsu", False))
    )
    if prenorm:
        t_abs, scale = _abs_filter_thresh(pooled_feats, config)
        soft_pre = float(output.clamp_min(0).sum().item())
        if scale > 0:
            output[output < t_abs] = 0
        kept = float((output > 0).float().sum().item())
        print(
            f"PRENORM thresh={t_abs:.6g} soft_pre={soft_pre:.2f} "
            f"kept={kept}/{output.numel()} norm={float(norm_coeff):.4f}",
            flush=True,
        )

    output = output / norm_coeff

    if (
        bool(getattr(config, "context_aware_sim", False))
        and feats is not None
        and getattr(config, "filter_background", False)
    ):
        output = _apply_context_sim(output, feats, bboxes, config)

    output = _bg_sub_integral(output, bboxes, config)

    pre_filter = (
        output.clone()
        if bool(getattr(config, "box_peak_residual", False))
        else None
    )
    filtered = _apply_hard_filter(
        output, norm_coeff, pooled_feats, config, prenorm_skipped=prenorm
    )
    if pre_filter is not None:
        filtered = _box_peak_residual(pre_filter, filtered, bboxes, config)
    return filtered

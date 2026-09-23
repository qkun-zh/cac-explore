"""Feature extraction, DEI/ADEI crop refine, exemplar convolution helpers."""
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

import numpy as np

from src.helpers import (
    get_features,
    compute_avg_conv_filter,
    rescale_tensor,
    resize_conv_maps,
    collapse_sizes,
    closest_odd_numbers,
    rescale_bbox,
    find_local_maxima,
    ellipse_coverage,
    split_image_into_four,
    merge_feature_maps,
    create_feature_pyramide,
)
from .state import device
from .postprocess import post_process_density_map

def _l2_ch(f):
    return f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _dual_mean(f1, f2, config):
    """#19: mean of final-grid feats. L2 only when the pipeline already L2s
    (cosine_similarity/normalize_features); otherwise plain mean so identical
    weights reproduce the single-backbone path exactly."""
    if not bool(getattr(config, "dual_convnext", False)) or f2 is None:
        return f1
    if f1.shape != f2.shape:
        return f1
    if config.cosine_similarity or config.normalize_features:
        return 0.5 * (_l2_ch(f1) + _l2_ch(f2))
    return 0.5 * (f1 + f2)


def _mlvl_pick_stage(stages, final_hw):
    """#21: deepest intermediate stage with spatial size 2x final per-crop grid,
    else stages[0]. stages: list of (B,C,h,w) from the same forward as vit_out."""
    if not stages:
        raise RuntimeError("mlvl_fuse: empty stages")
    th, tw = int(final_hw[0]) * 2, int(final_hw[1]) * 2
    picked = None
    for s in stages:
        if s is None or s.dim() != 4:
            continue
        if int(s.shape[-2]) == th and int(s.shape[-1]) == tw:
            picked = s
    if picked is None:
        for s in stages:
            if s is not None and s.dim() == 4:
                picked = s
                break
    if picked is None:
        raise RuntimeError("mlvl_fuse: no 4D stage grid")
    return picked


def _mlvl_align(mid, target_hw, target_c):
    """Bilinear mid -> target_hw, then exact integer channel repeat to target_c."""
    if mid.dim() == 3:
        mid = mid.unsqueeze(0)
    if tuple(mid.shape[-2:]) != tuple(target_hw):
        mid = F.interpolate(mid, size=tuple(target_hw), mode='bilinear', align_corners=False)
    c = int(mid.shape[1])
    tc = int(target_c)
    if c == tc:
        return mid
    if tc % c == 0:
        return mid.repeat(1, tc // c, 1, 1)
    raise RuntimeError(f"mlvl_fuse: channel align fail closed midC={c} finalC={tc}")


def _extract_feats(model, img, transform, map_keys, config, adei_on):
    """Feature extraction for primary or secondary backbone (same pipeline)."""
    if adei_on:
        blocks, adei_crops, _ = _dei_blocks16(model, img, transform, map_keys)
        feats = _merge_many(blocks).unsqueeze(0)
        if config.cosine_similarity or config.normalize_features:
            feats = _l2_ch(feats)
        return feats, blocks, adei_crops
    with torch.no_grad():
        feats = get_features(
            model, img, transform, map_keys,
            divide_et_impera=config.divide_et_impera,
            divide_et_impera_twice=config.divide_et_impera_twice
        )
        if config.cosine_similarity or config.normalize_features:
            feats = _l2_ch(feats)
    return feats, None, None


def _merge_many(maps):
    """Hierarchical 2x2 merge of 4**d (C,h,w) blocks in raster order -> (C,H,W).
    Same grouping as get_features' twice-merge, generalized to depth d."""
    cur = torch.stack(list(maps), dim=0)
    while cur.shape[0] > 1:
        n = cur.shape[0]
        if n % 4 != 0:
            raise ValueError(f"block count {n} is not 4**d")
        cur = torch.stack(
            [merge_feature_maps(cur[i * 4:(i + 1) * 4]) for i in range(n // 4)],
            dim=0,
        )
    return cur[0]


def _fwd_pyr(model, crop_imgs, transform, map_keys, return_stages=False):
    """Forward PIL crops -> (N,C,h,w) features (pyramid across map_keys).
    return_stages: also return backbone multi-level stage grids from the SAME forward."""
    device_local = next(model.parameters()).device
    batch = torch.stack([transform(c) for c in crop_imgs])
    with torch.no_grad():
        outs = model(batch.to(device_local))
    feats = None
    for i, k in enumerate(map_keys):
        f = outs[k]
        if i == 0:
            feats = f
        else:
            feats = create_feature_pyramide(feats, f)
    if not return_stages:
        return feats
    stages = outs.get("stages") if isinstance(outs, dict) else None
    if not stages:
        raise RuntimeError("mlvl_fuse: backbone did not return stages")
    return feats, stages


def _dei_blocks16(model, img, transform, map_keys, mlvl=False):
    """Standard twice-DEI: 16 level-2 crops plus their (C,h,w) feature blocks.
    mlvl: also return mid-level blocks picked from the same forward (#21)."""
    l1 = split_image_into_four(img)
    crops = [m for c in l1 for m in split_image_into_four(c)]
    if mlvl:
        feats, stages = _fwd_pyr(model, crops, transform, map_keys, return_stages=True)
        mid = _mlvl_pick_stage(stages, feats.shape[-2:])
        blocks = [feats[i] for i in range(feats.shape[0])]
        mid_blocks = [mid[i] for i in range(mid.shape[0])]
        return blocks, crops, mid_blocks
    feats = _fwd_pyr(model, crops, transform, map_keys)
    blocks = [feats[i] for i in range(feats.shape[0])]
    return blocks, crops, None


def _adei_deep(model, crop_img, transform, map_keys, block_hw, depth):
    """Re-process one level-2 crop at `depth` extra splits; merged features
    are downsampled back to the standard block size so the global grid (and
    the locked filter scale) stay unchanged. Deeper splits mean each output
    cell's features were computed at finer network sampling of the scene."""
    maps = [crop_img]
    for _ in range(max(1, int(depth))):
        maps = [q for c in maps for q in split_image_into_four(c)]
    feats = _fwd_pyr(model, maps, transform, map_keys)
    sub = [feats[i] for i in range(feats.shape[0])]
    merged = _merge_many(sub)
    merged = F.interpolate(
        merged.unsqueeze(0), size=tuple(block_hw), mode='bilinear', align_corners=False
    )
    return merged.squeeze(0)


def _convnext_stage_grids(model, img, transform):
    """Full-image forward for intermediate ConvNeXt hidden_states (CEF multi-res
    candidate mining only). Returns list of (1,C,h,w); empty if backbone has no
    accessible .net (non-convnext)."""
    if not hasattr(model, "net"):
        return []
    try:
        with torch.no_grad():
            x = transform(img).unsqueeze(0).to(next(model.parameters()).device)
            out = model.net(pixel_values=x, output_hidden_states=True)
    except Exception:
        return []
    stages = []
    hs_list = getattr(out, "hidden_states", None) or []
    for hs in hs_list:
        if hs is None:
            continue
        g = None
        if hs.dim() == 4:
            g = hs
        elif hs.dim() == 3:
            b, n, c = hs.shape
            if int(n ** 0.5) ** 2 == n:
                side = int(n ** 0.5)
                g = hs.permute(0, 2, 1).reshape(b, c, side, side)
            elif int((n - 1) ** 0.5) ** 2 == n - 1:
                side = int((n - 1) ** 0.5)
                g = hs[:, 1:, :].permute(0, 2, 1).reshape(b, c, side, side)
        if g is not None and min(g.shape[-2:]) >= 4:
            stages.append(g)
    return stages


def _cef_soft_kernels(model, img, transform, feats, bboxes, config):
    """CountSE-style multi-res CEF soft exemplars (#18).
    Candidates from intermediate stages (top n_ex by cosine to stage ann-proto,
    NMS = half median ann box), projected to final grid; descriptors from final
    feats in median ann box; spectral cluster keep largest; kernels ROI on final
    feats. Returns list of (C, kh, kw) tensors for conv_maps only."""
    if len(bboxes) == 0:
        return []
    n_ex = len(bboxes)
    fh, fw = int(feats.shape[-2]), int(feats.shape[-1])
    med_w = max(1, int(round(float(np.median([float(b[2] - b[0]) for b in bboxes])))))
    med_h = max(1, int(round(float(np.median([float(b[3] - b[1]) for b in bboxes])))))
    stages = _convnext_stage_grids(model, img, transform)
    if not stages:
        stages = [feats]
        if min(fh, fw) >= 8:
            stages.append(F.avg_pool2d(feats, 2))
        if min(fh, fw) >= 16:
            stages.append(F.avg_pool2d(feats, 4))
    # annotation proto on final grid (L2 mean of ROI means)
    ann_desc = []
    for b in bboxes:
        x1 = int(max(0, min(fw, float(b[0])))); y1 = int(max(0, min(fh, float(b[1]))))
        x2 = int(max(0, min(fw, float(b[2])))); y2 = int(max(0, min(fh, float(b[3]))))
        if x2 <= x1 or y2 <= y1:
            continue
        patch = feats[0, :, y1:y2, x1:x2].mean(dim=(1, 2))
        ann_desc.append(patch / patch.norm().clamp_min(1e-12))
    if not ann_desc:
        return []
    proto = torch.stack(ann_desc, dim=0).mean(dim=0)
    proto = proto / proto.norm().clamp_min(1e-12)

    cand_final = []  # (y, x) on final grid
    for stage in stages:
        sh, sw = int(stage.shape[-2]), int(stage.shape[-1])
        # stage proto from annotation boxes scaled to stage
        s_desc = []
        for b in bboxes:
            sx1 = int(np.floor(float(b[0]) * sw / fw)); sy1 = int(np.floor(float(b[1]) * sh / fh))
            sx2 = int(np.ceil(float(b[2]) * sw / fw)); sy2 = int(np.ceil(float(b[3]) * sh / fh))
            sx1 = max(0, min(sw, sx1)); sx2 = max(0, min(sw, sx2))
            sy1 = max(0, min(sh, sy1)); sy2 = max(0, min(sh, sy2))
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            p = stage[0, :, sy1:sy2, sx1:sx2].mean(dim=(1, 2))
            s_desc.append(p / p.norm().clamp_min(1e-12))
        if not s_desc:
            continue
        s_proto = torch.stack(s_desc, dim=0).mean(dim=0)
        s_proto = s_proto / s_proto.norm().clamp_min(1e-12)
        # cosine score map: (C,h,w) . (C,) -> (h,w)
        s_feat = stage[0] / stage[0].norm(dim=0, keepdim=True).clamp_min(1e-12)
        scores = (s_proto.view(-1, 1, 1) * s_feat).sum(dim=0)  # (sh, sw)
        # NMS radius in stage cells: half median ann box scaled to stage
        nms_r = max(1, int(round(0.5 * max(med_w * sw / fw, med_h * sh / fh))))
        flat = torch.argsort(scores.reshape(-1), descending=True)
        picked = []
        for idx in flat.tolist():
            y, x = divmod(int(idx), sw)
            ok = True
            for py, px in picked:
                if (y - py) ** 2 + (x - px) ** 2 < nms_r * nms_r:
                    ok = False
                    break
            if not ok:
                continue
            picked.append((y, x))
            fy = int(round((y + 0.5) * fh / sh - 0.5))
            fx = int(round((x + 0.5) * fw / sw - 0.5))
            fy = max(0, min(fh - 1, fy)); fx = max(0, min(fw - 1, fx))
            cand_final.append((fy, fx))
            if len(picked) >= n_ex:
                break
    if not cand_final:
        return []
    # dedup on final grid
    cand_final = list(dict.fromkeys(cand_final))
    # descriptors from final feats in median ann box
    descs = []
    locs = []
    for y, x in cand_final:
        x1 = max(0, min(fw, x - med_w // 2)); x2 = max(0, min(fw, x - med_w // 2 + med_w))
        y1 = max(0, min(fh, y - med_h // 2)); y2 = max(0, min(fh, y - med_h // 2 + med_h))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        d = feats[0, :, y1:y2, x1:x2].mean(dim=(1, 2))
        d = d / d.norm().clamp_min(1e-12)
        descs.append(d)
        locs.append((y, x, x1, y1, x2, y2))
    if len(descs) <= 1:
        keep_idx = list(range(len(descs)))
    else:
        D = torch.stack(descs, dim=0).cpu().numpy()
        keep_idx = _cef_largest_cluster(D)
    # soft kernels
    kernels = []
    for i in keep_idx:
        y, x, x1, y1, x2, y2 = locs[i]
        bbox_t = torch.tensor([float(x1), float(y1), float(x2), float(y2)])
        try:
            pooled = ops.roi_align(
                feats, [bbox_t.unsqueeze(0).float().to(feats.device)],
                output_size=(int(y2 - y1), int(x2 - x1)), spatial_scale=1.0,
            )
        except Exception:
            continue
        if config.ellipse_kernel_cleaning:
            ellipse = ellipse_coverage(pooled.shape[-2], pooled.shape[-1]).unsqueeze(0).unsqueeze(0).to(feats.device)
            pooled = pooled * ellipse
        kernels.append(pooled.squeeze(0))
    print(
        f"CEF n_cand={len(cand_final)} n_keep={len(kernels)} n_ex={n_ex} "
        f"med_box={med_w}x{med_h} n_stages={len(stages)}",
        flush=True,
    )
    return kernels


def _cef_largest_cluster(D):
    """Spectral clustering on cosine affinity; keep largest cluster indices.
    n_clusters=2 when M>2 (CountSE self-tune capped at 2); fallback keep-all."""
    M = D.shape[0]
    if M <= 2:
        return list(range(M))
    try:
        from sklearn.cluster import SpectralClustering
        S = D @ D.T
        aff = np.clip((S + 1.0) * 0.5, 1e-4, 1.0)
        np.fill_diagonal(aff, 1.0)
        sc = SpectralClustering(n_clusters=2, affinity="precomputed", random_state=0, assign_labels="kmeans")
        labels = sc.fit_predict(aff)
        vals, counts = np.unique(labels, return_counts=True)
        maj = int(vals[np.argmax(counts)])
        keep = [i for i, lab in enumerate(labels) if int(lab) == maj]
        return keep if keep else list(range(M))
    except Exception:
        return list(range(M))


def _adei_select(output, grid, bboxes, config):
    """Pick level-2 crops whose pass-1 response mass is >= thresh * median
    across the 16 crops (self-referential density gate), skipping any crop
    overlapping an annotation box so the ROI norm stays on pass-1 behavior.
    Returns raster indices; empty list when nothing qualifies."""
    t = float(getattr(config, "adaptive_dei_thresh", 3.0))
    max_sel = int(getattr(config, "adaptive_dei_max", 8))
    bw = int(grid) // 4
    bh = int(grid) // 4
    oh, ow = int(output.shape[-2]), int(output.shape[-1])
    masses, excl = [], []
    for j in range(16):
        g, c = divmod(j, 4)
        col = (g % 2) * 2 * bw + (c % 2) * bw
        row = (g // 2) * 2 * bh + (c // 2) * bh
        r0 = min(oh - 1, int(row * oh / grid))
        r1 = max(r0 + 1, min(oh, int((row + bh) * oh / grid)))
        c0 = min(ow - 1, int(col * ow / grid))
        c1 = max(c0 + 1, min(ow, int((col + bw) * ow / grid)))
        masses.append(float(output[r0:r1, c0:c1].sum()))
        hit = False
        for x1, y1, x2, y2 in bboxes:
            if col < x2 and col + bw > x1 and row < y2 and row + bh > y1:
                hit = True
                break
        excl.append(hit)
    med = float(np.median(masses))
    if med <= 0:
        return []
    cand = [j for j in range(16) if not excl[j] and masses[j] >= t * med]
    cand.sort(key=lambda j: masses[j], reverse=True)
    return cand[:max_sel]


def _density_warp_cdfs(pass1_map, h, w, gamma=1.0, sigma_cells=2.0):
    """AdaCount-style rectilinear importance warp from the pass-1 map.
    Row/col mass ^gamma, Gaussian-smoothed on the output grid, turned into
    normalized CDFs and resampled to pixel resolution. Returns (Cy, Cx):
    arrays where dest_norm -> src_norm for image sampling, plus forward maps."""
    o = pass1_map
    if hasattr(o, "detach"):
        o = o.detach().float().cpu().numpy()
    else:
        o = np.asarray(o, dtype=np.float32)
    oh, ow = o.shape
    row = np.maximum(o.sum(axis=1), 0.0) ** float(gamma)
    col = np.maximum(o.sum(axis=0), 0.0) ** float(gamma)

    def _smooth(v, s):
        s = float(s)
        if s <= 1e-6 or v.size < 3:
            return v
        r = max(1, int(round(3.0 * s)))
        xs = np.arange(-r, r + 1, dtype=np.float64)
        k = np.exp(-0.5 * (xs / s) ** 2)
        k /= k.sum()
        return np.convolve(np.pad(v, r, mode='edge'), k, mode='valid')

    row = _smooth(row, sigma_cells)
    col = _smooth(col, sigma_cells)

    def _cdf1(v, n_out):
        v = v.astype(np.float64)
        cum = np.cumsum(v)
        total = cum[-1]
        gx = (np.arange(v.size) + 0.5) / v.size
        if total <= 1e-12:
            cdf = gx.copy()  # flat -> identity
        else:
            # center-aligned CDF: mass attributed to cell centers so a uniform
            # field maps exactly to identity (no half-cell lag)
            cdf = (cum - 0.5 * v) / total
            cdf = np.clip(cdf, 0.0, 1.0)
        # dest_norm -> src_norm (inverse CDF with pinned 0/1 ends)
        xp = np.concatenate([[0.0], cdf, [1.0]])
        fp = np.concatenate([[0.0], gx, [1.0]])
        # keep xp strictly increasing for np.interp
        for i in range(1, xp.size):
            if xp[i] <= xp[i - 1]:
                xp[i] = xp[i - 1] + 1e-12
        dest = (np.arange(n_out) + 0.5) / n_out
        src = np.interp(dest, xp, fp)
        # forward: src_norm -> dest_norm
        xpf = np.concatenate([[0.0], gx, [1.0]])
        fpf = np.concatenate([[0.0], cdf, [1.0]])
        for i in range(1, xpf.size):
            if xpf[i] <= xpf[i - 1]:
                xpf[i] = xpf[i - 1] + 1e-12
        return src, xpf, fpf

    Cy = _cdf1(row, h)
    Cx = _cdf1(col, w)
    return Cy, Cx


def _warp_image_and_boxes(img, entry_boxes, Cy, Cx):
    """Warp PIL image row/col-wise via inverse-CDF sampling; forward-map boxes.
    entry_boxes: list of 4-corner arrays in pixels. Cy/Cx come from
    _density_warp_cdfs: (src_norm(dest), forward_xp, forward_fp) each."""
    from scipy.ndimage import map_coordinates
    img_np = np.asarray(img, dtype=np.float32)
    h, w = img_np.shape[0], img_np.shape[1]
    src_y_n, yf_s, yf_d = Cy
    src_x_n, xf_s, xf_d = Cx

    sy = src_y_n * h - 0.5
    sx = src_x_n * w - 0.5
    cy, cx = np.meshgrid(sy, sx, indexing='ij')
    if img_np.ndim == 3:
        out = np.stack([
            map_coordinates(img_np[..., ch], [cy, cx], order=1, mode='nearest')
            for ch in range(img_np.shape[2])
        ], axis=-1)
    else:
        out = map_coordinates(img_np, [cy, cx], order=1, mode='nearest')
    warped = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    new_boxes = []
    for box in entry_boxes:
        b = np.asarray(box, dtype=np.float64).reshape(-1, 2)
        xn2 = np.interp(b[:, 0] / w, xf_s, xf_d)
        yn2 = np.interp(b[:, 1] / h, yf_s, yf_d)
        new_boxes.append(np.stack([xn2 * w, yn2 * h], axis=1).tolist())
    return warped, new_boxes


def _scale_bbox_about_center(bbox_tensor, scale, H, W):
    """Scale an x1y1x2y2 box about its center, clamp to [0,H]x[0,W] feature grid."""
    x1, y1, x2, y2 = [float(v) for v in bbox_tensor.tolist()]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    hw, hh = 0.5 * (x2 - x1) * float(scale), 0.5 * (y2 - y1) * float(scale)
    nx1, nx2 = max(0.0, cx - hw), min(float(W), cx + hw)
    ny1, ny2 = max(0.0, cy - hh), min(float(H), cy + hh)
    return torch.tensor([nx1, ny1, nx2, ny2], dtype=torch.float32)


def _conv_one_exemplar(feats, bbox_tensor, config):
    """Single-scale kernel: ROI-Align -> optional ellipse -> depthwise conv.
    Returns (output, pooled, output_size)."""
    output_size = (
        int(bbox_tensor[3] - bbox_tensor[1]),
        int(bbox_tensor[2] - bbox_tensor[0])
    )
    if output_size[0] < 1 or output_size[1] < 1:
        return None, None, None
    pooled = ops.roi_align(
        feats, [bbox_tensor.unsqueeze(0).float().to(device)],
        output_size=output_size, spatial_scale=1.0
    )
    if config.ellipse_kernel_cleaning:
        ellipse = ellipse_coverage(pooled.shape[-2], pooled.shape[-1]).unsqueeze(0).unsqueeze(0).to(device)
        pooled = pooled * ellipse
    conv_weights = pooled.view(feats.shape[1], 1, *output_size)
    conv_layer = nn.Conv2d(
        in_channels=feats.shape[1],
        out_channels=1 if config.cosine_similarity else feats.shape[1],
        kernel_size=output_size,
        padding=0,
        groups=1 if config.cosine_similarity else feats.shape[1],
        bias=False
    )
    conv_layer.weight = nn.Parameter(pooled if config.cosine_similarity else conv_weights)
    with torch.no_grad():
        output = conv_layer(feats[0])
    return output, pooled, output_size


def _conv_and_post(feats, bboxes, config):
    conv_maps = []
    pooled_features_list = []
    output_sizes = []
    rescaled_bboxes = []

    ms_on = bool(getattr(config, "multi_scale_ex", False))
    ms_scales = [float(x) for x in str(getattr(config, "multi_scale_factors", "0.75,1.0,1.25") or "").split(',') if x.strip()]
    if ms_on and not ms_scales:
        ms_scales = [1.0]
    if ms_on and 1.0 not in ms_scales:
        ms_scales = sorted(set(ms_scales + [1.0]))

    Hg, Wg = int(feats.shape[-2]), int(feats.shape[-1])

    for bbox in bboxes:
        bbox_tensor = torch.tensor(bbox)

        if config.exemplar_avg or not ms_on:
            output, pooled, output_size = _conv_one_exemplar(feats, bbox_tensor, config)
            if output is None:
                continue
        else:
            # Multi-scale exemplar (CountSE SES-inspired): kernels at fixed
            # scales about the annotation box center; pixelwise MAX across
            # scales (after resize to the scale-1.0 response size). Annotation
            # box / pooled_features / output_size stay the scale-1.0 pair so
            # ROI-norm and hard-filter semantics are unchanged.
            ref_out, pooled, output_size = _conv_one_exemplar(feats, bbox_tensor, config)
            if ref_out is None:
                continue
            best = ref_out
            ref_hw = best.shape[-2:]
            for s in ms_scales:
                if abs(float(s) - 1.0) < 1e-6:
                    continue
                sb = _scale_bbox_about_center(bbox_tensor, s, Hg, Wg)
                s_out, _, _ = _conv_one_exemplar(feats, sb, config)
                if s_out is None:
                    continue
                if s_out.shape[-2:] != ref_hw:
                    s_out = F.interpolate(
                        s_out.unsqueeze(0), size=ref_hw, mode='bilinear', align_corners=False
                    ).squeeze(0)
                best = torch.maximum(best, s_out)
            output = best

        pooled_features_list.append(pooled)

        if config.exemplar_avg:
            continue

        if config.correct_bbox_resize:
            rescaled_bbox = rescale_bbox(bbox_tensor, output, feats)
        else:
            rescaled_bbox = bbox_tensor

        rescaled_bboxes.append(rescaled_bbox)

        if config.use_roi_norm and not config.roi_norm_after_mean:
            if config.cosine_similarity:
                output = output + 1.0
            pooled_output = ops.roi_align(
                output.unsqueeze(0), [rescaled_bbox.unsqueeze(0).float().to(device)],
                output_size=output_size, spatial_scale=1.0
            )
            output = output / pooled_output.sum()

        conv_maps.append(output)
        output_sizes.append(output_size)

    if config.exemplar_avg:
        pooled = compute_avg_conv_filter(pooled_features_list)
        output_size = pooled.shape[1:]
        conv_weights = pooled.view(pooled.shape[0], 1, *output_size)

        conv_layer = nn.Conv2d(
            in_channels=feats.shape[1],
            out_channels=1 if config.cosine_similarity else feats.shape[1],
            kernel_size=output_size,
            padding=0,
            groups=1 if config.cosine_similarity else feats.shape[1],
            bias=False
        )
        conv_layer.weight = nn.Parameter(pooled.unsqueeze(0) if config.cosine_similarity else conv_weights)

        with torch.no_grad():
            output = conv_layer(feats[0])

        if config.use_roi_norm and not config.roi_norm_after_mean:
            raise NotImplementedError("ROI norm after conv_mean is not implemented for average-based filter.")

        conv_maps.append(output)
        output_sizes.append(output_size)

    output = post_process_density_map(
        conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, config,
        feats=feats,
    )
    return output, conv_maps, pooled_features_list, rescaled_bboxes, output_sizes

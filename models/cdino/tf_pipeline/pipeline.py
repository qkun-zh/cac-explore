"""Single-image inference pipeline (features → exemplars → post → optional stages)."""
import os
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

import numpy as np
import copy as _copy

from src.utils import convert_4corners_to_x1y1x2y2
from src.helpers import (
    bboxes_tointeger,
    ellipse_coverage,
    get_features,
)
from src.readout import readout_count
from . import state as _state
from .state import device
from .features import (
    _l2_ch,
    _dual_mean,
    _extract_feats,
    _dei_blocks16,
    _merge_many,
    _mlvl_align,
    _adei_select,
    _adei_deep,
    _conv_and_post,
    _cef_soft_kernels,
    _density_warp_cdfs,
    _warp_image_and_boxes,
)
from .postprocess import post_process_density_map

def process_example(
    idx, img_filename, entry, model, transform, map_keys, img_dir, density_map_dir, config, return_maps=False, gt_count=None, flip_view=False, scale_view=None,
    img_pil=None, _warp_inner=False
):
    """
    - when gt_count is not None, it is used to compute the metrics, and the density map is not used
    - flip_view: run the full pipeline on the left-right mirrored image with
      mirrored exemplar boxes (TTA second view; caller averages the counts)
    - scale_view: resize the scene by this factor before DEI (w,h used for
      annotation normalization stay the original size so bboxes remain valid;
      feature grid is unchanged - fs stays dimensionally valid)
    - img_pil / _warp_inner: density_warp recursion - outer call derives
      pass-1 CDFs, warps the image + exemplar boxes (same CDF), and re-enters
      with the warped inputs; _warp_inner prevents re-warping.
    """
    img = img_pil if img_pil is not None else Image.open(os.path.join(img_dir, img_filename)).convert('RGB')
    if flip_view:
        img = ImageOps.mirror(img)

    if density_map_dir is None:
        assert gt_count is not None, "gt_count must be provided if density_map_dir is None"
        density_map = None
    else:
        if gt_count is not None:
            print(f"Warning: gt_count is provided but density_map_dir is not None. Ignoring gt_count.")

        density_map = np.load(os.path.join(density_map_dir, f"{img_filename.split('.')[0]}.npy"))

    # Density warp (AdaCount-style): pass-1 row/col mass -> CDF -> rectilinear
    # image warp + forward-warped exemplar boxes, then the standard pipeline.
    # Grid size and locked filter_thresh_scale stay valid (image keeps w x h);
    # GT count is warp-invariant. Incompatible with flip/scale views (warped
    # coordinates must come from one CDF only).
    if (
        bool(getattr(config, "density_warp", False))
        and not _warp_inner
        and not return_maps
    ):
        assert not flip_view and scale_view is None, \
            "density_warp is incompatible with flip_view/scale_view"
        w0, h0 = img.size
        _, pass1_map = process_example(
            idx, img_filename, entry, model, transform, map_keys, img_dir,
            density_map_dir, config, return_maps=True, gt_count=None,
            flip_view=False, scale_view=None, _warp_inner=True,
        )
        Cy, Cx = _density_warp_cdfs(
            pass1_map, h0, w0,
            gamma=float(getattr(config, "density_warp_gamma", 1.0)),
            sigma_cells=float(getattr(config, "density_warp_sigma_cells", 2.0)),
        )
        warped_img, warped_boxes = _warp_image_and_boxes(
            img, entry['box_examples_coordinates'], Cy, Cx
        )
        entry_w = dict(entry)
        entry_w['box_examples_coordinates'] = warped_boxes
        return process_example(
            idx, img_filename, entry_w, model, transform, map_keys, img_dir,
            density_map_dir, config, return_maps=return_maps, gt_count=gt_count,
            flip_view=flip_view, scale_view=scale_view,
            img_pil=warped_img, _warp_inner=True,
        )

    w, h = img.size
    if scale_view is not None and abs(float(scale_view) - 1.0) > 1e-6:
        _res = getattr(Image, "Resampling", Image)
        img = img.resize(
            (max(1, int(round(w * float(scale_view)))), max(1, int(round(h * float(scale_view))))),
            _res.BICUBIC,
        )

    adei_on = (
        bool(getattr(config, "adaptive_dei", False))
        and config.divide_et_impera
        and config.divide_et_impera_twice
        and not return_maps
    )
    dual_on = bool(getattr(config, "dual_convnext", False)) and _state.MODEL2 is not None
    mlvl_on = bool(getattr(config, "mlvl_fuse", False)) and not return_maps
    blocks = None
    blocks2_init = None
    adei_crops = None
    mid_grid = None

    # Attempt #19: dual backbone — fuse final-grid feats BEFORE ROI/ADEI/post.
    if dual_on:
        assert not mlvl_on, "mlvl_fuse (#21) incompatible with dual_convnext (#19)"
        feats1, blocks, adei_crops = _extract_feats(model, img, transform, map_keys, config, adei_on)
        feats2, blocks2_init, _ = _extract_feats(_state.MODEL2, img, transform, map_keys, config, adei_on)
        feats = _dual_mean(feats1, feats2, config)
        if config.cosine_similarity or config.normalize_features:
            feats = _l2_ch(feats)
        print(
            f"DUAL mean1={float(feats1.mean()):.4f} mean2={float(feats2.mean()):.4f} "
            f"fused={float(feats.mean()):.4f} shape={tuple(feats.shape)}",
            flush=True,
        )
    elif adei_on or mlvl_on:
        if not (config.divide_et_impera and config.divide_et_impera_twice):
            raise RuntimeError("mlvl_fuse (#21) requires DEI twice (16-crop merge)")
        blocks, adei_crops, mid_blocks = _dei_blocks16(
            model, img, transform, map_keys, mlvl=mlvl_on
        )
        feats = _merge_many(blocks).unsqueeze(0)
        if mlvl_on:
            if mid_blocks is None:
                raise RuntimeError("mlvl_fuse: missing mid blocks")
            mid_raw = _merge_many(mid_blocks).unsqueeze(0)
            mid_grid = _mlvl_align(mid_raw, feats.shape[-2:], feats.shape[1])
            feats = 0.5 * (feats + mid_grid)
            print(
                f"MLVL {img_filename} final={tuple(feats.shape)} "
                f"mid_raw={tuple(mid_raw.shape)} fused_mean={float(feats.mean()):.4f}",
                flush=True,
            )
        if config.cosine_similarity or config.normalize_features:
            feats = feats / feats.norm(dim=1, keepdim=True)
    else:
        with torch.no_grad():
            feats = get_features(
                model, img, transform, map_keys,
                divide_et_impera=config.divide_et_impera,
                divide_et_impera_twice=config.divide_et_impera_twice
            )
            if config.cosine_similarity or config.normalize_features:
                feats = feats / feats.norm(dim=1, keepdim=True)

    # Process exemplars
    ex_bboxes = [convert_4corners_to_x1y1x2y2(b) for b in entry['box_examples_coordinates']]
    if config.num_exemplars is not None:
        assert config.num_exemplars > 0, "num_exemplars must be greater than 0. config.num_exemplars = " + config.num_exemplars
        ex_bboxes = ex_bboxes[:config.num_exemplars]
    bboxes = np.array([(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in ex_bboxes]) * feats.shape[-1]
    if flip_view:
        _g = float(feats.shape[-1])
        bboxes = np.array(bboxes, dtype=np.float64)
        bboxes[:, [0, 2]] = _g - bboxes[:, [2, 0]]
    bboxes = bboxes_tointeger(bboxes, config.remove_bbox_intersection)

    output, conv_maps, pooled_features_list, rescaled_bboxes, output_sizes = _conv_and_post(
        feats, bboxes, config
    )

    # ADEI: pass-1 response identifies dense level-2 crops (mass >=
    # thresh*median, annotation-overlapping crops excluded so the ROI norm
    # stays on pass-1 behavior). Selected crops are re-processed one split
    # deeper (48px -> features at ~3px/scene instead of 6px/cell), merged
    # features downsampled back to the standard 64-grid, then conv+post run
    # again. The global grid and locked filter scale are unchanged; response
    # sharpening happens only where the scene is dense (outside norm boxes).
    if adei_on:
        _sel = _adei_select(output, feats.shape[-1], bboxes, config)
        if _sel:
            if dual_on and blocks2_init is not None:
                # Keep dual path: refine selected crops on BOTH backbones,
                # merge each side, then mean of L2 final-grid feats (#19 repair).
                bp = list(blocks)
                bs = list(blocks2_init)
                _depth = int(getattr(config, "adaptive_dei_depth", 1))
                for _j in _sel:
                    bp[_j] = _adei_deep(
                        model, adei_crops[_j], transform, map_keys,
                        blocks[_j].shape[-2:], _depth
                    )
                    bs[_j] = _adei_deep(
                        _state.MODEL2, adei_crops[_j], transform, map_keys,
                        blocks[_j].shape[-2:], _depth
                    )
                mp = _merge_many(bp).unsqueeze(0)
                ms = _merge_many(bs).unsqueeze(0)
                if config.cosine_similarity or config.normalize_features:
                    mp = _l2_ch(mp)
                    ms = _l2_ch(ms)
                feats = _dual_mean(mp, ms, config)
                if config.cosine_similarity or config.normalize_features:
                    feats = _l2_ch(feats)
            else:
                blocks2 = list(blocks)
                for _j in _sel:
                    blocks2[_j] = _adei_deep(
                        model, adei_crops[_j], transform, map_keys,
                        blocks[_j].shape[-2:],
                        int(getattr(config, "adaptive_dei_depth", 1))
                    )
                feats2 = _merge_many(blocks2).unsqueeze(0)
                if mlvl_on and mid_grid is not None:
                    feats2 = 0.5 * (
                        feats2 + _mlvl_align(mid_grid, feats2.shape[-2:], feats2.shape[1])
                    )
                    print(
                        f"MLVL-ADEI {img_filename} feats2={tuple(feats2.shape)} "
                        f"fused_mean={float(feats2.mean()):.4f}",
                        flush=True,
                    )
                if config.cosine_similarity or config.normalize_features:
                    feats2 = feats2 / feats2.norm(dim=1, keepdim=True)
                feats = feats2
            output, conv_maps, pooled_features_list, rescaled_bboxes, output_sizes = _conv_and_post(
                feats, bboxes, config
            )

    # Attempt #18: CountSE multi-res CEF soft exemplar kernels (#18).
    # pooled_features_list / rescaled_bboxes stay annotation-only (ROI-norm +
    # hard-filter thresh semantics unchanged). Soft kernels join conv_maps only.
    if bool(getattr(config, "soft_exemplar_cef", False)) and not return_maps:
        soft_k = _cef_soft_kernels(model, img, transform, feats, bboxes, config)
        if soft_k:
            for k in soft_k:
                if k.numel() == 0:
                    continue
                conv_maps.append(k)
                output_sizes.append(k.shape[-2:])
            output = post_process_density_map(
                conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, config,
                feats=feats,
            )

    # Attempt #15: AdaCount residual feature modulation (feature domain).
    # Prior = same conv_maps re-posted with hard filter OFF (ROI-norm path kept),
    # then s = minmax -> s^gamma -> zero-center; feats' = feats * (1 + alpha * s).
    # Re-run conv+post once. Constants fm_alpha/fm_gamma from AdaCount paper.
    # Does NOT change filter_thresh_scale / ADEI / exemplar boxes.
    if bool(getattr(config, "feature_modulation", False)) and not return_maps:
        _cfg_prior = _copy.copy(config)
        _cfg_prior.filter_background = False
        prior = post_process_density_map(
            conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, _cfg_prior
        )
        _s = prior.detach().float().clamp_min(0)
        _mx = float(_s.max().item()) if _s.numel() else 0.0
        if _mx > 0:
            _s = _s / _mx
            _g = float(getattr(config, "fm_gamma", 2.0) or 1.0)
            if abs(_g - 1.0) > 1e-6:
                _s = _s ** _g
            _s = _s - _s.mean()
            _a = float(getattr(config, "fm_alpha", 0.9) or 0.0)
            _gain = (1.0 + _a * _s).clamp_min(0.0)
            if _gain.shape[-2:] != feats.shape[-2:]:
                _gain = F.interpolate(
                    _gain.unsqueeze(0).unsqueeze(0),
                    size=feats.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)
            feats_fm = feats * _gain.unsqueeze(0).unsqueeze(0)
            output, conv_maps, pooled_features_list, rescaled_bboxes, output_sizes = _conv_and_post(
                feats_fm, bboxes, config
            )
            feats = feats_fm

    # Attempt #16: structure-routed dense second stage (spatial dual-path read).
    # Path A = current filtered density (locked fs=0.5 hard filter).
    # Path B = same conv_maps re-posted with hard filter OFF (ROI-norm kept).
    # Dense mask = local-mass structure on Path B: avg-pooled mass >=
    # dense_struct_thresh * median(positive local mass) — image-relative,
    # reuses ADEI's established "dense" constant (1.5), NOT a free scalar fs
    # gate and NOT a global threshold switch. Final map = Path B inside the
    # structural dense regions, Path A elsewhere; density sum reads Path A
    # structure where the scene is sparse (background stays wiped).
    if bool(getattr(config, "dense_struct_stage", False)) and not return_maps:
        _cfg_open = _copy.copy(config)
        _cfg_open.filter_background = False
        out_open = post_process_density_map(
            conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, _cfg_open
        )
        _k = max(3, int(getattr(config, "dense_struct_kernel", 5)))
        if _k % 2 == 0:
            _k += 1
        _pad = _k // 2
        _local = F.avg_pool2d(
            out_open.unsqueeze(0).unsqueeze(0),
            kernel_size=_k, stride=1, padding=_pad,
        ).squeeze(0).squeeze(0)
        _thr_s = float(getattr(config, "dense_struct_thresh", 1.5) or 1.5)
        _pos = _local[_local > 0]
        if _pos.numel() == 0:
            dense_mask = torch.zeros_like(out_open, dtype=torch.bool)
        else:
            _med = float(_pos.median().item())
            dense_mask = _local >= (_thr_s * max(_med, 1e-12))
        # spatial join: open path only where structure says dense
        output = torch.where(dense_mask, out_open, output)
        print(
            f"DENSESTAGE thr={_thr_s} k={_k} "
            f"dense_frac={float(dense_mask.float().mean().item()):.4f} "
            f"sumA={float(output.clamp_min(0).sum().item()):.2f} "
            f"sumB={float(out_open.clamp_min(0).sum().item()):.2f}",
            flush=True,
        )

    # P2: transductive second pass. Harvest top-R peaks from this image's own
    # pass-1 map as extra prototype kernels (crowd-typical patches), then
    # re-run post-processing. Extra kernels fire on the crowd (outside) but
    # mismatch the isolated annotation objects (in-box contribution ~weak),
    # so the ROI norm absorbs the in-box part and the gain is differential.
    p2 = int(getattr(config, "transductive_proto", 0) or 0)
    if p2 > 0 and not return_maps:
        if not (config.use_roi_norm and config.roi_norm_after_mean):
            raise NotImplementedError("transductive_proto requires roi_norm_after_mean=True")
        half = max(1, int(getattr(config, "transductive_half", 3)))
        work = output.detach().clone()
        picks = []
        for _ in range(p2):
            if work.numel() == 0:
                break
            idx = int(torch.argmax(work).item())
            py, px = divmod(idx, work.shape[1])
            val = float(work[py, px])
            y0 = max(0, py - half); y1 = min(work.shape[0], py + half + 1)
            x0 = max(0, px - half); x1 = min(work.shape[1], px + half + 1)
            work[y0:y1, x0:x1] = 0
            if val <= 0:
                break
            picks.append((py, px))
        if picks:
            sy = feats.shape[-1] / max(1, output.shape[0])
            sx = feats.shape[-1] / max(1, output.shape[1])
            for py, px in picks:
                cx = int((px + 0.5) * sx)
                cy = int((py + 0.5) * sy)
                bx1 = min(feats.shape[-1], cx + half + 1)
                bx0 = max(0, cx - half)
                by1 = min(feats.shape[-1], cy + half + 1)
                by0 = max(0, cy - half)
                if bx1 - bx0 < 2 or by1 - by0 < 2:
                    continue
                bbox_tensor = torch.tensor([float(bx0), float(by0), float(bx1), float(by1)])
                output_size = (int(by1 - by0), int(bx1 - bx0))
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
                    extra_map = conv_layer(feats[0])
                conv_maps.append(extra_map)
                output_sizes.append(output_size)
            if len(conv_maps) > len(pooled_features_list):
                # pooled_features_list / rescaled_bboxes stay annotation-only:
                # thresh and ROI norm must keep their original semantics.
                output = post_process_density_map(
                    conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, config
                )

    if return_maps:
        return density_map, output

    if density_map is not None:
        gt_count = density_map.sum()

    count = readout_count(
        output,
        count_readout=getattr(config, "count_readout", "density"),
        thresh_mode=getattr(config, "thresh_mode", "hybrid"),
        thresh_pct=getattr(config, "thresh_pct", 70.0),
        fixed_thresh=getattr(config, "readout_fixed_thresh", None),
        peak_kernel=getattr(config, "peak_kernel", 5),
        peak_dmin=getattr(config, "peak_dmin", 3),
        peak_dmerge=getattr(config, "peak_dmerge", 2),
        mass_restore=getattr(config, "mass_restore", "none"),
        mass_radius=getattr(config, "mass_radius", 4),
        mass_target=getattr(config, "mass_target", 1.0),
        mass_strength=getattr(config, "mass_strength", 1.0),
        peak_weight=getattr(config, "peak_weight", 0.0),
    )
    return gt_count, float(count)

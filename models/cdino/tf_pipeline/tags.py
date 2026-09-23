"""Result-tag and per-image CSV path builders."""
import os


def build_wave1_tag(args, resize_dim):
    _wave1 = ""
    if (args.count_readout != "density" or args.mass_restore != "none"
            or args.thresh_mode not in ("hybrid",) or args.filter_thresh_scale != 1.0
            or args.exemplar_reduce != "mean"
            or args.transductive_proto != 0
            or args.tta_flip != 0
            or args.dense_hr_gate > 0
            or args.multi_scale_ex
            or args.feature_modulation
            or args.dense_struct_stage
            or args.context_aware_sim
            or args.soft_exemplar_cef
            or args.dual_convnext
            or args.dual_density
            or args.mlvl_fuse
            or args.filter_otsu
            or args.filter_prenorm
            or args.roi_norm_per_exemplar
            or args.per_exemplar_filter
            or resize_dim != 512):
        _wave1 = (
            f"_{args.count_readout}th{args.thresh_mode}p{int(args.thresh_pct)}"
            f"mr{args.mass_restore}r{args.mass_radius}t{args.mass_target}"
            f"pw{args.peak_weight}k{args.peak_kernel}dmin{args.peak_dmin}"
            f"fs{args.filter_thresh_scale}"
        )
    if args.exemplar_reduce != "mean":
        _wave1 = _wave1 + f"_er{args.exemplar_reduce}"
    if args.multi_scale_ex:
        _ms = "".join(x.replace('.', '') for x in args.multi_scale_factors.split(',') if x.strip())
        _wave1 = _wave1 + f"_msx{_ms}"
    if args.feature_modulation:
        _wave1 = _wave1 + f"_fm{args.fm_alpha:g}g{args.fm_gamma:g}"
    if args.dense_struct_stage:
        _wave1 = _wave1 + f"_dss{args.dense_struct_thresh:g}k{args.dense_struct_kernel}"
    if args.context_aware_sim:
        _wave1 = _wave1 + f"_cx{args.context_fusion_ratio:g}t{args.context_t_div:g}"
    if args.soft_exemplar_cef:
        _wave1 = _wave1 + "_cef"
    if args.dual_convnext:
        _wave1 = _wave1 + "_dual"
    if args.dual_density:
        _wave1 = _wave1 + "_xd"
    if args.mlvl_fuse:
        _wave1 = _wave1 + "_mlv"
    if args.filter_otsu:
        _wave1 = _wave1 + "_otsu"
    if args.filter_prenorm:
        _wave1 = _wave1 + "_pn"
    if args.roi_norm_per_exemplar:
        _wave1 = _wave1 + "_pex"
    if args.per_exemplar_filter:
        _wave1 = _wave1 + "_pef"
    if args.mass_weight_exemplar:
        _wave1 = _wave1 + "_mwex"
    if args.box_peak_residual:
        _wave1 = _wave1 + "_bpr"
    if args.transductive_proto != 0:
        _wave1 = _wave1 + f"_p2{args.transductive_proto}h{args.transductive_half}"
    if args.tta_flip != 0:
        _wave1 = _wave1 + f"_tta{args.tta_flip}"
    if args.adaptive_dei:
        _wave1 = _wave1 + f"_adei{args.adaptive_dei_thresh}d{args.adaptive_dei_depth}"
        if args.adaptive_dei_max != 8:
            _wave1 = _wave1 + f"m{args.adaptive_dei_max}"
    if args.dense_fs_gate and args.dense_fs_gate > 0:
        _wave1 = _wave1 + f"_dfsg{args.dense_fs_gate:g}fs{args.dense_fs:g}"
    if args.dense_norm_gate and args.dense_norm_gate > 0:
        _wave1 = _wave1 + f"_dng{args.dense_norm_gate:g}fs{args.dense_fs:g}"
    if args.dense_hr_gate and args.dense_hr_gate > 0:
        _wave1 = _wave1 + f"_hr{args.dense_hr_gate:g}in{args.dense_hr_input}fs{args.dense_hr_fs:g}"
    if args.density_warp:
        _wave1 = _wave1 + f"_dwg{args.density_warp_gamma}s{args.density_warp_sigma_cells}"
    if resize_dim != 512:
        _wave1 = _wave1 + f"_in{resize_dim}"
    _scale_views = [float(x) for x in args.scale_views.split(',') if x.strip()] if args.scale_views else []
    if _scale_views:
        _wave1 = _wave1 + "_sv" + "x".join(str(v).replace('.', '') for v in _scale_views)
    _subtag = ""
    if args.subset_file:
        _subtag = "_sub" + os.path.splitext(os.path.basename(args.subset_file))[0]
    per_image_path = os.path.join(
        'results',
        f'per_image_{args.model_name}_{args.split}{_subtag}_'
        f"dei{int(args.divide_et_impera)}dei2{int(args.divide_et_impera_twice)}_"
        f"thr{int(args.use_threshold)}_"
        f"filt{int(args.filter_background)}ell{int(args.ellipse_normalization)}"
        f"{int(args.ellipse_kernel_cleaning)}{_wave1}.csv"
    )
    return _wave1, per_image_path

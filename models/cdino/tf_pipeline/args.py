"""CLI parser and experiment-flag validation (all assert gates live here)."""
import argparse
from src.helpers import str2bool


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dinov2_vitb14_reg')
    parser.add_argument('--img_dir', type=str, default='/raid/datasets/FSC147/images_384_VarV2')
    parser.add_argument('--density_map_dir', type=str, default='/raid/datasets/FSC147/gt_density_map_adaptive_384_VarV2')
    parser.add_argument('--annotation', type=str, default='annotations/annotation_FSC147_384.json')
    parser.add_argument('--splits', type=str, default='annotations/Train_Test_Val_FSC_147.json')
    parser.add_argument('--CARPK', type=str2bool, default=False)
    parser.add_argument('--log_file', type=str, default='results/results.csv')
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--subset_file', type=str, default=None, help='JSON list of image keys; overrides max_images')

    parser.add_argument('--divide_et_impera', type=str2bool, default=False)
    parser.add_argument('--divide_et_impera_twice', type=str2bool, default=False)
    parser.add_argument('--exemplar_avg', type=str2bool, default=False)
    parser.add_argument('--cosine_similarity', type=str2bool, default=False)
    parser.add_argument('--normalize_features', type=str2bool, default=False)
    parser.add_argument('--normalize_only_biggest_bbox', type=str2bool, default=False)
    parser.add_argument('--use_threshold', type=str2bool, default=False)
    parser.add_argument('--use_roi_norm', type=str2bool, default=True)
    parser.add_argument('--roi_norm_after_mean', type=str2bool, default=True)
    parser.add_argument('--use_minmax_norm', type=str2bool, default=True)
    parser.add_argument('--remove_bbox_intersection', type=str2bool, default=False)
    parser.add_argument('--correct_bbox_resize', type=str2bool, default=True)
    parser.add_argument('--scaling_coeff', type=float, default=1.0)
    parser.add_argument('--fixed_norm_coeff', type=float, default=None)
    parser.add_argument('--filter_background', type=str2bool, default=False)
    parser.add_argument('--ellipse_normalization', type=str2bool, default=False)
    parser.add_argument('--ellipse_kernel_cleaning', type=str2bool, default=False)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_exemplars', type=int, default=None,
                        help='cap annotation exemplar boxes (None = all; H0002 tests 1)')

    # Wave1: training-free readout / threshold / mass-restore
    parser.add_argument('--filter_thresh_scale', type=float, default=1.0)
    parser.add_argument('--dense_fs_gate', type=float, default=0.0,
                        help='deprecated attempt#11: pre-filter soft sum gate (refuted, keep 0)')
    parser.add_argument('--dense_norm_gate', type=float, default=0.0,
                        help='if >0, ROI-norm_coeff >= gate switches filter to dense_fs')
    parser.add_argument('--dense_fs', type=float, default=None,
                        help='filter_thresh_scale used when dense gate fires (default: keep filter_thresh_scale)')
    # attempt #13 gated high-res re-inference (pass-1 pred >= gate -> replace with HR pred)
    parser.add_argument('--dense_hr_gate', type=float, default=0.0,
                        help='if >0, re-run full pipeline at dense_hr_input when pass-1 pred >= gate; final = HR pred')
    parser.add_argument('--dense_hr_input', type=int, default=768,
                        help='HR re-inference input size (shared backbone, transform swap)')
    parser.add_argument('--dense_hr_fs', type=float, default=1.125,
                        help='filter_thresh_scale used on the HR re-inference pass')
    # dense-bin structural: aggregation across exemplar maps (mean = legacy)
    parser.add_argument('--exemplar_reduce', type=str, default='mean',
                        choices=['mean', 'max'])
    # attempt #14: multi-scale annotation-exemplar kernels (CountSE SES-style)
    parser.add_argument('--multi_scale_ex', type=str2bool, default=False,
                        help='build kernels at multi_scale_factors about each annotation box; max across scales')
    parser.add_argument('--multi_scale_factors', type=str, default='0.75,1.0,1.25',
                        help='comma-separated scales about box center (1.0 always included)')
    # attempt #15: AdaCount-style residual feature modulation (feature domain;
    # distinct from density_warp image warp which is refuted)
    parser.add_argument('--feature_modulation', type=str2bool, default=False,
                        help='residual gate feats*(1+alpha*zero-centered prior) then re-conv once')
    parser.add_argument('--fm_alpha', type=float, default=0.9,
                        help='modulation strength (AdaCount paper 0.9; locked for #15)')
    parser.add_argument('--fm_gamma', type=float, default=2.0,
                        help='AdaCount FM sharpen exponent (paper-fixed; #15 closed)')
    # attempt #16: structure-routed dense second stage (spatial dual-path)
    parser.add_argument('--dense_struct_stage', type=str2bool, default=False,
                        help='route dense map regions to unfiltered path via local-mass structure')
    parser.add_argument('--dense_struct_thresh', type=float, default=1.5,
                        help='dense structural criterion multiplier on median local mass (ADEI constant reuse)')
    parser.add_argument('--dense_struct_kernel', type=int, default=5,
                        help='local-mass pooling kernel for dense structure mask')
    # attempt #17: TFCounter context-aware similarity (background-context fusion)
    parser.add_argument('--context_aware_sim', type=str2bool, default=False,
                        help='fuse background prototype response into similarity before hard filter (TFCounter)')
    parser.add_argument('--context_fusion_ratio', type=float, default=0.5,
                        help='TFCounter FSC147 fusion_ratio (paper-locked 0.5)')
    parser.add_argument('--context_t_div', type=float, default=1.3,
                        help='TFCounter T = max(fsim)/T_div foreground threshold divisor')
    # attempt #18: CountSE multi-res CEF soft exemplar selection
    parser.add_argument('--soft_exemplar_cef', type=str2bool, default=False,
                        help='add CEF-filtered soft exemplar kernels from multi-res candidates')
    # attempt #19: 2xConvNeXt-T dual backbone (mean of L2-norm final feats)
    parser.add_argument('--dual_convnext', type=str2bool, default=False,
                        help='fuse second frozen ConvNeXt-T final-grid feats (L2 mean)')
    # attempt #20: cross-arch dual density (primary T + secondary ViT-S/16; mean of counts)
    parser.add_argument('--dual_density', type=str2bool, default=False,
                        help='run locked pipeline on dinov3-vits16 aux; pred=0.5*(pred1+pred2)')
    # attempt #21: multi-level feature fusion (same-forward mid stage + final grid)
    parser.add_argument('--mlvl_fuse', type=str2bool, default=False,
                        help='fuse penultimate multi-level stage with final-grid feats before ROI/ADEI/post')
    # attempt #22: Otsu valley hard-filter form (parameter-free; replaces abs thresh)
    parser.add_argument('--filter_otsu', type=str2bool, default=False,
                        help='#22: use Otsu tau on clamp_min(0) density instead of (1/area)*fs')
    # attempt #23: abs hard-filter BEFORE ROI-norm divide (order change)
    parser.add_argument('--filter_prenorm', type=str2bool, default=False,
                        help='#23: apply (1/area)*fs cut before /norm_coeff; skip post-divide cut')
    # attempt #24: per-exemplar ROI-norm before mean (each map / its own box sum)
    parser.add_argument('--roi_norm_per_exemplar', type=str2bool, default=False,
                        help='#24: divide each exemplar map by ROI sum in its own box, then mean')
    # attempt #25: per-exemplar abs hard-filter before mean (same locked formula on each map)
    parser.add_argument('--per_exemplar_filter', type=str2bool, default=False,
                        help='#25: apply (1/area)*fs cut to each minmax exemplar map, then mean')
    # attempt #26: mass-weighted exemplar aggregation (ROI mass -> weights before mean)
    parser.add_argument('--mass_weight_exemplar', type=str2bool, default=False,
                        help='#26: weight each minmax exemplar map by in-box ROI mass, then sum')
    # attempt #27: box-local peak residual re-injection after hard filter
    parser.add_argument('--box_peak_residual', type=str2bool, default=False,
                        help='#27: re-inject filter-wiped mass inside each exemplar box at pre-filter argmax')
    # attempt #28 / H0003: superlinear keep above locked hard-filter cut
    parser.add_argument('--thresh_expand', type=str2bool, default=False,
                        help='#28: cells >= locked cut become x*(x/t) instead of x (convex lift, cut unchanged)')
    # attempt #29 / H0004: exterior bg subtract with box-mass preserve
    parser.add_argument('--bg_sub_integral', type=str2bool, default=False,
                        help='#29: subtract exterior median bg after ROI-norm, restore box integral, then filter')
    # attempt #30 / H0005: always-on 2x2 tile-split local contrast + per-quadrant z
    parser.add_argument('--tile_split', type=str2bool, default=False,
                        help='#30: always-on 2x2 quadrants; local minmax + per-quadrant ROI-norm (empty quad uses global z), locked cut, stitch')
    # attempt #31 / H0006: pre-minmax local contrast = S / boxblur(S)
    parser.add_argument('--local_contrast', type=str2bool, default=False,
                        help='#31: divide exemplar-mean similarity by its avg-pool blur before minmax/z (locked k=5, eps=1e-6)')
    # attempt #32 / H0007: always-on input unsharp before backbone
    parser.add_argument('--input_unsharp', type=str2bool, default=False,
                        help='#32: PIL UnsharpMask on query image before features (locked r=2 p=150 t=3)')
    # attempt #34 / H0009: per-box closed unit-mass path, mean of box maps
    parser.add_argument('--boxwise_counts', type=str2bool, default=False,
                        help='#34: each box gets own minmax/z/cut; mean the filtered box maps; no shared z')
    # attempt #35 / H0010: self-guided density smooth before locked hard cut
    parser.add_argument('--guided_density', type=str2bool, default=False,
                        help='#35: guided filter on density (guide=src, locked r=2 eps=0.01) before hard filter')
    # attempt #39 / H0015: apply locked hard cut only outside annotation box union
    parser.add_argument('--outside_only_cut', type=str2bool, default=False,
                        help='#39: zero cells below thresh only outside exemplar box union; in-box kept')
    # attempt #40 / H0016: ROI-norm coefficient = median of per-box pools (not mean)
    parser.add_argument('--roi_norm_median', type=str2bool, default=False,
                        help='#40: robust z = median of per-box ellipse pool instead of mean')
    # attempt #41 / H0017: hard-cut denominator = sum of pooled box areas (not max)
    parser.add_argument('--filter_area_sum', type=str2bool, default=False,
                        help='#41: thresh = fs / sum(pooled areas) instead of fs / max(pooled area)')
    # attempt #42 / H0018: keep feature-grid boxes as-is (skip output/feat rescale)
    parser.add_argument('--skip_bbox_rescale', type=str2bool, default=False,
                        help='#42: force correct_bbox_resize off; ROI pools use feature-grid boxes')
    # attempt #43 / H0019: median half-plane suppression path (use_threshold)
    parser.add_argument('--median_cut_path', type=str2bool, default=False,
                        help='#43: force use_threshold; channel-mean then zero below map median')
    # attempt #44 / H0020: MWEx weight from full-grid response mass (not in-box ROI)
    parser.add_argument('--exemplar_global_weight', type=str2bool, default=False,
                        help='#44: MWEx weights use clamp_min(0).sum() over the whole map instead of in-box ROI mass')
    # attempt #45 / H0022: after /z, lift out-of-box positive mass to in-box parity
    parser.add_argument('--outbox_parity', type=str2bool, default=False,
                        help='#45: if outside-rectangle positive mass < inside, scale outside cells up to match inside')
    # P2: transductive second pass (0 = off); harvested kernel half-size in cells
    parser.add_argument('--transductive_proto', type=int, default=0)
    parser.add_argument('--transductive_half', type=int, default=3)
    # TTA: average counts over original + left-right mirrored view (0 = off)
    parser.add_argument('--tta_flip', type=int, default=0)
    # backbone input resolution for dinov3_convnext (default 512 = paper)
    parser.add_argument('--input_size', type=int, default=512)
    # scale-TTA: comma-separated scene-scale views, e.g. "0.8,1.25" (base 1.0 always included)
    parser.add_argument('--scale_views', type=str, default='')
    parser.add_argument('--adaptive_dei', type=str2bool, default=False,
                        help='pass-1-guided extra DEI split on dense crops (grid preserved)')
    parser.add_argument('--adaptive_dei_thresh', type=float, default=3.0,
                        help='crop mass >= thresh * median crop mass triggers deep refine')
    parser.add_argument('--adaptive_dei_depth', type=int, default=1,
                        help='extra split depth on selected crops (1 = 48px crops)')
    parser.add_argument('--adaptive_dei_max', type=int, default=8,
                        help='cap on refined crops per image')
    parser.add_argument('--density_warp', type=str2bool, default=False,
                        help='AdaCount-style pass-1 row/col density warp (rectilinear image+box warp, grid preserved)')
    parser.add_argument('--density_warp_gamma', type=float, default=1.0,
                        help='importance exponent for row/col mass (1.0 = linear mass)')
    parser.add_argument('--density_warp_sigma_cells', type=float, default=2.0,
                        help='Gaussian smoothing of row/col mass in output-grid cells')

    parser.add_argument('--count_readout', type=str, default='density',
                        choices=['density', 'peaks', 'hybrid'])
    parser.add_argument('--thresh_mode', type=str, default='hybrid',
                        choices=['fixed', 'percentile', 'kde', 'hybrid'])
    parser.add_argument('--thresh_pct', type=float, default=70.0)
    parser.add_argument('--readout_fixed_thresh', type=float, default=None)
    parser.add_argument('--peak_kernel', type=int, default=5)
    parser.add_argument('--peak_dmin', type=int, default=3)
    parser.add_argument('--peak_dmerge', type=int, default=2)
    parser.add_argument('--mass_restore', type=str, default='none',
                        choices=['none', 'peak'])
    parser.add_argument('--mass_radius', type=int, default=4)
    parser.add_argument('--mass_target', type=float, default=1.0)
    parser.add_argument('--mass_strength', type=float, default=1.0)
    parser.add_argument('--peak_weight', type=float, default=0.0)

    parser.add_argument('--save_preds_to_file', type=str2bool, default=False)
    parser.add_argument('--log_results', type=str2bool, default=True)
    parser.add_argument('--no_skip', type=str2bool, default=False)
    return parser


def validate_args(args):
    if args.num_exemplars is not None:
        assert isinstance(args.num_exemplars, int) and args.num_exemplars > 0, \
            'num_exemplars must be a positive int (or None for all boxes)'
        print(f'NUM_EXEMPLARS cap enabled: using first {args.num_exemplars} annotation box(es)',
              flush=True)
    assert not (args.dual_convnext and args.dual_density), \
        'dual_convnext (#19) and dual_density (#20) are mutually exclusive'
    assert not (args.mlvl_fuse and args.dual_convnext), \
        'mlvl_fuse (#21) incompatible with dual_convnext (#19)'
    assert not (args.mlvl_fuse and args.dual_density), \
        'mlvl_fuse (#21) incompatible with dual_density (#20)'
    if args.mlvl_fuse:
        assert args.model_name.startswith('dinov3_convnext'), \
            'mlvl_fuse (#21) requires dinov3_convnext (stages from same forward)'
        assert args.divide_et_impera and args.divide_et_impera_twice, \
            'mlvl_fuse (#21) requires DEI twice (16-crop merge)'
        assert not args.dense_hr_gate, 'mlvl_fuse incompatible with dense_hr_gate'
        assert not args.density_warp, 'mlvl_fuse incompatible with density_warp'
        assert not args.dual_convnext and not args.dual_density
        print('MLVL fuse enabled (#21): mid-stage + final mean before ROI/ADEI/post', flush=True)
    if args.filter_otsu:
        assert args.filter_background, 'filter_otsu (#22) requires filter_background True'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'filter_otsu incompatible with gate+scalar filters (#11/#12)'
        assert not args.dense_hr_gate, 'filter_otsu incompatible with dense_hr_gate (#13)'
        assert not args.mlvl_fuse and not args.dual_convnext and not args.dual_density, \
            'filter_otsu (#22) is a single-path filter-form change; no fusion flags'
        print('OTSU hard-filter form enabled (#22): tau replaces (1/area)*fs', flush=True)
    if args.filter_prenorm:
        assert args.filter_background, 'filter_prenorm (#23) requires filter_background True'
        assert not args.filter_otsu, 'filter_prenorm incompatible with filter_otsu (#22)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'filter_prenorm incompatible with gate+scalar filters (#11/#12)'
        assert not args.dense_hr_gate, 'filter_prenorm incompatible with dense_hr_gate (#13)'
        assert not args.mlvl_fuse and not args.dual_convnext and not args.dual_density, \
            'filter_prenorm (#23) is a single-path order change; no fusion flags'
        assert not args.context_aware_sim, 'filter_prenorm incompatible with context_aware_sim (#17)'
        assert not args.density_warp, 'filter_prenorm incompatible with density_warp (#10)'
        print('PRENORM hard-filter order enabled (#23): (1/area)*fs before /norm_coeff', flush=True)
    if args.roi_norm_per_exemplar:
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'roi_norm_per_exemplar (#24) requires use_roi_norm + roi_norm_after_mean'
        assert not args.filter_otsu, 'roi_norm_per_exemplar incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'roi_norm_per_exemplar incompatible with filter_prenorm (#23)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'roi_norm_per_exemplar incompatible with gate+scalar filters (#11/#12)'
        assert not args.dense_hr_gate, 'roi_norm_per_exemplar incompatible with dense_hr_gate (#13)'
        assert not args.mlvl_fuse and not args.dual_convnext and not args.dual_density, \
            'roi_norm_per_exemplar (#24) is a single-path norm change; no fusion flags'
        assert not args.context_aware_sim, 'roi_norm_per_exemplar incompatible with context_aware_sim (#17)'
        assert not args.density_warp, 'roi_norm_per_exemplar incompatible with density_warp (#10)'
        assert not args.transductive_proto, 'roi_norm_per_exemplar incompatible with transductive_proto (P2)'
        assert not args.per_exemplar_filter, 'roi_norm_per_exemplar (#24) incompatible with per_exemplar_filter (#25)'
        print('PER-EXEMPLAR ROI-norm enabled (#24): out_i = map_i / n_i then mean', flush=True)
    if args.per_exemplar_filter:
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'per_exemplar_filter (#25) requires use_roi_norm + roi_norm_after_mean'
        assert args.filter_background, 'per_exemplar_filter (#25) requires filter_background True'
        assert not args.roi_norm_per_exemplar, 'per_exemplar_filter incompatible with roi_norm_per_exemplar (#24)'
        assert not args.filter_otsu, 'per_exemplar_filter incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'per_exemplar_filter incompatible with filter_prenorm (#23)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'per_exemplar_filter incompatible with gate+scalar filters (#11/#12)'
        assert not args.dense_hr_gate, 'per_exemplar_filter incompatible with dense_hr_gate (#13)'
        assert not args.mlvl_fuse and not args.dual_convnext and not args.dual_density, \
            'per_exemplar_filter (#25) is a single-path filter-placement change; no fusion flags'
        assert not args.context_aware_sim, 'per_exemplar_filter incompatible with context_aware_sim (#17)'
        assert not args.density_warp, 'per_exemplar_filter incompatible with density_warp (#10)'
        assert not args.transductive_proto, 'per_exemplar_filter incompatible with transductive_proto (P2)'
        assert not args.feature_modulation, 'per_exemplar_filter incompatible with feature_modulation (#15)'
        assert not args.soft_exemplar_cef, 'per_exemplar_filter incompatible with soft_exemplar_cef (#18)'
        assert not args.multi_scale_ex, 'per_exemplar_filter incompatible with multi_scale_ex (#14)'
        assert args.exemplar_reduce == 'mean', 'per_exemplar_filter (#25) requires exemplar_reduce=mean'
        print('PER-EXEMPLAR FILTER enabled (#25): Filter(map_i) then mean, champion ROI-norm tail', flush=True)
    if args.mass_weight_exemplar:
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'mass_weight_exemplar (#26) requires use_roi_norm + roi_norm_after_mean'
        assert not args.roi_norm_per_exemplar, 'mass_weight_exemplar incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'mass_weight_exemplar incompatible with per_exemplar_filter (#25)'
        assert not args.filter_otsu, 'mass_weight_exemplar incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'mass_weight_exemplar incompatible with filter_prenorm (#23)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'mass_weight_exemplar incompatible with gate+scalar filters (#11/#12)'
        assert not args.dense_hr_gate, 'mass_weight_exemplar incompatible with dense_hr_gate (#13)'
        assert not args.mlvl_fuse and not args.dual_convnext and not args.dual_density, \
            'mass_weight_exemplar (#26) is a single-path aggregation change; no fusion flags'
        assert not args.context_aware_sim, 'mass_weight_exemplar incompatible with context_aware_sim (#17)'
        assert not args.density_warp, 'mass_weight_exemplar incompatible with density_warp (#10)'
        assert not args.transductive_proto, 'mass_weight_exemplar incompatible with transductive_proto (P2)'
        assert not args.feature_modulation, 'mass_weight_exemplar incompatible with feature_modulation (#15)'
        assert not args.soft_exemplar_cef, 'mass_weight_exemplar incompatible with soft_exemplar_cef (#18)'
        assert not args.multi_scale_ex, 'mass_weight_exemplar incompatible with multi_scale_ex (#14)'
        assert args.exemplar_reduce == 'mean', 'mass_weight_exemplar (#26) requires exemplar_reduce=mean'
        assert args.filter_background, 'mass_weight_exemplar (#26) requires filter_background True'
        print('MASS-WEIGHT EXEMPLAR enabled (#26): ROI-mass weights then mean, champion ROI-norm tail', flush=True)
    if args.box_peak_residual:
        assert args.mass_weight_exemplar, 'box_peak_residual (#27) stacks on default MWEx (#26)'
        assert args.filter_background, 'box_peak_residual (#27) requires filter_background True'
        assert not args.roi_norm_per_exemplar, 'box_peak_residual incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'box_peak_residual incompatible with per_exemplar_filter (#25)'
        assert not args.filter_otsu, 'box_peak_residual incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'box_peak_residual incompatible with filter_prenorm (#23)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'box_peak_residual incompatible with gate+scalar filters (#11/#12)'
        assert not args.dense_hr_gate, 'box_peak_residual incompatible with dense_hr_gate (#13)'
        assert not args.mlvl_fuse and not args.dual_convnext and not args.dual_density, \
            'box_peak_residual (#27) is a post-filter mass repair; no fusion flags'
        assert not args.context_aware_sim, 'box_peak_residual incompatible with context_aware_sim (#17)'
        assert not args.density_warp, 'box_peak_residual incompatible with density_warp (#10)'
        assert not args.transductive_proto, 'box_peak_residual incompatible with transductive_proto (P2)'
        assert not args.feature_modulation, 'box_peak_residual incompatible with feature_modulation (#15)'
        assert not args.soft_exemplar_cef, 'box_peak_residual incompatible with soft_exemplar_cef (#18)'
        assert not args.multi_scale_ex, 'box_peak_residual incompatible with multi_scale_ex (#14)'
        assert not (args.count_readout != 'density'), 'box_peak_residual (#27) requires count_readout=density'
        print('BOX PEAK RESIDUAL enabled (#27): re-inject wiped in-box mass at pre-filter argmax', flush=True)
    if args.thresh_expand:
        assert args.filter_background, 'thresh_expand (#28) requires filter_background True'
        assert not args.filter_otsu, 'thresh_expand incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'thresh_expand incompatible with filter_prenorm (#23)'
        assert not args.box_peak_residual, 'thresh_expand incompatible with box_peak_residual (#27)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'thresh_expand incompatible with gate+scalar filters (#11/#12)'
        assert not args.count_readout != 'density', 'thresh_expand (#28) requires count_readout=density'
        print('THRESH EXPAND enabled (#28): kept cells x -> x*(x/t) at locked cut t', flush=True)
    if args.bg_sub_integral:
        assert args.filter_background, 'bg_sub_integral (#29) requires filter_background True'
        assert not args.filter_otsu, 'bg_sub_integral incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'bg_sub_integral incompatible with filter_prenorm (#23)'
        assert not args.thresh_expand, 'bg_sub_integral incompatible with thresh_expand (#28)'
        assert not args.box_peak_residual, 'bg_sub_integral incompatible with box_peak_residual (#27)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'bg_sub_integral incompatible with gate+scalar filters (#11/#12)'
        assert not args.count_readout != 'density', 'bg_sub_integral (#29) requires count_readout=density'
        print('BG SUB INTEGRAL enabled (#29): exterior median bg out, box mass restored', flush=True)
    if args.tile_split:
        assert not args.filter_otsu, 'tile_split incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'tile_split incompatible with filter_prenorm (#23)'
        assert not args.roi_norm_per_exemplar, 'tile_split incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'tile_split incompatible with per_exemplar_filter (#25)'
        assert not args.box_peak_residual, 'tile_split incompatible with box_peak_residual (#27)'
        assert not args.thresh_expand, 'tile_split incompatible with thresh_expand (#28)'
        assert not args.bg_sub_integral, 'tile_split incompatible with bg_sub_integral (#29)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'tile_split incompatible with gate+scalar filters (#11/#12)'
        assert not args.context_aware_sim, 'tile_split incompatible with context_aware_sim (#17)'
        assert not args.dense_struct_stage, 'tile_split incompatible with dense_struct_stage (#16)'
        assert not args.count_readout != 'density', 'tile_split (#30) requires count_readout=density'
        print('TILE SPLIT enabled (#30): 2x2 local contrast + per-quadrant ROI-norm then locked cut', flush=True)
    if args.local_contrast:
        assert not args.filter_otsu, 'local_contrast incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'local_contrast incompatible with filter_prenorm (#23)'
        assert not args.roi_norm_per_exemplar, 'local_contrast incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'local_contrast incompatible with per_exemplar_filter (#25)'
        assert not args.box_peak_residual, 'local_contrast incompatible with box_peak_residual (#27)'
        assert not args.thresh_expand, 'local_contrast incompatible with thresh_expand (#28)'
        assert not args.bg_sub_integral, 'local_contrast incompatible with bg_sub_integral (#29)'
        assert not args.tile_split, 'local_contrast incompatible with tile_split (#30)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'local_contrast incompatible with gate+scalar filters (#11/#12)'
        assert not args.context_aware_sim, 'local_contrast incompatible with context_aware_sim (#17)'
        assert not args.count_readout != 'density', 'local_contrast (#31) requires count_readout=density'
        print('LOCAL CONTRAST enabled (#31): S / boxblur(S) before minmax and locked z', flush=True)
    if args.input_unsharp:
        assert not args.filter_otsu, 'input_unsharp incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'input_unsharp incompatible with filter_prenorm (#23)'
        assert not args.roi_norm_per_exemplar, 'input_unsharp incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'input_unsharp incompatible with per_exemplar_filter (#25)'
        assert not args.box_peak_residual, 'input_unsharp incompatible with box_peak_residual (#27)'
        assert not args.thresh_expand, 'input_unsharp incompatible with thresh_expand (#28)'
        assert not args.bg_sub_integral, 'input_unsharp incompatible with bg_sub_integral (#29)'
        assert not args.tile_split, 'input_unsharp incompatible with tile_split (#30)'
        assert not args.local_contrast, 'input_unsharp incompatible with local_contrast (#31)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'input_unsharp incompatible with gate+scalar filters (#11/#12)'
        assert not args.count_readout != 'density', 'input_unsharp (#32) requires count_readout=density'
        print('INPUT UNSHARP enabled (#32): UnsharpMask r=2 p=150 t=3 before backbone', flush=True)
    if args.boxwise_counts:
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'boxwise_counts (#34) requires use_roi_norm + roi_norm_after_mean'
        assert args.filter_background, 'boxwise_counts (#34) requires filter_background True'
        assert not args.filter_otsu, 'boxwise_counts incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'boxwise_counts incompatible with filter_prenorm (#23)'
        assert not args.roi_norm_per_exemplar, 'boxwise_counts incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'boxwise_counts incompatible with per_exemplar_filter (#25)'
        assert not args.mass_weight_exemplar or True, 'boxwise_counts replaces shared MWEx z'
        assert not args.box_peak_residual, 'boxwise_counts incompatible with box_peak_residual (#27)'
        assert not args.thresh_expand, 'boxwise_counts incompatible with thresh_expand (#28)'
        assert not args.bg_sub_integral, 'boxwise_counts incompatible with bg_sub_integral (#29)'
        assert not args.tile_split, 'boxwise_counts incompatible with tile_split (#30)'
        assert not args.local_contrast, 'boxwise_counts incompatible with local_contrast (#31)'
        assert not args.input_unsharp, 'boxwise_counts incompatible with input_unsharp (#32)'
        assert not args.cosine_similarity, 'boxwise_counts incompatible with cosine_similarity (#33)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'boxwise_counts incompatible with gate+scalar filters (#11/#12)'
        assert not args.count_readout != 'density', 'boxwise_counts (#34) requires count_readout=density'
        print('BOXWISE COUNTS enabled (#34): own minmax/z/cut per box, mean filtered maps', flush=True)
    if args.guided_density:
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'guided_density (#35) requires use_roi_norm + roi_norm_after_mean'
        assert args.filter_background, 'guided_density (#35) requires filter_background True'
        assert not args.filter_otsu, 'guided_density incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'guided_density incompatible with filter_prenorm (#23)'
        assert not args.roi_norm_per_exemplar, 'guided_density incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'guided_density incompatible with per_exemplar_filter (#25)'
        assert not args.box_peak_residual, 'guided_density incompatible with box_peak_residual (#27)'
        assert not args.thresh_expand, 'guided_density incompatible with thresh_expand (#28)'
        assert not args.bg_sub_integral, 'guided_density incompatible with bg_sub_integral (#29)'
        assert not args.tile_split, 'guided_density incompatible with tile_split (#30)'
        assert not args.local_contrast, 'guided_density incompatible with local_contrast (#31)'
        assert not args.input_unsharp, 'guided_density incompatible with input_unsharp (#32)'
        assert not args.cosine_similarity, 'guided_density incompatible with cosine_similarity (#33)'
        assert not args.boxwise_counts, 'guided_density incompatible with boxwise_counts (#34)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'guided_density incompatible with gate+scalar filters (#11/#12)'
        assert args.count_readout == 'density', 'guided_density (#35) requires count_readout=density'
        print('GUIDED DENSITY enabled (#35): self-guided r=2 eps=0.01 before hard cut', flush=True)
    if args.exemplar_avg:
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'exemplar_avg (#36) requires use_roi_norm + roi_norm_after_mean'
        assert args.filter_background, 'exemplar_avg (#36) requires filter_background True'
        assert not args.filter_otsu, 'exemplar_avg incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'exemplar_avg incompatible with filter_prenorm (#23)'
        assert not args.roi_norm_per_exemplar, 'exemplar_avg incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'exemplar_avg incompatible with per_exemplar_filter (#25)'
        assert not args.box_peak_residual, 'exemplar_avg incompatible with box_peak_residual (#27)'
        assert not args.thresh_expand, 'exemplar_avg incompatible with thresh_expand (#28)'
        assert not args.bg_sub_integral, 'exemplar_avg incompatible with bg_sub_integral (#29)'
        assert not args.tile_split, 'exemplar_avg incompatible with tile_split (#30)'
        assert not args.local_contrast, 'exemplar_avg incompatible with local_contrast (#31)'
        assert not args.input_unsharp, 'exemplar_avg incompatible with input_unsharp (#32)'
        assert not args.cosine_similarity, 'exemplar_avg incompatible with cosine_similarity (#33)'
        assert not args.boxwise_counts, 'exemplar_avg incompatible with boxwise_counts (#34)'
        assert not args.guided_density, 'exemplar_avg incompatible with guided_density (#35)'
        assert not args.multi_scale_ex, 'exemplar_avg incompatible with multi_scale_ex (#14)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'exemplar_avg incompatible with gate+scalar filters (#11/#12)'
        assert args.count_readout == 'density', 'exemplar_avg (#36) requires count_readout=density'
        print('EXEMPLAR AVG enabled (#36): one shared ROI-kernel mean before density assembly', flush=True)
    if args.normalize_features:
        assert not args.cosine_similarity, 'normalize_features (#37) is L2-only; cosine_similarity (#33) is closed'
        assert not args.exemplar_avg, 'normalize_features incompatible with exemplar_avg (#36)'
        assert not args.boxwise_counts, 'normalize_features incompatible with boxwise_counts (#34)'
        assert not args.guided_density, 'normalize_features incompatible with guided_density (#35)'
        assert args.count_readout == 'density', 'normalize_features (#37) requires count_readout=density'
        print('NORMALIZE FEATURES enabled (#37): channel-wise L2 on feats, multi-channel depthwise conv kept', flush=True)
    if args.normalize_only_biggest_bbox:
        assert args.normalize_features, 'normalize_only_biggest_bbox requires normalize_features (#37)'
        print('NORMALIZE ONLY BIGGEST BBOX enabled: L2 from largest annotation ROI only', flush=True)
    if args.outside_only_cut:
        assert args.filter_background, 'outside_only_cut (#39) requires filter_background True'
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'outside_only_cut (#39) requires use_roi_norm + roi_norm_after_mean'
        assert not args.filter_otsu, 'outside_only_cut incompatible with filter_otsu (#22)'
        assert not args.filter_prenorm, 'outside_only_cut incompatible with filter_prenorm (#23)'
        assert not args.roi_norm_per_exemplar, 'outside_only_cut incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'outside_only_cut incompatible with per_exemplar_filter (#25)'
        assert not args.box_peak_residual, 'outside_only_cut incompatible with box_peak_residual (#27)'
        assert not args.thresh_expand, 'outside_only_cut incompatible with thresh_expand (#28)'
        assert not args.bg_sub_integral, 'outside_only_cut incompatible with bg_sub_integral (#29)'
        assert not args.tile_split, 'outside_only_cut incompatible with tile_split (#30)'
        assert not args.local_contrast, 'outside_only_cut incompatible with local_contrast (#31)'
        assert not args.boxwise_counts, 'outside_only_cut incompatible with boxwise_counts (#34)'
        assert not args.guided_density, 'outside_only_cut incompatible with guided_density (#35)'
        assert not args.exemplar_avg, 'outside_only_cut incompatible with exemplar_avg (#36)'
        assert not args.normalize_features, 'outside_only_cut incompatible with normalize_features (#37)'
        assert not args.remove_bbox_intersection, 'outside_only_cut incompatible with remove_bbox_intersection (#38)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'outside_only_cut incompatible with gate+scalar filters (#11/#12)'
        assert args.count_readout == 'density', 'outside_only_cut (#39) requires count_readout=density'
        print('OUTSIDE ONLY CUT enabled (#39): hard cut only outside exemplar box union', flush=True)
    if args.roi_norm_median:
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'roi_norm_median (#40) requires use_roi_norm + roi_norm_after_mean'
        assert args.filter_background, 'roi_norm_median (#40) requires filter_background True'
        assert not args.roi_norm_per_exemplar, 'roi_norm_median incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'roi_norm_median incompatible with per_exemplar_filter (#25)'
        assert not args.boxwise_counts, 'roi_norm_median incompatible with boxwise_counts (#34)'
        assert not args.guided_density, 'roi_norm_median incompatible with guided_density (#35)'
        assert not args.exemplar_avg, 'roi_norm_median incompatible with exemplar_avg (#36)'
        assert not args.normalize_features, 'roi_norm_median incompatible with normalize_features (#37)'
        assert not args.remove_bbox_intersection, 'roi_norm_median incompatible with remove_bbox_intersection (#38)'
        assert not args.outside_only_cut, 'roi_norm_median incompatible with outside_only_cut (#39)'
        assert not args.filter_otsu and not args.filter_prenorm, \
            'roi_norm_median incompatible with filter_otsu/prenorm (#22/#23)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'roi_norm_median incompatible with gate+scalar filters (#11/#12)'
        assert args.count_readout == 'density', 'roi_norm_median (#40) requires count_readout=density'
        print('ROI NORM MEDIAN enabled (#40): z = median of per-box ellipse pools', flush=True)
    if args.filter_area_sum:
        assert args.filter_background, 'filter_area_sum (#41) requires filter_background True'
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'filter_area_sum (#41) requires use_roi_norm + roi_norm_after_mean'
        assert not args.filter_otsu and not args.filter_prenorm, \
            'filter_area_sum incompatible with filter_otsu/prenorm (#22/#23)'
        assert not args.roi_norm_per_exemplar, 'filter_area_sum incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'filter_area_sum incompatible with per_exemplar_filter (#25)'
        assert not args.box_peak_residual, 'filter_area_sum incompatible with box_peak_residual (#27)'
        assert not args.thresh_expand, 'filter_area_sum incompatible with thresh_expand (#28)'
        assert not args.bg_sub_integral, 'filter_area_sum incompatible with bg_sub_integral (#29)'
        assert not args.tile_split, 'filter_area_sum incompatible with tile_split (#30)'
        assert not args.boxwise_counts, 'filter_area_sum incompatible with boxwise_counts (#34)'
        assert not args.guided_density, 'filter_area_sum incompatible with guided_density (#35)'
        assert not args.exemplar_avg, 'filter_area_sum incompatible with exemplar_avg (#36)'
        assert not args.normalize_features, 'filter_area_sum incompatible with normalize_features (#37)'
        assert not args.remove_bbox_intersection, 'filter_area_sum incompatible with remove_bbox_intersection (#38)'
        assert not args.outside_only_cut, 'filter_area_sum incompatible with outside_only_cut (#39)'
        assert not args.roi_norm_median, 'filter_area_sum incompatible with roi_norm_median (#40)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'filter_area_sum incompatible with gate+scalar filters (#11/#12)'
        assert args.count_readout == 'density', 'filter_area_sum (#41) requires count_readout=density'
        print('FILTER AREA SUM enabled (#41): thresh denominator = sum of pooled box areas', flush=True)
    if args.skip_bbox_rescale:
        args.correct_bbox_resize = False
        assert not args.exemplar_avg, 'skip_bbox_rescale incompatible with exemplar_avg (#36)'
        assert not args.remove_bbox_intersection, 'skip_bbox_rescale incompatible with remove_bbox_intersection (#38)'
        assert not args.filter_area_sum, 'skip_bbox_rescale incompatible with filter_area_sum (#41)'
        assert args.count_readout == 'density', 'skip_bbox_rescale (#42) requires count_readout=density'
        print('SKIP BBOX RESCALE enabled (#42): correct_bbox_resize forced False', flush=True)
    if args.median_cut_path:
        args.use_threshold = True
        assert not args.exemplar_avg, 'median_cut_path incompatible with exemplar_avg (#36)'
        assert not args.remove_bbox_intersection, 'median_cut_path incompatible with remove_bbox_intersection (#38)'
        assert not args.skip_bbox_rescale, 'median_cut_path incompatible with skip_bbox_rescale (#42)'
        assert not args.filter_area_sum, 'median_cut_path incompatible with filter_area_sum (#41)'
        assert not args.roi_norm_median, 'median_cut_path incompatible with roi_norm_median (#40)'
        assert not args.outside_only_cut, 'median_cut_path incompatible with outside_only_cut (#39)'
        assert args.count_readout == 'density', 'median_cut_path (#43) requires count_readout=density'
        print('MEDIAN CUT PATH enabled (#43): use_threshold forces median half-plane', flush=True)
    if args.exemplar_global_weight:
        assert args.mass_weight_exemplar, 'exemplar_global_weight (#44) only changes MWEx weight source'
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'exemplar_global_weight (#44) requires use_roi_norm + roi_norm_after_mean'
        assert not args.roi_norm_per_exemplar, 'exemplar_global_weight incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'exemplar_global_weight incompatible with per_exemplar_filter (#25)'
        assert not args.boxwise_counts, 'exemplar_global_weight incompatible with boxwise_counts (#34)'
        assert not args.exemplar_avg, 'exemplar_global_weight incompatible with exemplar_avg (#36)'
        assert not args.normalize_features, 'exemplar_global_weight incompatible with normalize_features (#37)'
        assert not args.median_cut_path, 'exemplar_global_weight incompatible with median_cut_path (#43)'
        assert not args.outside_only_cut, 'exemplar_global_weight incompatible with outside_only_cut (#39)'
        assert not args.filter_area_sum, 'exemplar_global_weight incompatible with filter_area_sum (#41)'
        assert not args.skip_bbox_rescale, 'exemplar_global_weight incompatible with skip_bbox_rescale (#42)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'exemplar_global_weight incompatible with gate+scalar filters (#11/#12)'
        assert args.exemplar_reduce == 'mean', 'exemplar_global_weight (#44) requires exemplar_reduce=mean'
        assert args.filter_background, 'exemplar_global_weight (#44) requires filter_background True'
        assert args.count_readout == 'density', 'exemplar_global_weight (#44) requires count_readout=density'
        print('EXEMPLAR GLOBAL WEIGHT enabled (#44): MWEx weights from full-grid positive mass', flush=True)
    if args.outbox_parity:
        assert args.use_roi_norm and args.roi_norm_after_mean, \
            'outbox_parity (#45) requires use_roi_norm + roi_norm_after_mean'
        assert not args.roi_norm_per_exemplar, 'outbox_parity incompatible with roi_norm_per_exemplar (#24)'
        assert not args.per_exemplar_filter, 'outbox_parity incompatible with per_exemplar_filter (#25)'
        assert not args.boxwise_counts, 'outbox_parity incompatible with boxwise_counts (#34)'
        assert not args.exemplar_avg, 'outbox_parity incompatible with exemplar_avg (#36)'
        assert not args.normalize_features, 'outbox_parity incompatible with normalize_features (#37)'
        assert not args.median_cut_path, 'outbox_parity incompatible with median_cut_path (#43)'
        assert not args.outside_only_cut, 'outbox_parity incompatible with outside_only_cut (#39)'
        assert not args.filter_area_sum, 'outbox_parity incompatible with filter_area_sum (#41)'
        assert not args.skip_bbox_rescale, 'outbox_parity incompatible with skip_bbox_rescale (#42)'
        assert not args.exemplar_global_weight, 'outbox_parity incompatible with exemplar_global_weight (#44)'
        assert not args.dense_norm_gate and not args.dense_fs_gate, \
            'outbox_parity incompatible with gate+scalar filters (#11/#12)'
        assert args.count_readout == 'density', 'outbox_parity (#45) requires count_readout=density'
        print('OUTBOX PARITY enabled (#45): lift outside-rectangle mass to inside parity', flush=True)
    if args.remove_bbox_intersection:
        assert args.count_readout == 'density', 'remove_bbox_intersection (#38) requires count_readout=density'
        assert not args.exemplar_avg, 'remove_bbox_intersection incompatible with exemplar_avg (#36)'
        assert not args.boxwise_counts, 'remove_bbox_intersection incompatible with boxwise_counts (#34)'
        assert not args.guided_density, 'remove_bbox_intersection incompatible with guided_density (#35)'
        assert not args.normalize_features, 'remove_bbox_intersection incompatible with normalize_features (#37)'
        print('REMOVE BBOX INTERSECTION enabled (#38): ceil-inward integer boxes before ROI', flush=True)
    return args

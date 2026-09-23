"""Training-free FSC147 counting entry point.

Thin CLI: parse → validate → load backbone → evaluate. Heavy logic lives in
tf_pipeline.{args,backbones,features,pipeline,postprocess,metrics,tags}.
"""
import os
import json
import copy as _copy

from tqdm import tqdm
import numpy as np
import torchvision.transforms as T

from src.utils import log_results, exist_match_df, get_counting_metrics
from src.helpers import load_json

from .args import build_parser, validate_args
from .backbones import load_backbone
from . import state as _state
from .state import device, set_xd
from .pipeline import process_example
from .tags import build_wave1_tag
from .metrics import extended_metrics, density_bins, write_worst15


def _load_examples(args):
    examples = load_json(args.annotation)
    splits = load_json(args.splits)
    examples = {k: v for k, v in examples.items() if k in splits[args.split]}
    if args.subset_file:
        keep = set(load_json(args.subset_file))
        examples = {k: v for k, v in examples.items() if k in keep}
        print(f"subset_file={args.subset_file} n={len(examples)}", flush=True)
    elif args.max_images is not None:
        examples = dict(list(examples.items())[:args.max_images])
    return examples


def _maybe_load_xd(args):
    if not args.dual_density:
        return
    from src.dinov3_adapter import Dinov3VitBackbone
    per_crop = max(1, int(args.input_size) // 32)
    sz2 = per_crop * 16
    model = Dinov3VitBackbone(
        hf_name='facebook/dinov3-vits16-pretrain-lvd1689m', img_size=sz2
    ).to(device).eval()
    transform = T.Compose([
        T.Resize((sz2, sz2), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    set_xd(model, transform, ['vit_out'])
    print(
        f"XDEN aux=dinov3-vits16 img_size={sz2} params_M="
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}",
        flush=True,
    )


def _hr_setup(args):
    if not (args.dense_hr_gate and args.dense_hr_gate > 0):
        return None, None
    assert args.model_name.startswith('dinov3_convnext'), \
        'dense_hr_gate currently implemented for dinov3_convnext only'
    assert not args.dual_density, 'dense_hr_gate incompatible with dual_density (#20)'
    hr_dim = int(args.dense_hr_input)
    hr_transform = T.Compose([
        T.Resize((hr_dim, hr_dim), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    hr_config = _copy.copy(args)
    hr_config.filter_thresh_scale = float(args.dense_hr_fs)
    hr_config.dense_fs_gate = 0.0
    hr_config.dense_norm_gate = 0.0
    hr_config.dense_fs = None
    hr_config.density_warp = False
    hr_config.input_size = hr_dim
    return hr_transform, hr_config


def _predict_one(idx, img_filename, entry, model, transform, map_keys, args,
                 density_map_dir, gt_count, hr_transform=None, hr_config=None):
    gt, pred = process_example(
        idx, img_filename, entry, model, transform, map_keys,
        args.img_dir, density_map_dir, args, gt_count=gt_count,
    )
    if args.dual_density and _state.XD_MODEL is not None:
        _, pred2 = process_example(
            idx, img_filename, entry, _state.XD_MODEL, _state.XD_TRANSFORM, _state.XD_MAP_KEYS,
            args.img_dir, density_map_dir, args, gt_count=gt_count,
        )
        print(
            f"XDEN {img_filename} pred1={float(pred):.4f} pred2={float(pred2):.4f} "
            f"mean={0.5 * (float(pred) + float(pred2)):.4f}",
            flush=True,
        )
        pred = 0.5 * (float(pred) + float(pred2))
    if args.tta_flip != 0:
        _, pred_flip = process_example(
            idx, img_filename, entry, model, transform, map_keys,
            args.img_dir, density_map_dir, args, gt_count=gt_count, flip_view=True,
        )
        pred = 0.5 * (pred + pred_flip)
    scale_views = (
        [float(x) for x in args.scale_views.split(',') if x.strip()]
        if args.scale_views else []
    )
    if scale_views:
        preds_s = [pred]
        for sv in scale_views:
            _, p_s = process_example(
                idx, img_filename, entry, model, transform, map_keys,
                args.img_dir, density_map_dir, args, gt_count=gt_count, scale_view=sv,
            )
            preds_s.append(p_s)
        pred = sum(preds_s) / len(preds_s)
    if args.dense_hr_gate and args.dense_hr_gate > 0 and pred >= args.dense_hr_gate:
        old_sz = getattr(model, 'img_size', None)
        if old_sz is not None:
            model.img_size = int(args.dense_hr_input)
        try:
            _, pred_hr = process_example(
                idx, img_filename, entry, model, hr_transform, map_keys,
                args.img_dir, density_map_dir, hr_config, gt_count=gt_count,
            )
        finally:
            if old_sz is not None:
                model.img_size = old_sz
        print(
            f"HRCALL {img_filename} pred1={pred:.4f} predHR={pred_hr:.4f} "
            f"gate={args.dense_hr_gate} hr_in={args.dense_hr_input} hr_fs={args.dense_hr_fs}",
            flush=True,
        )
        pred = float(pred_hr)
    return gt, pred


def _running_log(predictions, targets, n_total):
    p = np.array(predictions, dtype=np.float64)
    t = np.array(targets, dtype=np.float64)
    e = p - t
    ae = np.abs(e)
    print(
        f"RUNNING n={len(predictions)}/{n_total} MAE={float(ae.mean()):.4f} "
        f"RMSE={float(np.sqrt(np.mean(e ** 2))):.4f} BIAS={float(e.mean()):.4f} "
        f"MED={float(np.median(ae)):.4f} P90={float(np.percentile(ae, 90)):.4f} "
        f"P95={float(np.percentile(ae, 95)):.4f} "
        f"OVER={int((e > 0).sum())} UNDER={int((e < 0).sum())} EXACT={int((e == 0).sum())}",
        flush=True,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    row_params_dict = {
        k: v for k, v in vars(args).items()
        if k not in ['img_dir', 'density_map_dir', 'annotation', 'splits', 'CARPK']
    }
    results_csv_path = args.log_file
    if os.path.exists(results_csv_path) and exist_match_df(results_csv_path, row_params_dict) and not args.no_skip:
        print("Already done for this configuration. Skipping...")
        return
    print("Parameters Recap:")
    print(json.dumps(vars(args), indent=4))

    validate_args(args)
    model, transform, resize_dim = load_backbone(args)
    examples = _load_examples(args)
    map_keys = ['vit_out'] if 'vit' in args.model_name or 'convnext' in args.model_name else ['map3']
    _maybe_load_xd(args)
    hr_transform, hr_config = _hr_setup(args)

    wave1, per_image_path = build_wave1_tag(args, resize_dim)
    args_dict = {
        k: v for k, v in vars(args).items()
        if k not in [
            'model_name', 'img_dir', 'density_map_dir', 'annotation', 'splits',
            'save_preds_to_file', 'log_results', 'CARPK', 'no_skip', 'log_file',
        ]
    }
    os.makedirs('results', exist_ok=True)
    per_image_fh = open(per_image_path, 'w', buffering=1)
    per_image_fh.write('idx,filename,gt,pred,err,abs_err,rel_err,ape,sq_err\n')
    print(f'per_image_log={per_image_path}', flush=True)

    density_map_dir = None if args.CARPK else args.density_map_dir
    if args.CARPK:
        print("Setting density_map_dir to None for CARPK dataset")

    predictions, targets, ids_out = [], [], []
    scale_views = (
        [float(x) for x in args.scale_views.split(',') if x.strip()]
        if args.scale_views else []
    )
    for idx, (img_filename, entry) in tqdm(
        enumerate(examples.items()), total=len(examples), dynamic_ncols=True
    ):
        gt_count = len(entry['points']) if density_map_dir is None else None
        gt, pred = _predict_one(
            idx, img_filename, entry, model, transform, map_keys, args,
            density_map_dir, gt_count, hr_transform=hr_transform, hr_config=hr_config,
        )
        predictions.append(pred)
        targets.append(gt)
        ids_out.append(img_filename)
        err = float(pred) - float(gt)
        abs_err = abs(err)
        rel_err = err / (float(gt) + 1e-8)
        ape = abs_err / (float(gt) + 1e-8)
        per_image_fh.write(
            f"{idx},{img_filename},{float(gt):.6f},{float(pred):.6f},"
            f"{err:.6f},{abs_err:.6f},{rel_err:.6f},{ape:.6f},{err * err:.6f}\n"
        )
        per_image_fh.flush()
        if len(predictions) % 25 == 0 or len(predictions) == len(examples):
            _running_log(predictions, targets, len(examples))

    per_image_fh.close()
    if args.save_preds_to_file:
        print("Saving predictions to file...")
        save_dir = os.path.join('results', args.model_name)
        os.makedirs(save_dir, exist_ok=True)
        out_file_name = (
            f"predictions_{args.model_name}_"
            f"{'_'.join([f'{k}_{v}' for k, v in args_dict.items()])}"
        )[:150] + ".npy"
        np.save(os.path.join(save_dir, out_file_name), predictions)
        np.save(os.path.join(save_dir, 'targets.npy'), targets)
        np.save(os.path.join(save_dir, 'ids.npy'), np.array(ids_out))
        print(f"Predictions saved to {os.path.join(save_dir, out_file_name)}")

    metrics = get_counting_metrics(predictions, targets)
    p, t, e, ae, ape, ext = extended_metrics(predictions, targets)
    bin_stats = density_bins(t, e, ae, ape, metrics=metrics)
    metrics.update({k: v for k, v in ext.items() if k not in ('n',)})
    metrics['n_images'] = ext['n']
    print('EXTENDED_METRICS', ext, flush=True)
    for row in bin_stats:
        print('DENSITY_BIN', row, flush=True)
    write_worst15(per_image_path, ids_out, t, p, e, ae, ape)
    print(f'per_image_path={per_image_path}', flush=True)
    print(metrics)

    if args.log_results:
        try:
            log_results({**args_dict, **metrics}, args.model_name, path=results_csv_path)
        except Exception as exc:
            print(f"[warn] log_results failed (non-fatal): {exc}")

#!/usr/bin/env python3
"""scripts/eval_res_sweep.py — N0026 eval-only input-resolution sweep on the frozen N0021 champion.

Runs full FSC147 val inference with /data/runs/N0021_dino_partialft/best.pth at each
input_size in {224,308,392,448,518}. All five are multiples of patch 14 (ps=S//14 exact:
16/22/28/32/37 cells per side) and the champion backbone is created with dynamic_img_size=True
(N0021 model.py:38), so only the data loader grid changes. Champion architecture imported from
tree/nodes/N0021_dino_partialft (single source of truth); loader/AMP patterns identical to
scripts/eval_readout_lab.py (which reproduced champion val MAE 20.441 @392).

Per resolution records per-image {img_id, N_gt, N_hat} + overall MAE/RMSE, and stratifies
errors two ways with edges FIXED ONCE and shared across resolutions:
  - GT-count terciles (edges = 33.33/66.67 percentiles of the FULL val GT distribution,
    computed from GT density-map sums — exactly the counts the loader emits);
  - buckets [0,25)/[25,75)/[75,200)/[200,500)/[500,inf) with SSE share per bucket
    (pins where RMSE lives; KEY question: does 448/518 improve the [500,inf) bucket,
    which carries ~76% of total SSE at 392 from only 17 images).
Prints the full 5-config table + pre-registered H0035 verdict:
  PASS  if some non-392 res improves RMSE >=3 with MAE regression <=0.5 vs the 392 baseline;
  FAIL  if every non-392 res degrades MAE >2.0;
  else INCONCLUSIVE.
Writes JSON (--out, default tree/nodes/N0026_res_sweep/res_results.json). --smoke runs the
extreme grids (224 and 518) on ~20 imgs each and writes <out>.smoke.json instead, so smoke
never clobbers real results.
"""
import argparse
import importlib.util
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE21 = os.path.join(REPO, "tree", "nodes", "N0021_dino_partialft")
DEFAULT_CKPT = "/data/runs/N0021_dino_partialft/best.pth"
DEFAULT_DATA_ROOT = "/data/dataset/FSC147"
ALL_SIZES = (224, 308, 392, 448, 518)
BUCKETS = ((0.0, 25.0), (25.0, 75.0), (75.0, 200.0), (200.0, 500.0), (500.0, math.inf))
TERCILE_PCTS = (100.0 / 3.0, 200.0 / 3.0)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def gt_val_counts(root):
    import numpy as np

    with open(os.path.join(root, "Train_Test_Val_FSC_147.json")) as f:
        ids = json.load(f)["val"]
    out = {}
    for im_id in ids:
        stem = im_id[:-4] if im_id.endswith(".jpg") else im_id
        d = np.load(os.path.join(root, "gt_density_map_adaptive_384_VarV2", f"{stem}.npy"))
        out[im_id] = float(d.sum())
    return out


def mae_of(rows):
    return sum(abs(r["N_hat"] - r["N_gt"]) for r in rows) / max(len(rows), 1)


def rmse_of(rows):
    return math.sqrt(sum((r["N_hat"] - r["N_gt"]) ** 2 for r in rows) / max(len(rows), 1))


def sse_of(rows):
    return sum((r["N_hat"] - r["N_gt"]) ** 2 for r in rows)


def stratum_stats(rows, cond, total_sse):
    sub = [r for r in rows if cond(r["N_gt"])]
    return {"n": len(sub), "mae": round(mae_of(sub), 4), "rmse": round(rmse_of(sub), 4),
            "sse_share": round(sse_of(sub) / max(total_sse, 1e-9), 6)}


def stratify(rows, ter_edges, total_sse):
    lo_e, hi_e = ter_edges
    terciles = {
        f"ter1[<{lo_e:.1f}]": stratum_stats(rows, lambda g: g < lo_e, total_sse),
        f"ter2[{lo_e:.1f},{hi_e:.1f})": stratum_stats(rows, lambda g: lo_e <= g < hi_e, total_sse),
        f"ter3[>={hi_e:.1f}]": stratum_stats(rows, lambda g: g >= hi_e, total_sse),
    }
    buckets = {f"[{lo:g},{hi:g})": stratum_stats(rows, lambda g, lo=lo, hi=hi: lo <= g < hi, total_sse)
               for lo, hi in BUCKETS}
    return terciles, buckets


def print_strat(title, stats_dict, show_share=True):
    share_col = f"{'sse%':>8}" if show_share else ""
    print(f"\n-- {title} --")
    print(f"{'stratum':<20} {'n':>5} {'MAE':>10} {'RMSE':>10}{share_col}")
    for name, st in stats_dict.items():
        line = f"{name:<20} {st['n']:>5} {st['mae']:>10.3f} {st['rmse']:>10.3f}"
        if show_share:
            line += f"{100.0 * st['sse_share']:>8.2f}"
        print(line)


def run_pass(args, model, device, use_amp, size):
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, os.path.join(REPO, "code"))
    from data.fsc147 import FSC147Density, collate_density

    root = args.data_root or DEFAULT_DATA_ROOT
    va = FSC147Density(root, size, "val")
    bs = args.bs or (8 if size <= 392 else 4)
    loader = DataLoader(va, batch_size=bs, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_density, pin_memory=True)
    n_total = len(va) if args.limit is None else min(args.limit, len(va))
    print(f"[sweep] S={size} bs={bs} ps={size // 14} val={len(va)} running={n_total} "
          f"amp={use_amp}", flush=True)
    rows = []
    idx = 0
    abs_err = 0.0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if idx >= n_total:
                break
            imgs = batch["imgs"].to(device)
            bboxes = batch["bboxes"].to(device)
            counts = batch["counts"].float()
            with torch.autocast("cuda", enabled=use_amp):
                out = model(imgs, bboxes)
            dens = out["density"].float().cpu()
            for j in range(dens.shape[0]):
                if idx >= n_total:
                    break
                n_hat = float(dens[j, 0].sum())
                n_gt = float(counts[j])
                rows.append({"img_id": va.ids[idx], "N_gt": round(n_gt, 4),
                             "N_hat": round(n_hat, 4)})
                abs_err += abs(n_hat - n_gt)
                idx += 1
            if (bi + 1) % 10 == 0:
                print(f"[sweep S={size}] batch {bi + 1}/{math.ceil(n_total / bs)} "
                      f"rows={idx} runMAE={abs_err / idx:.3f}", flush=True)
    return rows


def validate_rows(size, rows, expected):
    assert len(rows) == expected, f"S={size}: rows {len(rows)} != expected {expected}"
    bad = [r for r in rows if not (math.isfinite(r["N_gt"]) and math.isfinite(r["N_hat"]))]
    assert not bad, f"S={size}: non-finite values in {len(bad)} rows"
    keys = {"img_id", "N_gt", "N_hat"}
    assert all(keys <= set(r) for r in rows), f"S={size}: schema violation"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--sizes", type=int, nargs="+", default=list(ALL_SIZES))
    p.add_argument("--smoke", action="store_true",
                   help="extreme grids 224+518, ~20 imgs each, JSON -> <out>.smoke.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=os.path.join(REPO, "tree", "nodes", "N0026_res_sweep",
                                                 "res_results.json"))
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--data-root", default=None)
    p.add_argument("--bs", type=int, default=None, help="override per-size default (8<=392, 4 above)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    args = p.parse_args()
    if args.smoke:
        args.sizes = [224, 518]
        if args.limit is None:
            args.limit = 20
        args.out = args.out + ".smoke.json"

    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_mod = load_module(os.path.join(NODE21, "config.py"), "n0026_champ_cfg")
    model_mod = load_module(os.path.join(NODE21, "model.py"), "n0026_champ_model")
    cfg = dict(getattr(cfg_mod, "cfg"))
    model = model_mod.build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    meta = {k: ckpt[k] for k in ("epoch", "best_mae") if isinstance(ckpt, dict) and k in ckpt}
    print(f"[sweep] loaded {args.ckpt} meta={meta} device={device}", flush=True)

    root = args.data_root or cfg.get("data_root", DEFAULT_DATA_ROOT)
    counts = gt_val_counts(root)
    vals = np.array(sorted(counts.values()))
    ter_edges = (float(np.percentile(vals, TERCILE_PCTS[0])),
                 float(np.percentile(vals, TERCILE_PCTS[1])))
    print(f"[sweep] full-val GT distribution: n={len(vals)} min={vals.min():.1f} "
          f"median={np.median(vals):.1f} max={vals.max():.1f} "
          f"| tercile edges FIXED ONCE: [{ter_edges[0]:.3f}, {ter_edges[1]:.3f}]", flush=True)

    results = {}
    for size in args.sizes:
        rows = run_pass(args, model, device, bool(args.amp) and device.type == "cuda", size)
        validate_rows(size, rows, len(rows))
        total_sse = sse_of(rows)
        terciles, buckets = stratify(rows, ter_edges, total_sse)
        results[str(size)] = {"n": len(rows), "mae": round(mae_of(rows), 4),
                              "rmse": round(rmse_of(rows), 4), "sse": round(total_sse, 2),
                              "terciles": terciles, "buckets": buckets, "per_image": rows}
        print(f"\n[sweep S={size}] DONE n={len(rows)} MAE={results[str(size)]['mae']:.4f} "
              f"RMSE={results[str(size)]['rmse']:.4f}", flush=True)
        print_strat(f"S={size} terciles (edges fixed once: {ter_edges[0]:.2f}/{ter_edges[1]:.2f})",
                    terciles)
        print_strat(f"S={size} GT-count buckets + SSE share", buckets)

    base = results.get("392")
    others = [s for s in results if s != "392"]
    if base is not None and others:
        print("\n" + "=" * 72)
        print(f"RES SWEEP TABLE  (baseline S=392: MAE={base['mae']:.3f} RMSE={base['rmse']:.3f}; "
              f"full={args.limit is None})")
        print(f"{'res':>5} {'n':>5} {'MAE':>9} {'RMSE':>9} {'dMAE':>8} {'dRMSE':>8}")
        for s in sorted(results, key=int):
            dma = results[s]["mae"] - base["mae"] if s != "392" else 0.0
            drm = results[s]["rmse"] - base["rmse"] if s != "392" else 0.0
            print(f"{s:>5} {results[s]['n']:>5} {results[s]['mae']:>9.3f} {results[s]['rmse']:>9.3f}"
                  f"{dma:>+8.3f}{drm:>+8.3f}")
        tail = "[500,inf)"
        print(f"\n-- TAIL bucket {tail} (KEY: carried ~76% of SSE @392 via 17 imgs) --")
        print(f"{'res':>5} {'n':>5} {'MAE':>9} {'RMSE':>9} {'sse%':>8}")
        for s in sorted(results, key=int):
            st = results[s]["buckets"][tail]
            print(f"{s:>5} {st['n']:>5} {st['mae']:>9.3f} {st['rmse']:>9.3f}"
                  f"{100.0 * st['sse_share']:>8.2f}")

        deltas = {s: (results[s]["rmse"] - base["rmse"], results[s]["mae"] - base["mae"])
                  for s in others}
        pass_hits = [s for s, (drm, dma) in deltas.items() if drm <= -3.0 and dma <= 0.5]
        all_fail = all(dma > 2.0 for _, dma in deltas.values())
        if pass_hits:
            verdict = "PASS"
            why = f"{pass_hits} improves RMSE>=3 with dMAE<=0.5 vs 392"
        elif all_fail:
            verdict = "FAIL"
            why = f"every non-392 res degrades MAE >2.0 ({deltas})"
        else:
            verdict = "INCONCLUSIVE"
            why = f"neither pass nor fail bar met ({deltas})"
        print(f"\nVERDICT H0035: {verdict} — {why}")
        print("=" * 72)

    payload = {
        "meta": {"ckpt": args.ckpt, "sizes": args.sizes, "amp": bool(args.amp),
                 "limit": args.limit, "num_val_images_full": len(counts),
                 "tercile_edges": list(ter_edges),
                 "buckets": [[lo, hi if math.isfinite(hi) else None] for lo, hi in BUCKETS],
                 "h0035_pass_rule": "exists non-392 res: dRMSE<=-3 AND dMAE<=+0.5 vs 392",
                 "h0035_fail_rule": "every non-392 res: dMAE>+2.0"},
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    if args.smoke:
        for s, entry in results.items():
            validate_rows(int(s), entry["per_image"], entry["n"])
        tot = sum(e["n"] for e in results.values())
        nonfin = sum(1 for e in results.values() for r in e["per_image"]
                     if not (math.isfinite(r["N_gt"]) and math.isfinite(r["N_hat"])))
        print(f"SMOKE_OK sizes={args.sizes} rows={tot} nonfinite={nonfin} out={args.out}")
    else:
        print(f"[sweep] results written to {args.out}")


if __name__ == "__main__":
    main()

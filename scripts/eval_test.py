#!/usr/bin/env python3
"""eval_test.py — score a trained node's best.pth on the FSC147 val/test split.

    python scripts/eval_test.py <node> [--split val|test] [--dense-threshold 500]

<node> is a tree id (N0002_h0001) or a path to the node/repro dir. best.pth is
resolved as <node>/run/latest/best.pth then <node>/best.pth (repro_run layout).
Evaluation is inference-only, augment=false, EMA-inference weights (what
BestCheckpoint recorded — canonical protocol). For `use_counttau` checkpoints it
also extracts the mechanism diagnostics (w_g, w_t, pinned temp) and the per-image
count surrogate g, so a H0010 verdict is one command:
    <split>_result.json  — mae/rmse/bias/under-share/dense-tail + counttau state
    <split>_perimage.json — per-image preds/gts/ids (+g surrogate when counttau)

Run on cac-server with /data/miniconda/envs/cac/bin/python.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cac.hub as hub  # noqa: E402

hub.setup_hf_env()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cac.config import load_config  # noqa: E402
from cac.data.fsc147 import FSC147Density, collate_density  # noqa: E402
from cac.data.pl_datamodule import worker_init_fn  # noqa: E402
from cac.engine.runner import _model_from_cfg, checksum  # noqa: E402
from cac.expt.node import TrajectoryTree  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_node(arg: str) -> str:
    if os.path.isdir(arg):
        return arg
    mat = TrajectoryTree(ROOT).materialize()
    if arg in mat:
        return mat[arg]["path"]
    sys.exit(f"node {arg} not found in tree/")


def _find_best(nd: str) -> str:
    cand = [os.path.join(nd, "run", "latest", "best.pth"), os.path.join(nd, "best.pth")]
    for c in cand:
        if os.path.exists(c):
            return c
    sys.exit(f"{nd}: no best.pth (run/latest or repro top-level)")


def _counttau_state(state, pairs) -> dict:
    out = {}
    for key in state:
        low = key.lower()
        for suffix, short in pairs:
            if low.endswith(suffix):
                v = state[key].detach().cpu().reshape(-1).tolist()
                out.setdefault(short, v)
                break
    return out


def main() -> int:
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: python scripts/eval_test.py <node> [--split val|test] "
                 "[--dense-threshold 500] [--batch 16] [--workers 4]")
    node = args[0]
    split = "test"
    dense_thr = 500.0
    BATCH, WORKERS = 0, None
    i = 1
    while i < len(args):
        if args[i] == "--split" and i + 1 < len(args):
            split = args[i + 1]; i += 2
        elif args[i] == "--dense-threshold" and i + 1 < len(args):
            dense_thr = float(args[i + 1]); i += 2
        elif args[i] == "--batch" and i + 1 < len(args):
            BATCH = int(args[i + 1]); i += 2
        elif args[i] == "--workers" and i + 1 < len(args):
            WORKERS = int(args[i + 1]); i += 2
        else:
            sys.exit(f"unknown argument: {args[i]}")

    nd = _resolve_node(node)
    node_id = os.path.basename(nd)
    cfgp = os.path.join(nd, "config.toml")
    if not os.path.exists(cfgp):
        sys.exit(f"{node}: missing config.toml")
    cfg = load_config(cfgp)
    best_path = _find_best(nd)
    out_json = os.path.join(nd, f"{split}_result.json")
    per_json = os.path.join(nd, f"{split}_perimage.json")

    seed = int(cfg.get("seed", 20260830))
    torch.manual_seed(seed)
    model = _model_from_cfg(cfg, nd)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(best_path, map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    data_root = str(cfg.get("data_root", "/data/dataset/FSC147"))
    S = int(cfg.get("input_size", 384))
    ds = FSC147Density(data_root, S, split, augment=False)
    dl = DataLoader(ds, batch_size=BATCH or int(cfg.get("batch_size", 16)),
                    num_workers=WORKERS if WORKERS is not None else int(cfg.get("num_workers", 4)),
                    collate_fn=collate_density, shuffle=False,
                    generator=torch.Generator().manual_seed(seed),
                    worker_init_fn=worker_init_fn(seed))

    use_cf = bool(cfg.get("use_counttau", False))
    Hf = S // 4
    preds, gts, ids, gs = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            imgs = batch["imgs"].to(dev)
            bbox = batch["bboxes"].to(dev)
            b3 = batch.get("bboxes3")
            if b3 is not None:
                b3 = b3.to(dev)
                try:
                    out = model(imgs, bbox, b3)
                except TypeError:
                    out = model(imgs, bbox)
            else:
                out = model(imgs, bbox)
            dens = out["density"] if isinstance(out, dict) else out
            pred = dens.flatten(1).sum(1).cpu()
            preds.extend(float(p) for p in pred)
            gts.extend(float(c) for c in batch["counts"])
            ids.extend(str(x) for x in batch["ids"])
            if use_cf:
                hs = model.backbone.forward_feature_map(imgs)
                fine = model.head.fuser(hs[0], hs[1])
                g = torch.log(1.0 + fine.detach().mean(dim=1, keepdim=True).clamp_min(0)
                              .sum(dim=(2, 3)) / float(Hf * Hf)).view(-1)
                gs.extend(float(v) for v in g.cpu())

    n = len(gts)
    abs_d = [abs(p - g) for p, g in zip(preds, gts)]
    mae = sum(abs_d) / n
    rmse = (sum((p - g) ** 2 for p, g in zip(preds, gts)) / n) ** 0.5
    mean_gt = sum(gts) / n
    mean_pred = sum(preds) / n
    under = [(p, g) for p, g in zip(preds, gts) if p < g]
    over = [(p, g) for p, g in zip(preds, gts) if p >= g]
    tail = [d for d, (p, g) in zip(abs_d, zip(preds, gts)) if g > dense_thr]
    tail_n = len(tail)
    tail_mae = sum(tail) / tail_n if tail_n else None

    res = {"node": node_id, "split": split, "seed": seed, "data_root": data_root,
           "input_size": S, "n": n, "mae": mae, "rmse": rmse,
           "mean_gt": mean_gt, "mean_pred": mean_pred, "bias_mean_delta": mean_pred - mean_gt,
           "n_under": len(under), "n_over": len(over),
           "sum_under": sum(g - p for p, g in under), "sum_over": sum(p - g for p, g in over),
           "dense_threshold": dense_thr, "n_gt_gt_thr": tail_n, "mae_gt_gt_thr": tail_mae,
           "best_path": best_path, "ckpt_epoch": ckpt.get("epoch"),
           "config_sha256": checksum(cfgp), "model_sha256": checksum(os.path.join(nd, "model.py"))}
    if use_cf:
        cs = _counttau_state(ckpt["model"], (("counttau.w_g", "w_g"), ("counttau.w_t", "w_t"), ("simport.temp", "temp")))
        for k, v in cs.items():
            res[f"ct_{k}"] = v
        if gs:
            g0 = min(gs); g1 = max(gs); gm = sum(gs) / len(gs)
            res["ct_g_min"], res["ct_g_max"], res["ct_g_mean"], res["ct_g_spread"] = g0, g1, gm, g1 - g0
    json.dump(res, open(out_json, "w"), indent=2)
    per = {"node": node_id, "split": split, "pred": preds, "gt": gts, "ids": ids,
           "mae": mae, "rmse": rmse, "ckpt_epoch": ckpt.get("epoch")}
    if gs:
        per["g"] = gs
    json.dump(per, open(per_json, "w"), indent=2)

    print(json.dumps(res, indent=2))
    print(f"[eval_test] wrote {out_json} + {per_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
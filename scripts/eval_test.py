#!/usr/bin/env python3
"""eval_test.py — score a trained node's best.pth on the FSC147 test split.

    python scripts/eval_test.py <node> [--split test]

<node> is a tree id (N0002_h0001) or a direct path. The node's config.toml +
model.py build the model; run/latest/best.pth must exist next to it.
Evaluation is inference-only, augment=false, EMA-inference weights (that is
what BestCheckpoint recorded — canonical protocol). Writes:
    <node>/test_result.json      — metrics + small tail summary
    <node>/test_perimage.json    — per-image preds/gt/ids for local analysis

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
from cac.engine.runner import _model_from_cfg  # noqa: E402
from cac.expt.node import TrajectoryTree  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_node(arg: str) -> str:
    if os.path.isdir(arg):
        return arg
    mat = TrajectoryTree(ROOT).materialize()
    if arg in mat:
        return mat[arg]["path"]
    sys.exit(f"node {arg} not found in tree/")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: python scripts/eval_test.py <node> [--split test|val]")
    node = args[0]
    split = "test"
    i = 1
    while i < len(args):
        if args[i] == "--split" and i + 1 < len(args):
            split = args[i + 1]
            i += 2
        else:
            i += 1

    nd = _resolve_node(node)
    node_id = os.path.basename(nd)
    cfgp = os.path.join(nd, "config.toml")
    if not os.path.exists(cfgp):
        sys.exit(f"{node}: missing config.toml")
    cfg = load_config(cfgp)

    best_path = os.path.join(nd, "run", "latest", "best.pth")
    if not os.path.exists(best_path):
        sys.exit(f"{node}: no run/latest/best.pth (node not trained yet)")

    out_json = os.path.join(nd, "test_result.json")
    per_json = os.path.join(nd, "test_perimage.json")

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
    dl = DataLoader(ds, batch_size=int(cfg.get("batch_size", 16)),
                    num_workers=int(cfg.get("num_workers", 4)),
                    collate_fn=collate_density, shuffle=False,
                    generator=torch.Generator().manual_seed(seed),
                    worker_init_fn=worker_init_fn(seed))

    preds: list[float] = []
    gts: list[float] = []
    ids: list[str] = []
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

    n = len(gts)
    abs_d = [abs(p - g) for p, g in zip(preds, gts)]
    mae = sum(abs_d) / n
    rmse = (sum((p - g) ** 2 for p, g in zip(preds, gts)) / n) ** 0.5
    mean_gt = sum(gts) / n
    mean_pred = sum(preds) / n
    tail = [(p, g) for p, g in zip(preds, gts) if g > 500]
    tail_mae = sum(abs(p - g) for p, g in tail) / len(tail) if tail else None

    res = {"node": node_id, "split": split, "seed": seed, "data_root": data_root,
           "n": n, "mae": mae, "rmse": rmse, "mean_gt": mean_gt,
           "mean_pred": mean_pred, "bias_mean_delta": mean_pred - mean_gt,
           "n_gt_gt_500": len(tail), "mae_gt_gt_500": tail_mae,
           "ckpt_epoch": ckpt.get("epoch"), "config_sha256": None,
           "model_sha256": None, "best_path": best_path}
    for key, path in (("config_sha256", cfgp), ("model_sha256", os.path.join(nd, "model.py"))):
        if os.path.exists(path):
            from cac.engine.runner import checksum
            res[key] = checksum(path)
    json.dump(res, open(out_json, "w"), indent=2)
    json.dump({"node": node_id, "split": split, "pred": preds, "gt": gts, "ids": ids,
               "mae": mae, "rmse": rmse, "ckpt_epoch": ckpt.get("epoch")},
              open(per_json, "w"), indent=2)

    print(json.dumps(res, indent=2))
    print(f"[eval_test] wrote {out_json} + {per_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
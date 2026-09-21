#!/usr/bin/env python3
"""negsup_diagnostic.py — H0013 mechanism reads (run on cac-server, post-training).

Loads the node's best.pth, runs one val pass capturing the NegSup suppression
field per image (no training, no gradients), and reports:
  - lam_final (expect decisively > 0 / >= 1e-2)
  - supp active-cell fraction on sparse (gt<50) vs dense (gt>500) val images
    (expect sparse/dense ratio >= 3 — the dense-safety selectivity proof)
  - dense-block + sparse-slice movement vs a baseline perimage dump (optional)

Usage:
    python scripts/negsup_diagnostic.py <node_dir> [--split val]
        [--baseline-perimage /data/repro/baseline_aug_v2/val_perimage.json]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cac.hub as hub  # noqa: E402

hub.setup_hf_env()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cac.config import load_config  # noqa: E402
from cac.data.fsc147 import FSC147Density, collate_density  # noqa: E402
from cac.data.pl_datamodule import worker_init_fn  # noqa: E402
from cac.engine.runner import _model_from_cfg  # noqa: E402

SPARSE_THR = 50.0
DENSE_THR = 500.0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: python scripts/negsup_diagnostic.py <node_dir> [--split val] "
                 "[--baseline-perimage path]")
    nd = args[0]
    split = "val"
    base_per = None
    i = 1
    while i < len(args):
        if args[i] == "--split" and i + 1 < len(args):
            split = args[i + 1]; i += 2
        elif args[i] == "--baseline-perimage" and i + 1 < len(args):
            base_per = args[i + 1]; i += 2
        else:
            sys.exit(f"unknown argument: {args[i]}")

    cfg = load_config(os.path.join(nd, "config.toml"))
    cands = [os.path.join(nd, "run", "latest", "best.pth"), os.path.join(nd, "best.pth")]
    best_path = next((c for c in cands if os.path.exists(c)), None)
    if best_path is None:
        sys.exit(f"{nd}: no best.pth")
    seed = int(cfg.get("seed", 20260830))
    torch.manual_seed(seed)
    model = _model_from_cfg(cfg, nd)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(best_path, map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    lam = float(model.head.negsup.lam.detach().reshape(-1)[0])
    print(f"lam_final = {lam:.6f}  (read 1: expect >= 1e-2)")

    S = int(cfg.get("input_size", 384))
    ds = FSC147Density(str(cfg.get("data_root", "/data/dataset/FSC147")), S, split, augment=False)
    dl = DataLoader(ds, batch_size=int(cfg.get("batch_size", 16)),
                    num_workers=int(cfg.get("num_workers", 4)),
                    collate_fn=collate_density, shuffle=False,
                    generator=torch.Generator().manual_seed(seed),
                    worker_init_fn=worker_init_fn(seed))
    use_cc = bool(cfg.get("use_cellcal", False))

    sp_act, de_act, sp_n, de_n = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for batch in dl:
            imgs = batch["imgs"].to(dev)
            bbox = batch["bboxes"].to(dev)
            b3 = batch.get("bboxes3")
            bboxes_in = b3.to(dev) if b3 is not None else bbox
            if bboxes_in.dim() == 2:
                bboxes_in = bboxes_in.unsqueeze(1)
            h2, h3 = model.backbone.forward_feature_map(imgs)
            fine = model.head.fuser(h2, h3)
            e = model.head.exemplar(h3, bboxes_in, model.head.S)
            kw = {"use_cellcal": True, "w_c": model.head.cellcal.w_c} if use_cc else {}
            _, pos_d = model.head.simprior(fine, e, return_top1=True, **kw)
            adj = model.head.negsup.adjustment(fine.detach(), h3.detach(), bboxes_in, pos_d)
            frac = (adj > 1e-6).float().mean(dim=(1, 2, 3)).cpu()
            for f, c in zip(frac.tolist(), batch["counts"].tolist()):
                g = float(c)
                if g < SPARSE_THR:
                    sp_act += f; sp_n += 1
                if g > DENSE_THR:
                    de_act += f; de_n += 1
    sp_m = sp_act / max(1, sp_n)
    de_m = de_act / max(1, de_n)
    print(f"sparse gt<{SPARSE_THR} (n={sp_n}): supp-active frac = {sp_m:.4f}")
    print(f"dense gt>{DENSE_THR} (n={de_n}): supp-active frac = {de_m:.4f}")
    print(f"selectivity ratio sparse/dense = {sp_m / max(1e-9, de_m):.2f}  (read 2: expect >= 3)")

    if base_per and os.path.exists(base_per):
        bp = json.load(open(base_per))
        node_per = os.path.join(nd, f"{split}_perimage.json")
        if os.path.exists(node_per):
            ip = json.load(open(node_per))
            assert bp["ids"] == ip["ids"], "order mismatch"
            bd = [abs(p - g) for p, g in zip(bp["pred"], bp["gt"])]
            ind = [abs(p - g) for p, g in zip(ip["pred"], ip["gt"])]
            tbl = ["935.jpg", "7656.jpg", "3425.jpg", "865.jpg", "1956.jpg", "6969.jpg",
                   "3433.jpg", "975.jpg", "840.jpg", "3428.jpg", "3665.jpg", "3481.jpg",
                   "3488.jpg", "3436.jpg", "3484.jpg"]
            print("dense-block (id,gt,base,negsup):")
            for i in range(len(bd)):
                if bp["ids"][i] in tbl:
                    print("  ", bp["ids"][i], round(bp["gt"][i]),
                          round(bp["pred"][i], 1), round(ip["pred"][i], 1))
            sp = [i for i in range(len(bd)) if bp["gt"][i] < SPARSE_THR]
            print(f"sparse slice: base {sum(bd[i] for i in sp)/len(sp):.3f} "
                  f"negsup {sum(ind[i] for i in sp)/len(sp):.3f} (read 4: expect <= 5.45)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
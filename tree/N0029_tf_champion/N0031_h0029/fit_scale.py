#!/usr/bin/env python3
"""Recompute H0029 train-fit scale and val/test MAEs from N0015 dumps."""
import json
from pathlib import Path

BASE = Path("/data/cac/tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014")


def load(split):
    d = json.loads((BASE / f"{split}_perimage.json").read_text())
    return d["pred"], d["gt"]


def mae(p, g):
    return sum(abs(a - b) for a, b in zip(p, g)) / len(p)


def main():
    tp, tg = load("train")
    vp, vg = load("val")
    ep, eg = load("test")
    s = sum(a * b for a, b in zip(tp, tg)) / sum(a * a for a in tp)
    out = {
        "scale": s,
        "train_mae_raw": mae(tp, tg),
        "val_mae_raw": mae(vp, vg),
        "val_mae_cal": mae([x * s for x in vp], vg),
        "test_mae_raw": mae(ep, eg),
        "test_mae_cal": mae([x * s for x in ep], eg),
        "bar_val": 18.50,
        "bar_test": 19.70,
        "pass_val": mae([x * s for x in vp], vg) <= 18.50,
        "pass_test": mae([x * s for x in ep], eg) <= 19.70,
    }
    out["verdict"] = "SUPPORTED" if out["pass_val"] and out["pass_test"] else "REFUTED"
    print(json.dumps(out, indent=2))
    Path(__file__).with_name("result.json").write_text(json.dumps({**out, "node": "N0031_h0029", "hyp": "H0029"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""counttau_verdict.py — paired v2 verdict for H0010 (run on cac-server).

Reads baseline + innovation `<split>_result.json` / `<split>_perimage.json`
produced by scripts/eval_test.py (same seed, same val split, same order).

Usage:
    python scripts/counttau_verdict.py <baseline_dir> <innov_dir> [--split val]

Prints the paired table: MAE, RMSE, bias, under-share, dense-tail gt>500 mean
|Δ|, and (when the innovation checkpoint carries `use_counttau`) the mechanism
diagnostics w_g/w_t/temp and the per-image g surrogate spread. Verdict line:
REACHED if innov MAE <= baseline-0.30 AND dense-tail mean |Δ| < baseline's
(dense baseline falls back to the pre-v2 511.7 when baseline has no tail stat).
"""
from __future__ import annotations

import json
import os
import sys

DENSE_THR = 500.0


def _load(dirp, name):
    p = os.path.join(dirp, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def _tail(per, thr=DENSE_THR):
    xs = [(p, g) for p, g in zip(per["pred"], per["gt"]) if g > thr]
    if not xs:
        return None, 0
    return sum(abs(p - g) for p, g in xs) / len(xs), len(xs)


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit("usage: python scripts/counttau_verdict.py <baseline_dir> <innov_dir> [--split val]")
    bdir, idir = args[0], args[1]
    split = "val"
    if "--split" in args:
        split = args[args.index("--split") + 1]

    br, bp = _load(bdir, f"{split}_result.json"), _load(bdir, f"{split}_perimage.json")
    ir, ip = _load(idir, f"{split}_result.json"), _load(idir, f"{split}_perimage.json")
    if not br or not ir or not bp or not ip:
        sys.exit(f"missing results under {bdir} or {idir} (split={split})")

    bt, bn = _tail(bp)
    it, inn = _tail(ip)
    dense_base = bt if bt is not None else 511.7

    def f(x, nd=3):
        return "-" if x is None else round(x, nd)

    print(f"paired {split} verdict (same seed, same protocol v2):")
    print(f"  baseline: n={br['n']} mae={f(br['mae'])} rmse={f(br['rmse'])} "
          f"bias={f(br['bias_mean_delta'])} under={br['n_under']}/{br['n']} "
          f"dense>{DENSE_THR}: n={bn} mean|del|={f(bt)}")
    print(f"  innov   : n={ir['n']} mae={f(ir['mae'])} rmse={f(ir['rmse'])} "
          f"bias={f(ir['bias_mean_delta'])} under={ir['n_under']}/{ir['n']} "
          f"dense>{DENSE_THR}: n={inn} mean|del|={f(it)}")
    dmae = ir["mae"] - br["mae"]
    print(f"  d_mae = {f(dmae,4)}   (bar: <= -0.30  ->  target {f(br['mae']-0.30)})")
    if it is not None and dense_base is not None:
        print(f"  d_dense = {f(it - dense_base,2)}   (bar: < 0  vs dense baseline {f(dense_base)})")
    else:
        print("  dense tail: not comparable (missing innov tail)")

    if "ct_w_g" in ir:
        wg = ir["ct_w_g"]
        print(f"  mechanism: w_g={f(wg[0],4) if wg else '-'} w_t={f(ir.get('ct_w_t',[None])[0],4) if ir.get('ct_w_t') else '-'} "
              f"pinned temp={f(ir.get('ct_simport.temp',[None])[0],5) if ir.get('ct_simport.temp') else '-'} "
              f"g[mean,min,max]={f(ir.get('ct_g_mean')),f(ir.get('ct_g_min')),f(ir.get('ct_g_max'))} "
              f"g_spread={f(ir.get('ct_g_spread'))}")
    reach = (dmae <= -0.30) and (it is not None and it < dense_base)
    print("  VERDICT: " + ("H0010 REACHED (bar met)" if reach else "bar NOT met (below 22.26-v2 target / dense tail not beaten)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
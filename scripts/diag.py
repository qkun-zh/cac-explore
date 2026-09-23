#!/usr/bin/env python3
"""Per-image / log diagnostic: density-bin MAE share + worst offenders.

Usage:
  python3 scripts/diag.py --log /tmp/Nxxxx_sub286.log
  python3 scripts/diag.py --csv /tmp/per_image_....csv

DENSITY_BIN lines come from a run log (preferred; already aggregated).
CSV mode reads entry.py per-image rows: idx,filename,gt,pred,err,abs_err,...
Prints bin table, share of total abs error, worst-15, and a 200+ target hint.
"""
import argparse, ast, csv, re, sys
from pathlib import Path

BINS = [(0, 10), (10, 30), (30, 80), (80, 200), (200, 10 ** 9)]
BIN_RE = re.compile(r"DENSITY_BIN\s+(\{.*\})")
WORST_RE = re.compile(r"^WORST\s+idx=(\d+)\s+file=(\S+)\s+gt=([0-9.]+)\s+pred=([0-9.]+)\s+err=([0-9.-]+)")
EXT_RE = re.compile(r"EXTENDED_METRICS\s+(\{.*\})")


def _bin_name(lo, hi):
    return f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"


def _parse_dict(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def from_log(path: Path):
    text = path.read_text(errors="replace")
    rows, worst, ext = [], [], None
    for line in text.splitlines():
        m = BIN_RE.search(line)
        if m:
            d = _parse_dict(m.group(1))
            if isinstance(d, dict) and "bin" in d:
                rows.append(d)
        m = WORST_RE.match(line.strip())
        if m:
            worst.append(
                {"idx": int(m.group(1)), "file": m.group(2),
                 "gt": float(m.group(3)), "pred": float(m.group(4)),
                 "err": float(m.group(5))}
            )
        m = EXT_RE.search(line)
        if m and ext is None:
            ext = _parse_dict(m.group(1))
    return rows, worst, ext


def from_csv(path: Path):
    bins = [{"bin": _bin_name(lo, hi), "n": 0, "MAE": 0.0, "sum_ae": 0.0}
            for lo, hi in BINS]
    recs = []
    with path.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            gt, pred = float(r["gt"]), float(r["pred"])
            ae = abs(pred - gt)
            err = pred - gt
            recs.append({"idx": int(float(r["idx"])), "file": r.get("filename", ""),
                         "gt": gt, "pred": pred, "err": err})
            for b in bins:
                lo, hi = BINS[bins.index(b)]
                if lo <= gt < hi:
                    b["n"] += 1
                    b["sum_ae"] += ae
                    break
    rows = []
    for b in bins:
        if b["n"] == 0:
            continue
        rows.append({"bin": b["bin"], "n": b["n"],
                     "MAE": b["sum_ae"] / b["n"], "sum_ae": b["sum_ae"]})
    worst = sorted(recs, key=lambda r: -abs(r["err"]))[:15]
    return rows, worst, None


def report(rows, worst, ext):
    if not rows:
        print("no DENSITY_BIN rows / empty csv", file=sys.stderr)
        return 1
    total = sum(r["MAE"] * r["n"] for r in rows)
    n = sum(r["n"] for r in rows)
    mae = total / n if n else float("nan")
    if ext and "MAE" in ext:
        mae = float(ext["MAE"])
        n = int(ext.get("n", n))
    print(f"n={n} mae={mae:.4f} abs_err_sum={total:.2f}")
    print(f"{'bin':<12} {'n':>5} {'MAE':>8} {'share%':>7}")
    dense_sum = 0.0
    for r in rows:
        s = r["MAE"] * r["n"]
        share = 100.0 * s / total if total else 0.0
        if r["bin"].startswith("200"):
            dense_sum = s
        print(f"{r['bin']:<12} {r['n']:>5} {r['MAE']:>8.2f} {share:>7.1f}")
    if dense_sum and n and total:
        cut = 0.5 * dense_sum / n
        print(f"TARGET: 200+ owns {100 * dense_sum / total:.1f}% of abs err; "
              f"halving it moves global MAE by ~{cut:.2f} -> ~{mae - cut:.2f}")
    if worst:
        print("worst15 by |err|:")
        for w in worst[:15]:
            print(f"  idx={w['idx']} gt={w['gt']:.1f} pred={w['pred']:.1f} "
                  f"err={w['err']:.1f} {w['file']}")
    return 0


def main():
    p = argparse.ArgumentParser(description="density-bin diagnostic")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--log", help="run log with DENSITY_BIN/WORST lines")
    g.add_argument("--csv", help="per-image CSV from entry.py")
    a = p.parse_args()
    if a.log:
        path = Path(a.log)
        if not path.is_file():
            sys.exit(f"FAIL: no such log {path}")
        rows, worst, ext = from_log(path)
    else:
        path = Path(a.csv)
        if not path.is_file():
            sys.exit(f"FAIL: no such csv {path}")
        rows, worst, ext = from_csv(path)
    sys.exit(report(rows, worst, ext))


if __name__ == "__main__":
    main()

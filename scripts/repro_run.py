#!/usr/bin/env python3
"""repro_run.py — replicate a node's exact training run into a scratch dir.

Diagnostic only: trains the node's committed config/model via run_train, writes
logs under --out, and NEVER touches the node's info.json/result.json. Used to
measure run-to-run variance (same seed) — the noise floor for eval deltas.

Usage: python scripts/repro_run.py <node_id> --out <dir> [--budget-seconds S]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cac.hub as hub  # noqa: E402

hub.setup_hf_env()

from cac.config import load_config  # noqa: E402
from cac.engine.runner import run_train  # noqa: E402
from cac.expt.node import TrajectoryTree  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: python scripts/repro_run.py <node_id> --out <dir> [--budget-seconds S]")
    node = args[0]
    out = None
    budget = None
    i = 1
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
        elif args[i].startswith("--out="):
            out = args[i][6:]
            i += 1
        elif args[i] == "--budget-seconds" and i + 1 < len(args):
            budget = float(args[i + 1])
            i += 2
        elif args[i].startswith("--budget-seconds="):
            budget = float(args[i].split("=", 1)[1])
            i += 1
        else:
            sys.exit(f"unknown argument: {args[i]}")
    if not out:
        sys.exit("--out is required")

    mat = TrajectoryTree(ROOT).materialize()
    if node not in mat:
        sys.exit(f"node {node} not in tree/")
    nd = mat[node]["path"]
    cfg = load_config(os.path.join(nd, "config.toml"))
    res = run_train(cfg, out, node_dir=nd, budget_seconds=budget)
    keep = {k: res.get(k) for k in ("best_mae", "best_epoch", "n_epochs_done",
                                    "budget_hit", "elapsed")}
    keep["node"] = node
    keep["mode"] = "repro"
    json.dump(keep, open(os.path.join(out, "repro.json"), "w"), indent=2)
    print(json.dumps(keep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

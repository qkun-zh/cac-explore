#!/usr/bin/env python3
"""Engine smoke — verify model/config contract and memory fit on one synthetic step.

Usage: python scripts/smoke.py tree/<node>/config.toml [--node-dir tree/<node>]
The coding agent's card is not done until this exits 0 on the server.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cac.hub as hub  # noqa: E402

hub.setup_hf_env()

from cac.config import load_config  # noqa: E402
from cac.engine.runner import run_train  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/smoke.py <config.toml> [--node-dir <dir>] [--seed N]")
    cfgp = sys.argv[1]
    node_dir = None
    for i, a in enumerate(sys.argv[2:], start=2):
        if a == "--node-dir":
            node_dir = sys.argv[i + 1]
    cfg = load_config(cfgp)
    log_dir = os.path.join(os.path.dirname(cfgp), "run", "smoke")
    res = run_train(cfg, log_dir, node_dir=node_dir, smoke=True)
    print("SMOKE_RESULT", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
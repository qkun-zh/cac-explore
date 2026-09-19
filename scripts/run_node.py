#!/usr/bin/env python3
"""run_node.py — the executor agent's driver on cac-server.

Moves the working dir permanently to the server copy of the chosen node:
    python run_node.py N0002_...
then smoke-test one step, run the whole training budget, dump result.json
(+ log_dir + checksums) where the local repo expects it, and leave on the
server: tree/<node>/run/ logdir, model weights, tensorboard.

Lifecycle handled here:
  1. config exists + parses
  2. model.py exists (or falls back to registry)
  3. smoke (one step) — above the fold, then budget
  4. write result.json + feedback/notes.md pointer by the executor
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cac.hub as hub  # noqa: E402

hub.setup_hf_env()

from cac.config import load_config
from cac.engine.runner import checksum, run_train
from cac.expt.node import TrajectoryTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def booked_hyps(idea_md: str) -> list[str]:
    """Extract the hypothesis ids booked into a node from its idea.md
    ("1. **H0010** — text" lines). Empty list if unparsable."""
    import re
    hyps: list[str] = []
    if not os.path.exists(idea_md):
        return hyps
    for line in open(idea_md):
        m = re.match(r"^\s*\d+\.\s+\*\*([HN][0-9]{4})\*\*", line)
        if m:
            hyps.append(m.group(1))
    return hyps


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("usage: python run_node.py <node_id> [--epochs N] [--budget-seconds S] [--smoke] [--off]")
    node = sys.argv[1]
    flags = sys.argv[2:]
    tree = TrajectoryTree(ROOT)
    mat = tree.materialize()
    if node not in mat:
        sys.exit(f"node {node} not in tree/")

    nd = mat[node]["path"]
    cfgp = os.path.join(nd, "config.toml")
    if not os.path.exists(cfgp):
        sys.exit(f"{node}: missing config.toml")
    cfg = load_config(cfgp)

    log_dir = os.path.join(nd, "run", "latest")
    tree.set_status(node, "running")
    t0 = time.time()
    try:
        kwargs: dict = {}
        i = 0
        while i < len(flags):
            flag = flags[i]
            k, eq, v = flag.partition("=")
            if not eq and k in ("--epochs", "--budget-seconds") and i + 1 < len(flags):
                v = flags[i + 1]
                i += 2
            else:
                i += 1
            if k in ("--epochs", "--budget-seconds"):
                kwargs[k[2:].replace("-", "_")] = float(v) if "budget" in k else int(v)
            elif k == "--smoke":
                kwargs["smoke"] = True
            elif k == "--off":
                kwargs["off"] = True
        res = run_train(cfg, log_dir, node_dir=nd, **kwargs)
    except Exception as e:
        tree.set_status(node, "failed")
        print(f"[failed] {e}", flush=True)
        return 1

    res["node"] = node
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["config_sha256"] = checksum(cfgp)
    res["model_sha256"] = checksum(os.path.join(nd, "model.py")) if os.path.exists(os.path.join(nd, "model.py")) else None
    json.dump(res, open(os.path.join(nd, "result.json"), "w"), indent=2)

    if not res.get("smoke") and not res.get("off"):
        best = res.get("best_mae")
        if best is not None:
            tree.update_metrics(node, best_metric=best, train_seconds=res.get("elapsed_s"),
                                epochs=res.get("n_epochs_done"))
        try:
            tree.mark_tested(node, booked_hyps(os.path.join(nd, "idea.md")))
        except Exception as e:
            print(f"[warn] could not mark tested hypotheses: {e}", flush=True)
        hit = kwargs.get("budget_seconds") and res.get("budget_hit")
        tree.set_status(node, "timeout" if hit else "done")
    else:
        tree.set_status(node, "done" if res.get("off") else "done")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
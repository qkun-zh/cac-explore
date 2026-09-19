#!/usr/bin/env python3
"""curve_check.py — print val/mae curve summary for a node on cac-server.
Usage: /data/miniconda/envs/cac/bin/python /data/cac/curve_check.py <node_id>
"""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

node = sys.argv[1]
fs = sorted(
    glob.glob(f"/data/cac/tree/N0001_champion/{node}/run/latest/tb/t/events*"),
    key=os.path.getmtime,
)
if not fs:
    print(f"{node}: no tb yet")
    sys.exit(0)
ea = EventAccumulator(fs[-1])
ea.Reload()
tags = ea.Tags()["scalars"]
if "val/mae" not in tags:
    print(f"{node}: no val/mae")
    sys.exit(0)
ev = ea.Scalars("val/mae")
best_i = min(range(len(ev)), key=lambda i: ev[i].value)
print(f"{node}: epochs={len(ev)} best={ev[best_i].value:.4f}@ep{best_i + 1} "
      f"last={ev[-1].value:.4f}")
print("tail:", [round(e.value, 3) for e in ev[-9:]])

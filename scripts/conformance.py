#!/usr/bin/env python3
"""Conformance gate — run BEFORE every commit, must exit 0.

Checks (via cac.expt.gates.Conformance):
  1. filesystem lineage == info.json lineage for every node
  2. every status in the canonical set; required fields present & typed
  3. ledger: format gates, phantom-id ban, evidence_type & strength bounds,
     ts present
  4. journal jsonl parseable
  5. index.json exists (rebuildable)
  6. PROTOCOL PINS (2026-09-19): the runner must keep honoring cfg.augment /
     cfg.seed / cfg.num_workers into FSC147DataModule and cfg.deterministic
     into cudnn flags. These pins exist because a migration silently dropped
     the augment/seed wiring and cost ~2 MAE. Plain source scans: if a refactor
     moves them, update the pins — never delete them.

 Common failures every module/agent hits before a bump:
   - edited info.json instead of using discovery (leaves lineage drift)
   - appended an evidence before a create (phantom)
   - nested node outside its parent dir
+  - appended journal by hand and glued two JSON objects on one line
+    (use scripts/journal.py)
+  - unwired a protocol key in the runner (conformance pins catch it)
 Exit code 1 = BLOCK COMMIT.
 """
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cac.expt.gates import Conformance


def _protocol_pins() -> list[str]:
    """Regression pins for the 2026-09-19 protocol lessons (journal)."""
    errs: list[str] = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner = os.path.join(root, "src", "cac", "engine", "runner.py")
    try:
        src = open(runner).read()
    except OSError as e:
        return [f"cannot read runner.py: {e}"]
    checks = [
        ('augment=bool(cfg.get("augment"',
         'cfg.augment must reach FSC147DataModule (silent no-aug cost ~2 MAE)'),
        ('num_workers=int(cfg.get("num_workers"',
         'cfg.num_workers must reach FSC147DataModule'),
        ('cfg.get("deterministic"',
         'cudnn determinism must keep honoring cfg.deterministic'),
        ("torch.backends.cudnn.deterministic = True",
         'deterministic cudnn flag must exist in runner'),
    ]
    for needle, why in checks:
        if needle not in src:
            errs.append(f"protocol pin missing: {why}")
    if src.count('seed=int(cfg.get("seed"') < 2:
        errs.append("protocol pin missing: cfg.seed must wire into BOTH smoke and real datamodules")
    return errs


def main() -> int:
    c = Conformance()
    rep = c.report()
    pin_errs = _protocol_pins()
    errors = list(rep["errors"]) + pin_errs
    ok = rep["ok"] and not pin_errs
    if ok:
        print("CONFORMANCE OK")
        for w in rep["warnings"]:
            print(f"  warn: {w}")
        return 0
    print("CONFORMANCE FAILED — do not commit")
    for e in errors:
        print(f"  ERR: {e}")
    for w in rep["warnings"]:
        print(f"  warn: {w}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
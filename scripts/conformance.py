#!/usr/bin/env python3
"""Conformance gate — run BEFORE every commit, must exit 0.

Checks (via cac.expt.gates.Conformance):
  1. filesystem lineage == info.json lineage for every node
  2. every status in the canonical set; required fields present & typed
  3. ledger: format gates, phantom-id ban, evidence_type & strength bounds,
     ts present
  4. journal jsonl parseable
  5. index.json exists (rebuildable)

Common failures every module/agent hits before a bump:
  - edited info.json instead of using discovery (leaves lineage drift)
  - appended an evidence before a create (phantom)
  - nested node outside its parent dir
Exit code 1 = BLOCK COMMIT.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cac.expt.gates import Conformance


def main() -> int:
    c = Conformance()
    rep = c.report()
    if rep["ok"]:
        print("CONFORMANCE OK")
        for w in rep["warnings"]:
            print(f"  warn: {w}")
        return 0
    print("CONFORMANCE FAILED — do not commit")
    for e in rep["errors"]:
        print(f"  ERR: {e}")
    for w in rep["warnings"]:
        print(f"  warn: {w}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
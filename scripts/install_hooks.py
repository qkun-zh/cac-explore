#!/usr/bin/env python3
"""Install .git/hooks/pre-commit to run scripts/check.py. Safe to re-run."""
from pathlib import Path
import os, stat, sys

ROOT = Path(__file__).resolve().parent.parent
HOOK = ROOT / ".git" / "hooks" / "pre-commit"
BODY = """#!/bin/sh
# cac: block commits while scripts/check.py is red
cd "$(git rev-parse --show-toplevel)" || exit 1
exec python3 scripts/check.py
"""


def main() -> int:
    if not (ROOT / ".git").exists():
        print("no .git here; skip hook install")
        return 0
    HOOK.parent.mkdir(parents=True, exist_ok=True)
    HOOK.write_text(BODY)
    HOOK.chmod(HOOK.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed {HOOK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

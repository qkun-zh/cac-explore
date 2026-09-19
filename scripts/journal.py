#!/usr/bin/env python3
"""journal.py — the ONLY way to append journal/events.jsonl.

Usage: python scripts/journal.py --kind <kind> --detail "<text>"
Writes exactly one JSON object per line (keys ts/kind/detail) with a trailing
newline. This exists because ad-hoc appends produced concatenated JSON objects
(two events glued on one line) twice — breaking conformance every time.
Never write events.jsonl by hand, heredoc, or `cat >>`; always use this CLI.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL = os.path.join(ROOT, "journal", "events.jsonl")


def main() -> int:
    p = argparse.ArgumentParser(prog="journal")
    p.add_argument("--kind", required=True, help="event kind (e.g. run, evidence, repair, booking, baseline, protocol_decision, hygiene)")
    p.add_argument("--detail", required=True, help="one-paragraph factual detail")
    args = p.parse_args()
    kind, detail = args.kind.strip(), args.detail.strip()
    if not kind or not detail:
        sys.exit("kind and detail must be non-empty")
    if "\n" in kind or "\r" in kind:
        sys.exit("kind must be a single line")
    ev = {"ts": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          "kind": kind, "detail": detail.replace("\n", " ")}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    print(f"journaled [{kind}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

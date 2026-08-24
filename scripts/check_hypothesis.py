#!/usr/bin/env python3
"""Hypothesis structure validator (quality gate, pre-booking).

Enforces the canonical format:
  IF [choice] IN [scope], THEN [effect], BECAUSE [mechanism]. DISPROVED IF [criterion].

Checks (7-dim gate, automated subset): keyword presence & order, non-empty
segments, measurable falsifier (number or comparison), scope present,
mechanism length. Exits 1 on failure.

Usage:
  python3 scripts/check_hypothesis.py --text "IF ..."
  python3 scripts/check_hypothesis.py --file idea.md
  python3 scripts/check_hypothesis.py --all          # validate ledger creates
  echo "IF ..." | python3 scripts/check_hypothesis.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "memory" / "hypotheses.jsonl"

MARKERS = ["IF", "IN", "THEN", "BECAUSE", "DISPROVED"]
MARKER_RE = [(m, re.compile(rf"\b{m}\b")) for m in MARKERS]
FALSIFIER_RE = re.compile(r"DISPROVED\s+IF\s*(.+)$", re.DOTALL)
MEASURE = re.compile(r"(<=|>=|==|<|>|\d)", re.IGNORECASE)
MECHANISM_MIN = 20


def validate(text: str):
    errors, warnings = [], []
    raw = text.strip()
    if not raw:
        return ["empty text"], []

    pos, last = [], -1
    for m, rx in MARKER_RE:
        mt = rx.search(raw)
        if not mt:
            errors.append(f"missing marker {m!r}")
            pos.append(None)
        else:
            pos.append(mt.start())
            if mt.start() < last:
                errors.append(f"marker {m!r} out of order")
            last = max(last, mt.start())

    complete = len(pos) == 5 and all(p is not None for p in pos)

    def seg(a, b):
        return raw[pos[a] + len(MARKERS[a]):pos[b]].strip(" ,.:;")

    choice = seg(0, 1) if complete else None
    scope = seg(1, 2) if complete else None
    effect = seg(2, 3) if complete else None
    mech = seg(3, 4) if complete else None

    if choice is not None and not choice:
        errors.append("empty [choice]")
    if scope is not None and not scope:
        errors.append("empty [scope]")
    if effect is not None and not effect:
        errors.append("empty [effect]")
    if mech is not None and not mech:
        errors.append("empty [mechanism] — BECAUSE must give a causal story")
    elif mech and len(mech) < MECHANISM_MIN:
        warnings.append(f"mechanism very short ({len(mech)} chars)")

    fm = FALSIFIER_RE.search(raw)
    falsifier = fm.group(1).strip() if fm else ""
    if "DISPROVED" in raw:
        if not falsifier:
            errors.append("empty falsification criterion")
        elif not MEASURE.search(falsifier):
            errors.append("falsifier not measurable: needs a number/comparison (e.g. MAE>=X)")
        if re.search(r"\bOR\b", falsifier) and not re.search(r"\bAND\b", falsifier):
            warnings.append("compound OR-falsifier: ensure every disjunct alone kills the hyp")

    if len(raw) > 1200:
        warnings.append(f"hypothesis very long ({len(raw)} chars); consider splitting")
    if re.search(r"\bmaybe\b|\bperhaps\b|\bmight\b", raw, re.IGNORECASE):
        warnings.append("hedging language in a falsifiable claim")
    return errors, warnings


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--text")
    g.add_argument("--file")
    g.add_argument("--all", action="store_true", help="validate all create events in ledger")
    args = ap.parse_args()

    if args.all:
        n_pass = n_fail = 0
        for line in LEDGER.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("type") != "create":
                continue
            errs, warns = validate(ev.get("text", ""))
            hid = ev.get("hyp_id", "?")
            status = "FAIL" if errs else "pass"
            n_fail += bool(errs)
            n_pass += not errs
            print(f"{hid}: {status}")
            for e in errs:
                print(f"   ERROR {e}")
            for w in warns:
                print(f"   warn  {w}")
        print(f"\n{n_pass} pass, {n_fail} fail")
        sys.exit(1 if n_fail else 0)

    text = args.text or (Path(args.file).read_text() if args.file else sys.stdin.read())
    errs, warns = validate(text)
    for e in errs:
        print(f"ERROR {e}", file=sys.stderr)
    for w in warns:
        print(f"warn  {w}")
    if errs:
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()

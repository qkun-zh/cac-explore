#!/usr/bin/env python3
"""Calibration monitor (arXiv:2604.12999 §4.3).

Replays memory/hypotheses.jsonl in file (ledger) order, reconstructs the
confidence c(h) just before each evidence event, and reports whether the
hypothesis prediction was correct (supports=hit, contradicts=miss),
binned by confidence-at-test. Read-only.

Usage: python3 scripts/calibration_report.py [--eta 0.20] [--json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "memory" / "hypotheses.jsonl"

BINS = [(0.25, 0.50), (0.50, 0.75), (0.75, 1.001)]
CLASS_CONFIRMED = 0.75
CLASS_REFUTED = 0.25


def load_events():
    events, bad = [], []
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
            ev["_line"] = i
            events.append(ev)
        except json.JSONDecodeError as e:
            bad.append((i, str(e)))
    return events, bad


def bin_of(c):
    for lo, hi in BINS:
        if lo <= c < hi:
            return f"[{lo:.2f},{min(hi,1.0):.2f})"
    return "<0.25"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=0.20)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    events, bad = load_events()
    if bad:
        print(f"WARN: {len(bad)} unparseable lines: {bad}", file=sys.stderr)

    conf = {}       # hyp_id -> current confidence
    meta = {}       # hyp_id -> {"text":..., "tags":...}
    tests = []      # {"hyp","conf_before","hit","node","strength"}
    warns = []

    for ev in events:
        hid = ev.get("hyp_id")
        typ = ev.get("type")
        ts = ev.get("ts", "?")
        if typ == "create":
            conf[hid] = float(ev.get("confidence", 0.5))
            meta[hid] = {"text": ev.get("text", ""), "tags": ev.get("tags", [])}
        elif typ == "evidence":
            if hid not in conf:
                warns.append(f"L{ev['_line']}: evidence for unknown {hid} (skipped)")
                continue
            etype = ev.get("evidence_type")
            w = float(ev.get("strength", 1.0))
            c0 = conf[hid]
            if etype == "supports":
                hit, c1 = True, c0 + args.eta * w * (1 - c0)
            elif etype == "contradicts":
                hit, c1 = False, c0 - args.eta * w * c0
            else:  # neutral: logged, not scored
                hit, c1 = None, c0
            tests.append({"hyp": hid, "conf_before": round(c0, 4),
                          "hit": hit, "node": ev.get("source_node", "?"),
                          "strength": w})
            conf[hid] = min(0.99, max(0.01, c1))
        else:
            warns.append(f"L{ev['_line']}: unknown event type {typ!r}")

    # bin aggregation over scored tests
    agg = {f"[{lo:.2f},{min(hi,1.0):.2f})": {"n": 0, "hits": 0} for lo, hi in BINS}
    agg["<0.25"] = {"n": 0, "hits": 0}
    for t in tests:
        if t["hit"] is None:
            continue
        b = bin_of(t["conf_before"])
        agg[b]["n"] += 1
        agg[b]["hits"] += int(t["hit"])

    if args.json:
        print(json.dumps({"bins": agg, "tests": tests,
                          "standings": {h: {"confidence": round(c, 4)} for h, c in sorted(conf.items())},
                          "warnings": warns}, indent=2))
        return

    print("=== Hypothesis Prediction Calibration (eta=%.2f) ===" % args.eta)
    print(f"{'conf@test':<14}{'N':>4}{'correct':>9}{'rate':>8}")
    total_n = total_hits = 0
    for b in [x for x, _ in zip(agg.keys(), range(len(agg)))]:
        n, h = agg[b]["n"], agg[b]["hits"]
        total_n += n
        total_hits += h
        rate = f"{h/n:.0%}" if n else "-"
        print(f"{b:<14}{n:>4}{h:>9}{rate:>8}")
    if total_n:
        rate = f"{total_hits/total_n:.0%}"
        print(f"{'overall':<14}{total_n:>4}{total_hits:>9}{rate:>8}")

    print("\n=== Current Standings ===")
    print(f"{'hyp':<7}{'conf':>7}{'class':<11}{'tests':>6}")
    for h, c in sorted(conf.items()):
        cls = ("confirmed" if c > CLASS_CONFIRMED
               else "refuted" if c < CLASS_REFUTED else "uncertain")
        n = sum(1 for t in tests if t["hyp"] == h)
        print(f"{h:<7}{c:>7.3f}{cls:<11}{n:>6}")

    lag = [h for h, c in conf.items() if CLASS_REFUTED <= c < 0.45
           and any(t["hyp"] == h and t["hit"] is False for t in tests)]
    if lag:
        print("\nNOTE classification lag: " + ", ".join(sorted(lag))
              + " hold refuting evidence but sit >=0.25 under eta=%.2f;" % args.eta)
        print("single-event refutations cannot cross the refuted bar.")
        print("Treat 'refuted' in STATE.md as the operational set; ledger conf is advisory.")

    if warns:
        print("\nWARNINGS:")
        for w in warns:
            print(" -", w)


if __name__ == "__main__":
    main()

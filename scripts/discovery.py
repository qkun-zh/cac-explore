#!/usr/bin/env python3
"""Discovery CLI — the agent's exclusive front door to the evolution machinery.

Usage (run from repo root):
  discovery tree                       - print the trajectory tree (ASCII)
  discovery parent                     - Eq.2-4: pick parent node (deterministic)
  discovery hypo  <parent> [--dry]     - Eq.5-6: propose hypotheses, write node files (or --dry print only)
  discovery validate [--all]           - format gate on memory/hypotheses.jsonl
  discovery rebuild                    - rebuild memory/index.json (TF-IDF + structural novelty judge)
  discovery calibration [--json]       - hypothesis-prediction calibration report
  discovery evidence <hyp_id> --type <t> --strength <w> [--node <n>]
                                        - append one evidence event (append-only; phantom ids banned)
  discovery prove <hyp_id>             - show confidence math for a hypothesis
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cac.expt import Memory, TrajectoryTree, select_hypo, select_parent, validate
from cac.expt.constants import CONF_CONFIRMED, CONF_INIT, CONF_REFUTED, EVIDENCE_TYPES


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="discovery")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("tree", help="render the trajectory tree")
    d.add_argument("--json", action="store_true")

    d = sub.add_parser("parent", help="select parent node (Eq.2-4)")
    d.add_argument("--json", action="store_true")

    d = sub.add_parser("hypo", help="select hypotheses & instantiate a child node")
    d.add_argument("parent", help="parent node id")
    d.add_argument("--dry", action="store_true", help="print candidates only, write nothing")
    d.add_argument("--seed", type=int, default=None, help="manual seed (disables default seed)")
    d.add_argument("--new", action="append", dest="new_hyps", metavar="\"IF ... DISPROVED IF ...\"",
                   help="pre-register 1+ NEW hypotheses (format + novelty gated) and add them to the pool")
    d.add_argument("--book", action="append", dest="book_hyps", metavar="H0012",
                   help="mandate 1+ EXISTING hypothesis ids into this child (selected adjoins are then "
                        "capped at one compatible hypothesis per the composition-feasibility rule)")
    d.add_argument("--solo", action="store_true", help="test ONLY the mandated/new hypothesis, no adjoin")

    d = sub.add_parser("validate", help="format-gate the ledger")
    d.add_argument("--all", action="store_true", help="report all violations (not just first per hyp)")

    d = sub.add_parser("rebuild", help="rebuild memory/index.json")

    d = sub.add_parser("calibration", help="calibration report")
    d.add_argument("--json", action="store_true")

    d = sub.add_parser("evidence", help="append an evidence event (append-only)")
    d.add_argument("hyp_id")
    d.add_argument("--type", dest="etype", required=True, choices=sorted(EVIDENCE_TYPES),
                   help="supports | contradicts | neutral")
    d.add_argument("--strength", type=float, default=1.0,
                   help="weight in [0,1] (default 1.0)")
    d.add_argument("--node", default="?", help="source node id (for the audit trail)")
    d.add_argument("--note", default="",
                   help="short human note (goes into the event json)")

    d = sub.add_parser("prove", help="show the confidence math for a hypothesis")
    d.add_argument("hyp_id")

    d = sub.add_parser("feedback", help="write a feedback note from a node")
    d.add_argument("node")
    d.add_argument("text", nargs="?", default=None)
    return p.parse_args(argv)


def cmd_tree(args) -> None:
    t = TrajectoryTree()
    mat = t.materialize()
    if args.json:
        out = {nid: {k: r.get(k) for k in ("status", "best_metric", "parent", "title")}
               for nid, r in mat.items()}
        print(json.dumps(out, indent=2, sort_keys=True))
        return
    print(t.render())


def cmd_parent(args) -> None:
    tree = TrajectoryTree()
    mat = tree.materialize()
    if not mat:
        print("tree is empty — run `scripts/boot.py`? no: push the champion using `python -m cac.ops.import_champion`")
        return
    mem = Memory()
    idx = mem.build_index()
    nid = select_parent(tree, idx)
    if nid and args.json:
        print(json.dumps({"parent": nid}, indent=2))


def cmd_hypo(args) -> None:
    tree = TrajectoryTree()
    mem = Memory()
    idx = mem.build_index()
    parent = args.parent
    if parent not in tree.materialize():
        sys.exit(f"no such node: {parent}")

    # pre-register NEW hypotheses (the paper's generator role) — gated only
    mandated: list[str] = []
    if args.book_hyps:
        for hid in args.book_hyps:
            if hid not in idx.get("hypotheses", {}):
                sys.exit(f"--book unknown hypothesis id: {hid}")
            mandated.append(hid)
        print(f"[ledger] mandated existing: {', '.join(mandated)}")

    if args.new_hyps:
        for text in args.new_hyps:
            errs, _ = validate(text)
            if errs:
                sys.exit(f"new hypothesis fails format gate: {errs}")
            from novelty_check import check
            if check(text) != 0:
                sys.exit("new hypothesis fails the novelty gate — restate with a fresh mechanism")
            nid = _next_hyp_id(mem)
            mem.create(nid, text, source=parent)
            idx = mem.build_index()
            mandated.append(nid)
            print(f"[ledger] created {nid}")

    rs = select_hypo(parent, idx, seed=args.seed, verbose=True, mandated=mandated or None,
                     max_adjoin=0 if args.solo else 1)
    if not rs:
        print("no candidates — all uncertain hypotheses already tested on this ancestry")
        print("hint: pre-register new hypotheses with `hypo <parent> --new \"IF ...\"` "
              "(or author them via the idea agent and --new them)")
        return
    if args.dry:
        print(f"[dry] would instantiate child of {parent} with: {rs}")
        return

    child_id = _child_name(tree, parent, rs)
    hyp = idx.get("hypotheses", {})
    lines = [f"# {child_id} — evolution of {parent}", "",
             "## Hypothesis Set",
             f"The executor agent tests the following {len(rs)} hypotheses. Test each exactly ONCE,",
             "in one job, minimizing wall-clock. Run the training, observe the val metrics, then",
             "record the outcome in the ledger and in feedback/notes.md."]
    for i, h in enumerate(rs, 1):
        ev = hyp.get(h, {})
        lines += [f"{i}. **{h}** — {ev.get('text', '?')}",
                  f"   → current conf {float(ev.get('confidence', CONF_INIT)):.3f} — one test event",
                  ""]
    lines += ["## Procedure", "1. Copy parent's config.toml here; edit ONLY to encode the hypotheses.",
              "2. Copy parent's model.py here (or modify) exactly as constrained by the hypotheses.",
              "3. On cac-server: transfer this dir, run `python run_node.py <node_id>`.",
              "4. When done: record evidence in the ledger + write feedback/notes.md + result.json."]
    child_path = tree.register(child_id, parent=parent,
                               title=f"evolve {parent} on {rs[0]} (+{len(rs)-1})")
    with open(os.path.join(child_path, "idea.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[created] {child_id} nested under {parent}")
    print(f"[dir     ] {child_path}")
    for todo in ["idea.md", "config.toml", "model.py", "feedback/"]:
        print(f"[todo    ] {todo} — write next, then conformance")


def _next_hyp_id(mem: Memory) -> str:
    nums = [int(h[1:]) for h in mem.hyp_ids() if h.startswith("H") and h[1:].isdigit()]
    return f"H{max(nums, default=0) + 1:04d}"


def _child_name(tree: TrajectoryTree, parent: str, rs: list[str]) -> str:
    base = tree.next_id().rstrip("_")
    tag = (rs[0] or "evolve").lower()
    tag = re.sub(r"[^a-z0-9]", "", tag)[:12]
    return f"{base}_{tag}" if tag else base + "_evolve"


def cmd_validate(args) -> None:
    mem = Memory()
    n_bad = 0
    for i, ev in enumerate(mem.events(), 1):
        if ev.get("type") != "create":
            continue
        errs, _ = validate(ev.get("text", ""))
        if errs:
            n_bad += 1
            print(f"L{i} {ev.get('hyp_id')}: {errs}")
    print(f"validated {sum(1 for e in mem.events() if e.get('type')=='create')} hypotheses — "
          f"{n_bad} bad" + (" (all good)" if n_bad == 0 else ""))


def cmd_rebuild(args) -> None:
    idx = Memory().build_index()
    print(f"rebuilt memory/index.json — {len(idx['hypotheses'])} hypotheses indexed")


def cmd_calibration(args) -> None:
    from cac.expt.gates import Calibration
    c = Calibration()
    print(c.report(as_json=args.json))


def cmd_evidence(args) -> None:
    mem = Memory()
    if not (0.0 <= args.strength <= 1.0):
        sys.exit("strength must be in [0,1]")
    mem.evidence(args.hyp_id, args.etype, args.strength, args.node, note=args.note)
    c = mem.build_index()["hypotheses"][args.hyp_id]["confidence"]
    print(f"recorded {args.etype} w={args.strength} on {args.hyp_id} — conf now {c:.4f}")


def cmd_prove(args) -> None:
    mem = Memory()
    ev = mem.build_index()["hypotheses"].get(args.hyp_id)
    if ev is None:
        sys.exit(f"no such hypothesis: {args.hyp_id}")
    c = ev.get("confidence", CONF_INIT)
    print(f"{args.hyp_id}  conf={c:.4f} ({_status(c)})\n")
    for e in ev.get("log", []):
        if not e.get("evidence_type"):
            continue
        print(f"  [{e.get('source_node', e.get('node', '?'))}] {e.get('evidence_type'):<12} w={e.get('strength', 0.0):.2f}  {e.get('note', '')}")
    total = {k: sum(float(e.get("strength", 1.0)) for e in ev.get("log", [])
                     if e.get("evidence_type") == k) for k in EVIDENCE_TYPES}
    a = 1.0 + total["supports"]
    b = 1.0 + total["contradicts"] + total["neutral"]
    ok = a + b > 2
    mode = (a - 1) / (a + b - 2) if ok else None
    print(f"\nposterior ~ Beta({a:.2f}, {b:.2f})  mode={'{:.4f}'.format(mode) if ok else '(-)'}")


def _status(c: float) -> str:
    if c > CONF_CONFIRMED:
        return "confirmed"
    if c < CONF_REFUTED:
        return "refuted"
    return "uncertain"


def main() -> None:
    args = _parse_args(sys.argv[1:])
    {"tree": cmd_tree, "parent": cmd_parent, "hypo": cmd_hypo, "validate": cmd_validate,
     "rebuild": cmd_rebuild, "calibration": cmd_calibration, "evidence": cmd_evidence,
     "prove": cmd_prove}[args.cmd](args)


if __name__ == "__main__":
    main()
"""Calibration monitor + conformance checks.

Calibration (§hypothesis-prediction calibration): replay the ledger in order,
reconstruct confidence just before each evidence event, bin by
confidence-at-test, count supports-as-hits. `neutral` events are logged but
not scored.

Conformance: the anti-drift guard. Verifies that the filesystem tree, the
memory ledger, the journal, and the constants are all internally consistent.
Every commit is gated on `scripts/conformance.py` being green.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from cac.expt.constants import (CONF_CEIL, CONF_FLOOR, CONF_REFUTED, CONF_CONFIRMED,
                                ETA, EVIDENCE_TYPES, K_SYNTH, TAU_MAX_SECONDS)
from cac.expt.hypothesis import Memory, validate
from cac.expt.node import (EXPANDABLE, INFO_REQUIRED, INFO_TYPES, NODE_RE, STATUSES,
                           TrajectoryTree)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BINS = [(0.25, 0.50), (0.50, 0.75), (0.75, 1.001)]


def _bin_of(c: float) -> str:
    for lo, hi in BINS:
        if lo <= c < hi:
            return f"[{lo:.2f},{min(hi, 1.0):.2f})"
    return "<0.25"


class Calibration:
    def __init__(self, repo: str = ROOT, eta: float = ETA):
        self.repo = repo
        self.eta = eta
        self.ledger = os.path.join(repo, "memory", "hypotheses.jsonl")

    def report(self, as_json: bool = False) -> str | dict[str, Any]:
        if not os.path.exists(self.ledger):
            empty = {"bins": {}, "tests": [], "standings": {}, "warnings": []}
            return json.dumps(empty, indent=2) if as_json else "(no ledger)"
        conf: dict[str, float] = {}
        tests: list[dict[str, Any]] = []
        warns: list[str] = []
        for i, line in enumerate(open(self.ledger), 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as e:
                warns.append(f"L{i}: unparseable — {e}")
                continue
            hid = ev.get("hyp_id")
            typ = ev.get("type")
            if typ == "create":
                conf[hid] = float(ev.get("confidence", 0.5))
            elif typ == "evidence":
                if hid not in conf:
                    warns.append(f"L{i}: evidence before create for {hid}")
                    continue
                etype = ev.get("evidence_type")
                w = float(ev.get("strength", 1.0))
                c0 = conf[hid]
                if etype == "supports":
                    hit, c1 = True, c0 + self.eta * w * (1 - c0)
                elif etype == "contradicts":
                    hit, c1 = False, c0 - self.eta * w * c0
                else:  # neutral: logged, not scored
                    hit, c1 = None, c0
                tests.append({"hyp": hid, "conf_before": round(c0, 4), "hit": hit,
                              "node": ev.get("source_node", "?"), "strength": w})
                conf[hid] = min(CONF_CEIL, max(CONF_FLOOR, c1))
            else:
                warns.append(f"L{i}: unknown event type {typ!r}")

        agg = {f"[{lo:.2f},{min(hi, 1.0):.2f})": {"n": 0, "confirm": 0, "conf_sum": 0.0}
               for lo, hi in BINS}
        agg["<0.25"] = {"n": 0, "confirm": 0, "conf_sum": 0.0}
        dir_n = dir_ok = 0
        for t in tests:
            c0 = t["conf_before"]
            if t["hit"] is not None:
                b = _bin_of(c0)
                agg[b]["n"] += 1
                agg[b]["confirm"] += int(t["hit"])
                agg[b]["conf_sum"] += c0
                if c0 != 0.5:  # decisive prediction: high→support, low→contradict
                    t["dir_right"] = (c0 > 0.5) == bool(t["hit"])
                    dir_n += 1
                    dir_ok += int(t["dir_right"])

        rel_error = None
        acc_dir = None
        if dir_n:
            acc_dir = dir_ok / dir_n
        total_n = total_c = 0
        for v in agg.values():
            total_n += v["n"]
            total_c += v["confirm"]
        if total_n:
            rel_error = sum(
                v["n"] / total_n * abs(v["confirm"] / v["n"] - v["conf_sum"] / v["n"])
                for v in agg.values() if v["n"])
        for v in agg.values():
            if v["n"]:
                v["rate"] = v["confirm"] / v["n"]
                v["mean_conf"] = v["conf_sum"] / v["n"]
            else:
                v["rate"] = v["mean_conf"] = None
            del v["conf_sum"]

        if as_json:
            return {"bins": agg, "tests": tests,
                    "standings": {h: round(c, 4) for h, c in sorted(conf.items())},
                    "reliability_error": rel_error,
                    "reliability_warn": bool(rel_error is not None and rel_error > 0.2),
                    "direction": {"decisive": dir_n, "correct": dir_ok,
                                  "accuracy": acc_dir},
                    "warnings": warns}

        lines = [f"=== Hypothesis Prediction Calibration (eta={self.eta:.2f}) ===",
                 f"{'conf@test':<14}{'N':>4}{'confirm':>8}{'rate':>8}{'pred_conf':>11}"]
        for b, v in agg.items():
            n, c = v["n"], v["confirm"]
            rate = f"{v['rate']:.0%}" if n else "-"
            pred = f"{v['mean_conf']:.2f}" if n else "-"
            lines.append(f"{b:<14}{n:>4}{c:>8}{rate:>8}{pred:>11}")
        if total_n:
            lines.append(f"{'overall':<14}{total_n:>4}{total_c:>8}"
                         f"{f'{total_c/total_n:.0%}':>8}      -")
        if rel_error is not None:
            lines.append(f"reliability error (weighted |rate − pred_conf|): {rel_error:.3f}")
            if rel_error > 0.2:
                lines.append("WARNING: reliability error > 0.20 — confidence at test is drifting")
        if acc_dir is not None:
            lines.append(f"direction accuracy (decisive tests, c≠0.5): {dir_ok}/{dir_n} = {acc_dir:.0%}")
        lines.append("\n=== Current Standings ===")
        lines.append(f"{'hyp':<7}{'conf':>7}{'class':<11}{'tests':>6}")
        for h, c in sorted(conf.items()):
            cls = ("confirmed" if c > CONF_CONFIRMED else "refuted" if c < CONF_REFUTED else "uncertain")
            n = sum(1 for t in tests if t["hyp"] == h)
            lines.append(f"{h:<7}{c:>7.3f}{cls:<11}{n:>6}")
        if warns:
            lines.append("\nWARNINGS:")
            lines += [" - " + w for w in warns]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# conformance (anti-drift / anti-gaming guard)
# ---------------------------------------------------------------------------

class Conformance:
    def __init__(self, repo: str = ROOT):
        self.repo = repo
        self.tree = TrajectoryTree(repo)
        self.mem = Memory(repo)
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _err(self, msg: str) -> None:
        self.errors.append(msg)

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def check_tree(self) -> None:
        mat = self.tree.materialize()
        dup = {nid for nid in mat if sum(1 for r in mat.values() if r["id"] == nid) > 1}
        if dup:
            self._err(f"duplicate node ids in tree: {sorted(dup)}")
        for nid, r in sorted(mat.items()):
            if not NODE_RE.match(nid):
                self._err(f"node dir name violates pattern: {nid}")
            ip = os.path.join(r["path"], "info.json")
            if not os.path.exists(ip):
                self._err(f"{nid}: missing info.json")
                continue
            info = json.load(open(ip))
            for k in INFO_REQUIRED:
                if k not in info:
                    self._err(f"{nid}: info.json missing required field {k!r}")
            for k, t in INFO_TYPES.items():
                if k in info and not isinstance(info[k], t):
                    self._err(f"{nid}: info.json field {k!r} has wrong type {type(info[k]).__name__}")
            if info.get("status") not in STATUSES:
                self._err(f"{nid}: invalid status {info.get('status')!r}")
            parent = info.get("parent")
            if parent is not None:
                if parent not in mat:
                    self._err(f"{nid}: declared parent {parent!r} not a node")
                if r.get("parent_id") != parent:
                    self._err(f"{nid}: declared parent {parent!r} != physical parent {r.get('parent_id')!r} "
                              "(filesystem lineage and info.json disagree)")
            rel = os.path.relpath(r["path"], os.path.join(self.repo, "tree"))
            actual_parent_id = r.get("parent_id")
            if parent is None and actual_parent_id is not None:
                self._err(f"{nid}: declared root but physically nested under {actual_parent_id}")
            for h in info.get("tested_hypotheses", []):
                if h not in self.mem.hyp_ids():
                    self._err(f"{nid}: tested_hypotheses references unknown id {h}")
            if info.get("status") in EXPANDABLE:
                if info.get("best_metric") is None:
                    self._err(f"{nid}: expandable ({info['status']}) but best_metric is null")

    def check_ledger(self) -> None:
        created: set[str] = set()
        for i, ev in enumerate(self.mem.events(), 1):
            hid = ev.get("hyp_id")
            typ = ev.get("type")
            if ev.get("ts") is None:
                self._err(f"L{i}: event without ts")
            if typ == "create":
                created.add(hid)
                text = ev.get("text") or ""
                errs, _ = validate(text)
                if errs:
                    self._err(f"L{i}: create {hid} fails format gate: {errs}")
            elif typ == "evidence":
                if hid not in created:
                    self._err(f"L{i}: evidence for {hid} before its create (phantom id)")
                if ev.get("evidence_type") not in EVIDENCE_TYPES:
                    self._err(f"L{i}: unknown evidence_type {ev.get('evidence_type')!r}")
                w = ev.get("strength")
                if not isinstance(w, (int, float)) or not (0.0 <= float(w) <= 1.0):
                    self._err(f"L{i}: strength {w!r} out of [0,1]")
                if ev.get("source_node") is None:
                    self._warn(f"L{i}: evidence without source_node")
            else:
                self._err(f"L{i}: unknown event type {typ!r}")

    def check_journal(self) -> None:
        jp = os.path.join(self.repo, "journal", "events.jsonl")
        if not os.path.exists(jp):
            self._warn("journal/events.jsonl missing (create on first op)")
            return
        for i, line in enumerate(open(jp), 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                self._err(f"journal L{i}: unparseable — {e}")

    def check_memory_dir(self) -> None:
        if not os.path.exists(os.path.join(self.repo, "memory", "index.json")):
            self._warn("memory/index.json missing (run `discovery rebuild`)")

    def report(self) -> dict[str, Any]:
        self.check_tree()
        self.check_ledger()
        self.check_journal()
        self.check_memory_dir()
        return {"errors": self.errors, "warnings": self.warnings,
                "ok": not self.errors}
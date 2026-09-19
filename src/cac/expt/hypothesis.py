"""Hypothesis memory bank.

Append-only ledger (`memory/hypotheses.jsonl`) + format gate + confidence
update (Eq. 1) + materialized index. The bank is the evolutionary memory: it
decides what gets built on and what is never retried.

Honesty rules enforced here (anti-gaming):
  - evidence for a hyp id that was never `create`d RAISES (phantom-id ban).
  - evidence_type is restricted to the canonical set {supports, contradicts,
    neutral}; `neutral` is logged but NOT scored (no silent confidence drift).
  - strength must be a real number in [0,1].
  - the ledger is append-only; corrections are new events, never edits.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from cac.expt.constants import (CONF_CEIL, CONF_FLOOR, CONF_INIT, CONF_REFUTED,
                                CONF_CONFIRMED, ETA, EVIDENCE_TYPES)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MARKERS = ["IF", "IN", "THEN", "BECAUSE", "DISPROVED"]
_MARKER_RE = [(m, re.compile(rf"\b{m}\b")) for m in MARKERS]
_FALSIFIER_RE = re.compile(r"DISPROVED\s+IF\s*(.+)$", re.DOTALL)
_MEASURE = re.compile(r"(<=|>=|==|<|>|\d)", re.IGNORECASE)
_MECHANISM_MIN = 20


def validate(text: str) -> tuple[list[str], list[str]]:
    """7-dim format gate (the mechanically checkable subset). Returns (errors, warnings)."""
    errors, warnings = [], []
    raw = text.strip()
    if not raw:
        return ["empty text"], []

    pos, last = [], -1
    for m, rx in _MARKER_RE:
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

    def seg(a: int, b: int) -> str:
        return raw[pos[a] + len(MARKERS[a]):pos[b]].strip(" ,.:;")

    if complete:
        for idx, name, nxt in ((0, "choice", 1), (1, "scope", 2), (2, "effect", 3), (3, "mechanism", 4)):
            s = seg(idx, nxt)
            if not s:
                errors.append(f"empty [{name}]")
            elif name == "mechanism" and len(s) < _MECHANISM_MIN:
                warnings.append(f"mechanism very short ({len(s)} chars)")

    fm = _FALSIFIER_RE.search(raw)
    falsifier = fm.group(1).strip() if fm else ""
    if "DISPROVED" in raw:
        if not falsifier:
            errors.append("empty falsification criterion")
        elif not _MEASURE.search(falsifier):
            errors.append("falsifier not measurable: needs a number/comparison (e.g. MAE>=X)")
        if re.search(r"\bOR\b", falsifier) and not re.search(r"\bAND\b", falsifier):
            warnings.append("compound OR-falsifier: ensure every disjunct alone kills the hyp")

    if len(raw) > 1200:
        warnings.append(f"hypothesis very long ({len(raw)} chars)")
    if re.search(r"\bmaybe\b|\bperhaps\b|\bmight\b", raw, re.IGNORECASE):
        warnings.append("hedging language in a falsifiable claim")
    return errors, warnings


def apply_evidence(conf: float, evidence_type: str, strength: float, eta: float = ETA) -> float:
    """Eq. 1: supports c<-c+eta*w*(1-c); contradicts c<-c-eta*w*c; neutral no-op."""
    if evidence_type == "supports":
        return conf + eta * strength * (1 - conf)
    if evidence_type == "contradicts":
        return conf - eta * strength * conf
    return conf


class Memory:
    def __init__(self, repo: str = ROOT, eta: float = ETA):
        self.repo = repo
        self.eta = eta
        self.ledger_path = os.path.join(repo, "memory", "hypotheses.jsonl")
        self.index_path = os.path.join(repo, "memory", "index.json")

    # ---- ledger ----------------------------------------------------------------
    def _append(self, ev: dict[str, Any]) -> None:
        ev = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **ev}
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        line = json.dumps(ev, ensure_ascii=False)
        with open(self.ledger_path, "a+") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            if pos:
                f.seek(pos - 1)
                if f.read(1) != "\n":
                    f.write("\n")
            f.write(line + "\n")

    def events(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.ledger_path):
            return []
        evs = []
        for i, line in enumerate(open(self.ledger_path), 1):
            if not line.strip():
                continue
            try:
                evs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"ledger line {i} unparseable: {e}") from e
        return evs

    def hyp_ids(self) -> set[str]:
        return {e["hyp_id"] for e in self.events() if e.get("type") == "create"}

    # ---- writes (append-only) --------------------------------------------------
    def create(self, hyp_id: str, text: str, source: str | None = None,
               confidence: float = CONF_INIT) -> None:
        errs, _ = validate(text)
        if errs:
            raise ValueError(f"hypothesis fails format gate: {errs}")
        if not (CONF_FLOOR <= confidence <= CONF_CEIL):
            raise ValueError(f"initial confidence must be within [{CONF_FLOOR},{CONF_CEIL}]")
        self._append({"type": "create", "hyp_id": hyp_id, "text": text,
                      "confidence": float(confidence), "source_node": source})

    def evidence(self, hyp_id: str, evidence_type: str, strength: float,
                 source_node: str, note: str | None = None) -> None:
        if evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"evidence_type {evidence_type!r} not in {EVIDENCE_TYPES}")
        w = float(strength)
        if not (0.0 <= w <= 1.0):
            raise ValueError(f"strength must be in [0,1], got {strength}")
        if hyp_id not in self.hyp_ids():
            raise ValueError(f"evidence for unknown hypothesis {hyp_id} — create it first (phantom-id ban)")
        self._append({"type": "evidence", "hyp_id": hyp_id, "evidence_type": evidence_type,
                      "strength": w, "source_node": source_node, "note": note})

    # ---- index materialization ---------------------------------------------------
    def build_index(self) -> dict[str, Any]:
        """Replay ledger -> {hyp_id: {text, confidence, status, n_tested, log}}."""
        hyp: dict[str, Any] = {}
        for ev in self.events():
            hid = ev.get("hyp_id")
            if ev.get("type") == "create":
                h = hyp.setdefault(hid, {"text": None, "confidence": CONF_INIT,
                                         "status": "uncertain", "n_tested": 0, "log": []})
                h["text"] = ev.get("text")
                h["confidence"] = float(ev.get("confidence", CONF_INIT))
                h["created"] = ev.get("ts")
            else:
                h = hyp.get(hid)
                if h is None:
                    continue  # orphan event; conformance flags it
            h["log"].append({k: ev[k] for k in
                             ("type", "ts", "evidence_type", "strength", "source_node", "note")
                             if k in ev})
            if ev.get("type") == "evidence":
                h["confidence"] = apply_evidence(h["confidence"], ev.get("evidence_type"),
                                                 float(ev.get("strength", 1.0)), self.eta)
                h["n_tested"] += 1
            if h["confidence"] > CONF_CONFIRMED:
                h["status"] = "confirmed"
            elif h["confidence"] < CONF_REFUTED:
                h["status"] = "refuted"
            else:
                h["status"] = "uncertain"
        # drop any bucket that never received a create event
        for hid in list(hyp):
            if hyp[hid]["text"] is None:
                del hyp[hid]
        idx = {"_meta": {"eta": self.eta}, "rebuilt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "hypotheses": hyp}
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        json.dump(idx, open(self.index_path, "w"), indent=2, ensure_ascii=False)
        return idx

    def summary(self) -> str:
        hyp = self.build_index()["hypotheses"]
        if not hyp:
            return "(memory bank empty)"
        lines = [f"{'id':<14}{'conf':>6}  {'status':<10} tests  text"]
        for hid in sorted(hyp):
            h = hyp[hid]
            lines.append(f"{hid:<14}{h['confidence']:>6.2f}  {h['status']:<10}{h['n_tested']:>4}   "
                         f"{str(h['text'] or '')[:72]}")
        return "\n".join(lines)
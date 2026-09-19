"""Trajectory tree as a directory tree.

The trajectory tree IS the filesystem: <repo>/tree/ mirrors the lineage of all
proposed architectures. A node's directory physically lives inside its parent
node's directory, so parent/child relationships are enforced by the filesystem
and cannot drift out of sync with a registry. All per-node state lives in the
node directory's `info.json`.

    tree/
    └── N0001_champion/
        ├── info.json            # status, best_metric, train_seconds, tested_hypotheses...
        ├── idea.md
        ├── model.py
        ├── config.toml
        ├── feedback/            # feedback/*.md
        ├── synthesis.md
        ├── result.json          # latest run result (== server result.json shape)
        └── N0002_xxx/           # child node — nested inside its parent
            └── info.json

Conventions
  - A node dir name is N<4-digit>_<short-name> (NODE_RE).
  - Every level of tree/ except the root must be a node dir or a node asset.
  - There is no separate node registry: `materialize()` walks the filesystem.
  - `info.json` is the ONLY mutable per-node record; idea/model/config are
    written once by their authoring role and never edited silently.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Iterator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TREE_DIR = os.path.join(ROOT, "tree")

NODE_RE = re.compile(r"^N\d{4}_\w[\w.\-]*$")

STATUSES = ("proposed", "coded", "running", "done", "failed", "timeout", "synthesized")
# statuses an expansion (parent-selection) candidate must be in
EXPANDABLE = ("done", "synthesized")

INFO_REQUIRED = ("id", "parent", "status", "created_ts", "tested_hypotheses")
INFO_TYPES = {
    "status": str,
    "parent": (str, type(None)),
    "best_metric": (float, int, type(None)),
    "train_seconds": (float, int, type(None)),
    "epochs": (int, type(None)),
    "tested_hypotheses": list,
}


def info_path(nid_dir: str) -> str:
    return os.path.join(nid_dir, "info.json")


def is_node_dir(path: str) -> bool:
    return NODE_RE.match(os.path.basename(path)) is not None


class TrajectoryTree:
    """Walk the filesystem under tree/ as the live topology."""

    def __init__(self, repo: str = ROOT):
        self.repo = repo
        self.tree_dir = os.path.join(repo, "tree")

    # ---- topology (filesystem = truth) --------------------------------------
    def walk_node_dirs(self) -> Iterator[str]:
        """Yield node dir paths in topological-ish (shallow-first) order."""
        if not os.path.isdir(self.tree_dir):
            return
        for dirpath, dirnames, _ in os.walk(self.tree_dir):
            dirnames[:] = [d for d in dirnames if is_node_dir(os.path.join(dirpath, d)) or d == "archive"]
            if dirpath != self.tree_dir and is_node_dir(dirpath):
                yield dirpath

    def parent_of(self, nid_dir: str) -> str | None:
        """Immediate ancestor node dir (physical walk-up), or None for a root."""
        cur = os.path.dirname(nid_dir)
        while cur.startswith(self.tree_dir) and cur != self.tree_dir:
            if is_node_dir(cur):
                return cur
            cur = os.path.dirname(cur)
        return None

    def child_dirs(self, nid_dir: str) -> list[str]:
        out = []
        if os.path.isdir(nid_dir):
            for d in sorted(os.listdir(nid_dir)):
                p = os.path.join(nid_dir, d)
                if is_node_dir(p):
                    out.append(p)
        return out

    def ancestor_dirs(self, nid_dir: str) -> list[str]:
        chain = []
        cur = self.parent_of(nid_dir)
        while cur:
            chain.append(cur)
            cur = self.parent_of(cur)
        return chain

    # ---- registry (materialized snapshot, derived from the filesystem) ------
    def _read_info(self, nid_dir: str) -> dict[str, Any]:
        ip = info_path(nid_dir)
        if os.path.exists(ip):
            return json.load(open(ip))
        return {}

    def materialize(self) -> dict[str, dict[str, Any]]:
        """{node_id: {...info.json, 'path': dir, 'parent_id': id|None}}."""
        nodes: dict[str, dict[str, Any]] = {}
        for d in self.walk_node_dirs():
            info = self._read_info(d)
            nid = info.get("id") or os.path.basename(d)
            rec = dict(info)
            rec["id"] = nid
            rec["path"] = d
            p = self.parent_of(d)
            rec["parent_id"] = os.path.basename(p) if p else None
            nodes[nid] = rec
        return nodes

    def ancestors(self, nid: str) -> list[dict[str, Any]]:
        mat = self.materialize()
        rec = mat.get(nid)
        if not rec:
            return []
        chain, cur = [], rec.get("parent_id")
        while cur and cur in mat:
            chain.append(mat[cur])
            cur = mat[cur].get("parent_id")
        return chain

    def descendants(self, nid: str) -> list[str]:
        mat = self.materialize()
        out, stack = [], [nid]
        while stack:
            cur = stack.pop()
            for nid2, r in mat.items():
                if r.get("parent_id") == cur and nid2 not in out:
                    out.append(nid2)
                    stack.append(nid2)
        return out

    def next_id(self) -> str:
        nums = []
        for d in self.walk_node_dirs():
            m = re.match(r"N(\d{4})", os.path.basename(d))
            if m:
                nums.append(int(m.group(1)))
        return f"N{max(nums, default=0) + 1:04d}_"

    # ---- lifecycle ------------------------------------------------------------
    def register(self, nid: str, parent: str | None, title: str = "") -> str:
        """Create a node dir (under parent dir if given, else under tree/). Raise if the
        id is a duplicate or the parent does not exist."""
        if not NODE_RE.match(nid):
            raise ValueError(f"node id {nid!r} must match {NODE_RE.pattern}")
        mat = self.materialize()
        if nid in mat:
            raise ValueError(f"node already registered: {nid}")
        if parent is not None and parent not in mat:
            raise ValueError(f"unknown parent node: {parent}")
        base = self.tree_dir if parent is None else mat[parent]["path"]
        d = os.path.join(base, nid)
        os.makedirs(os.path.join(d, "feedback"), exist_ok=True)
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        info = {"id": nid, "parent": parent, "status": "proposed",
                "created_ts": now, "title": title,
                "best_metric": None, "train_seconds": None, "epochs": None,
                "tested_hypotheses": [], "updated_ts": now}
        with open(info_path(d), "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return d

    def _update(self, nid: str, patch: dict[str, Any]) -> dict[str, Any]:
        mat = self.materialize()
        rec = mat.get(nid)
        if not rec:
            raise KeyError(f"node not registered: {nid}")
        ip = info_path(rec["path"])
        info = json.load(open(ip)) if os.path.exists(ip) else {}
        info.update(patch)
        info["updated_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(ip, "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return info

    def set_status(self, nid: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status {status!r} not in {STATUSES}")
        self._update(nid, {"status": status})

    def update_metrics(self, nid: str, best_metric: float | None = None,
                       train_seconds: float | None = None, epochs: int | None = None,
                       tested: list[str] | None = None) -> None:
        patch: dict[str, Any] = {}
        if best_metric is not None:
            patch["best_metric"] = float(best_metric)
        if train_seconds is not None:
            patch["train_seconds"] = float(train_seconds)
        if epochs is not None:
            patch["epochs"] = int(epochs)
        if tested:
            mat = self.materialize()
            cur = list(mat[nid].get("tested_hypotheses") or [])
            patch["tested_hypotheses"] = list(dict.fromkeys(cur + tested))
        self._update(nid, patch)

    def mark_tested(self, nid: str, tested: list[str]) -> None:
        self.update_metrics(nid, tested=tested)

    # ---- rendering ------------------------------------------------------------
    def render(self) -> str:
        """ASCII rendering of the lineage directory tree."""
        lines: list[str] = []
        for dirpath, dirnames, _ in os.walk(self.tree_dir):
            dirnames[:] = sorted(d for d in dirnames if is_node_dir(os.path.join(dirpath, d)))
            depth = dirpath[len(self.tree_dir):].count(os.sep)
            if dirpath == self.tree_dir:
                lines.append("tree/")
                continue
            if not is_node_dir(dirpath):
                continue
            info = self._read_info(dirpath)
            name = os.path.basename(dirpath)
            meta = info.get("status", "-")
            if info.get("best_metric") is not None:
                meta = f"{meta} mae={float(info['best_metric']):.2f}"
            lines.append(f"{'  ' * depth}{name}  [{meta}]")
        return "\n".join(lines)
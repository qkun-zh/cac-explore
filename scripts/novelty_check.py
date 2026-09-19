#!/usr/bin/env python3
"""Novelty check — a local stand-in for the paper's embedding-similarity gate.

Two-part judge against memory/index.json:
  1. TF-IDF cosine similarity of the hypothesis text against every prior
     hypothesis; the paper gates on embedding distance inside K_SYNTH ties.
  2. Structural judge: same (scope, effect), (mechanism), or (falsifier)
     signature as an existing hypothesis is flagged — the paper's black-box
     observer would not distinguish them either.

Exit 0 on a genuine novelty, 2 on a near-duplicate restatement.
"""
from __future__ import annotations

import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cac.expt.hypothesis import Memory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if t and len(t) > 1]


def _tfidf(texts: list[str]) -> tuple[dict[str, tuple[float, ...]], list[str]]:
    vocab: dict[str, int] = {}
    docs = [_tokens(t) for t in texts]
    n = len(docs)
    for d in docs:
        for t in set(d):
            vocab.setdefault(t, 0)
            vocab[t] += 1
    keys = sorted(vocab)
    vecs = {}
    for i, d in enumerate(docs):
        tf = {t: d.count(t) for t in d}
        v = []
        for t in keys:
            idf = math.log((n + 1) / (1 + vocab[t])) + 1.0
            v.append(tf.get(t, 0.0) * idf)
        nrm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs[i] = tuple(x / nrm for x in v)
    return vecs, keys


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _signature(text: str) -> tuple[str, str, str]:
    m = re.search(r"IF\s+(.+?)\s+IN\s+(.+?)\s+THEN\s+(.+?)\s+BECAUSE", text)
    return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip()) if m else ("", "", "")


def check(text: str, hyp_id: str | None = None, threshold: float = 0.82,
          k_synth: int = 3, verbose: bool = True) -> int:
    mem = Memory()
    idx = mem.build_index().get("hypotheses", {})
    existing = [(hid, h["text"]) for hid, h in sorted(idx.items())]
    if not existing:
        if verbose:
            print("(memory empty — nothing to compare against; treat as novel)")
        return 0
    texts = [t for _, t in existing] + [text]
    vecs, _ = _tfidf(texts)
    newvec = vecs[len(existing)]
    sims = [(hid, _cosine(newvec, vecs[i])) for i, (hid, _) in enumerate(existing)]
    sims.sort(key=lambda x: -x[1])
    t_s, m_s, e_s = _signature(text)
    sibling = None
    for hid, _ in existing:
        t2, m2, e2 = _signature(mem.build_index().get("hypotheses", {}).get(hid, {}).get("text", ""))
        if hid != hyp_id and t2 and (t_s == t2 and (m_s == m2 or e_s == e2)):
            sibling = hid
            break
    s, worst = sims[0][1], sims[0][0]
    tail = f"{s:.3f} {worst}"
    if verbose:
        print(f"top similarities: " + ", ".join(f"{hid}={v:.3f}" for hid, v in sims[:k_synth]))
        if sibling:
            print(f"structural twin of {sibling}")
    if not sibling and s < threshold:
        if verbose:
            print(f"NOVEL (top sim {s:.3f} < {threshold})")
        return 0
    if verbose:
        print(f"DUPLICATE-ish (top sim {s:.3f} >= {threshold}" + (f" AND structural twin {sibling}" if sibling else "") + ")")
    return 2


if __name__ == "__main__":
    text = sys.argv[1]
    hid = sys.argv[2] if len(sys.argv) > 2 else None
    sys.exit(check(text, hid))
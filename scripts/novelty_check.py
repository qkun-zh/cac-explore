#!/usr/bin/env python3
"""Novelty gate stage-1: TF-IDF similarity retrieval over past ideas.

Paper analog (arXiv:2604.12999 §2.3 Redundancy Filtering): retrieve top-k
most similar archived concepts; an LLM judge then decides duplicate-vs-novel
on structural principles (stage-2 lives in the Idea dispatch prompt).
This script never blocks by itself — it surfaces candidates for judgment.

Corpus: tree/nodes/*/idea.md, tree/archive*/**/idea.md, plus hypothesis
texts from memory/hypotheses.jsonl (create events).

Usage:
  python3 scripts/novelty_check.py --file tree/nodes/N0028_x/idea.md
  python3 scripts/novelty_check.py --text "proposed idea ..." [--top 3]
Exit 0 always; verdict field is advisory.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STOP = set("""a an and are as at be by for from has have how in into is it its of on or
that the their then there these they this to was were will with within without which
what when where who why can could should would may might must not no nor but if than
using use used uses new based via each per more most less least very much many also
between among during after before under over above below out up down off same other
""".split())

TOK = re.compile(r"[a-z]{3,}")


def tokens(text):
    return [t for t in TOK.findall(text.lower()) if t not in STOP]


def tfidf_matrix(docs):
    tfs, df = [], {}
    for d in docs.values():
        counts = {}
        for t in d:
            counts[t] = counts.get(t, 0) + 1
        tfs.append(counts)
        for t in counts:
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    idf = {t: math.log(n / c) + 1.0 for t, c in df.items()} if n else {}
    vecs = []
    for counts in tfs:
        total = sum(counts.values()) or 1
        v = {t: (c / total) * idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append((v, norm))
    return vecs


def cosine(a, an, b, bn):
    if len(a) > len(b):
        a, b = b, a
    dot = sum(x * b.get(t, 0.0) for t, x in a.items())
    return dot / (an * bn)


def load_corpus():
    docs = {}
    for pat in ("tree/nodes/*/idea.md", "tree/archive*/*/idea.md"):
        for p in ROOT.glob(pat):
            docs[str(p.relative_to(ROOT))] = p.read_text(errors="ignore")
    for line in (ROOT / "memory" / "hypotheses.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("type") == "create" and ev.get("text"):
            docs[f"hypothesis:{ev.get('hyp_id', '?')}"] = ev["text"]
    return docs


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--text")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    cand_text = (ROOT / args.file).read_text(errors="ignore") if args.file else args.text
    cand_path = None
    if args.file:
        p = Path(args.file)
        p = p if p.is_absolute() else ROOT / p
        try:
            cand_path = str(p.resolve().relative_to(ROOT))
        except ValueError:
            cand_path = str(p)

    corpus = load_corpus()
    if cand_path:
        corpus.pop(cand_path, None)
    if not corpus:
        print("corpus empty — nothing to compare", file=sys.stderr)
        sys.exit(0)

    docs = {k: tokens(v) for k, v in corpus.items()}
    docs["__CANDIDATE__"] = tokens(cand_text)
    vecs = tfidf_matrix(docs)
    cv, cn = vecs[-1]

    scored = []
    for (name, _), (v, norm) in zip(docs.items(), vecs):
        if name == "__CANDIDATE__":
            continue
        scored.append((cosine(cv, cn, v, norm), name))
    scored.sort(reverse=True)

    top = scored[: args.top]
    best = top[0][0] if top else 0.0
    verdict = "NOVEL" if best < 0.30 else "REVIEW" if best < 0.55 else "LIKELY-DUPLICATE"

    print(f"candidate: {cand_path or '(text)'}")
    print(f"{'sim':>6}  document")
    for s, name in top:
        print(f"{s:>6.3f}  {name}")
    print(f"\nverdict(stage-1): {verdict}  (best={best:.3f})")
    print("stage-2: LLM judge on structural principles required before registration.")


if __name__ == "__main__":
    main()

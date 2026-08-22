"""重放 memory/hypotheses.jsonl 重建 memory/index.json 快照。"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(ROOT, "memory", "hypotheses.jsonl")
IDX = os.path.join(ROOT, "memory", "index.json")

ETA = 0.20
LO, HI, INIT = 0.01, 0.99, 0.5

hyp = {}
for line in open(J):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    h = e.get("hyp_id")
    if not h:
        continue
    m = hyp.setdefault(h, {"text": "", "confidence": INIT, "n_tested": 0,
                           "status": "uncertain", "tags": [], "log": []})
    t = e.get("type")
    if t == "create":
        m["text"] = e.get("text", m["text"])
        m["tags"] = e.get("tags", m["tags"])
        if "confidence" in e:
            m["confidence"] = float(e["confidence"])
    elif t == "evidence":
        w = float(e.get("strength", 1.0))
        c = m["confidence"]
        et = e.get("evidence_type")
        if et == "supports":
            c = c + ETA * w * (1 - c)
        elif et == "contradicts":
            c = c - ETA * w * c
        m["confidence"] = round(min(HI, max(LO, c)), 4)
        if et in ("supports", "contradicts"):
            m["n_tested"] += 1
        m["log"].append({"source_node": e.get("source_node"), "evidence_type": et,
                         "strength": w, "reasoning": e.get("reasoning", "")[:200]})
    elif t == "revise":
        m["text"] = e.get("text", m["text"])
    m["status"] = ("confirmed" if m["confidence"] > 0.75
                   else "refuted" if m["confidence"] < 0.25 else "uncertain")

meta = json.load(open(IDX)).get("_meta", {})
json.dump({"_meta": meta, "hypotheses": hyp}, open(IDX, "w"), indent=2, ensure_ascii=False)
print(f"[rebuild_index] {len(hyp)} hypotheses -> {IDX}")

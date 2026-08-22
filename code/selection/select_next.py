"""code/selection/select_next.py — HypoExplore 双重选择。

用法:
  python code/selection/select_next.py parent                 # 选下一个要扩展的父节点
  python code/selection/select_next.py hypo --parent N0001_x  # 为该父节点选假设组 Q_t
"""
import argparse
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TREE = os.path.join(ROOT, "tree", "tree.json")
MEM = os.path.join(ROOT, "memory", "index.json")


def load(path):
    return json.load(open(path))


def ancestors(tree, nid):
    chain, cur = [], nid
    while cur:
        n = tree["nodes"].get(cur)
        if not n:
            break
        chain.append(n)
        cur = n.get("parent")
    return chain


def select_parent(tree):
    meta = tree["_meta"]
    lam_acc, lam_par = meta.get("lambda_acc", 0.85), meta.get("lambda_parent", 0.60)
    tau_max = float(meta.get("tau_max_min", 30)) * 60
    idx = load(MEM)["hypotheses"]
    nodes = {k: v for k, v in tree["nodes"].items() if v.get("status") in ("synthesized", "done")}
    if not nodes:
        print("无可扩展节点（需要 status ∈ synthesized|done）")
        return None
    accs = [v.get("best_metric", 0.0) or 0.0 for v in nodes.values()]
    lo, hi = min(accs), max(accs)
    rows = []
    for nid, n in nodes.items():
        acc_norm = (n.get("best_metric", 0.0) - lo) / (hi - lo) if hi > lo else 1.0
        tau = min(float(n.get("train_seconds", 0) or 0), tau_max)
        quality = lam_acc * acc_norm + (1 - lam_acc) * (1 - tau / tau_max)
        tested = set()
        for a in ancestors(tree, nid):
            tested.update(a.get("tested_hypotheses", []))
        # 论文定义：H_active = 全部 uncertain 假设；H_tested = 其中已在本支线祖先链测过的
        active = [h for h, m in idx.items() if m.get("status", "uncertain") == "uncertain"]
        n_tested = len([h for h in active if h in tested])
        avail = 0.0 if not active else 1 - n_tested / len(active)
        score = lam_par * quality + (1 - lam_par) * avail
        rows.append((score, quality, avail, acc_norm, nid))
    rows.sort(reverse=True)
    print(f"{'node':<28}{'score':>8}{'qual':>8}{'avail':>8}{'acc_n':>8}")
    for score, q, av, an, nid in rows:
        print(f"{nid:<28}{score:>8.3f}{q:>8.3f}{av:>8.3f}{an:>8.3f}")
    best = rows[0][4]
    print(f"\nPARENT -> {best}")
    return best


def select_hypo(parent, seed=None):
    rng = random.Random(seed)
    tree = load(TREE)
    idx = load(MEM)["hypotheses"]
    tested = set()
    for a in ancestors(tree, parent):
        tested.update(a.get("tested_hypotheses", []))
    cand = [(h, m) for h, m in idx.items()
            if m.get("status", "uncertain") == "uncertain" and h not in tested]
    if not cand:
        print(f"{parent} 无可用假设（全部已测或已定论）。建议 Idea Agent 提出新假设。")
        return []
    scored = []
    for h, m in cand:
        sup = sum(e.get("strength", 1.0) for e in m.get("log", []) if e.get("evidence_type") == "supports")
        con = sum(e.get("strength", 1.0) for e in m.get("log", []) if e.get("evidence_type") == "contradicts")
        theta = rng.betavariate(1 + sup, 1 + con)
        c = float(m.get("confidence", 0.5))
        epistemic = 1 - abs(2 * c - 1)
        scored.append((h, c, sup, con, theta, epistemic, m.get("text", "")[:80]))
    K = 2
    exploit = sorted(scored, key=lambda r: -r[4])[:K]
    explore = sorted(scored, key=lambda r: -r[5])[:K]
    seen, qt = set(), []
    print(f"{'hyp_id':<12}{'conf':>6}{'sup':>5}{'con':>5}   text")
    for r in exploit + explore:
        if r[0] not in seen:
            seen.add(r[0]); qt.append(r[0])
            print(f"{r[0]:<12}{r[1]:>6.2f}{r[2]:>5.1f}{r[3]:>5.1f}   {r[6]}")
    print(f"\nQ_t (parent={parent}) -> {qt}")
    return qt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["parent", "hypo"])
    p.add_argument("--parent", default=None)
    a = p.parse_args()
    if a.mode == "parent":
        select_parent(load(TREE))
    else:
        assert a.parent, "--parent 必填"
        select_hypo(a.parent)

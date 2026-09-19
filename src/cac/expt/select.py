"""Dual selection — the paper's decision rules, operating on the directory tree.

Parent selection (Eq. 2-4):
    quality(n) = λ_acc·Acc_norm + (1−λ_acc)·(1 − min(τ(n),τ_max)/τ_max)
    avail(n)   = 1 − |tested_on_ancestry ∩ active| / |active|
    score(n)   = λ_parent·quality(n) + (1−λ_parent)·avail(n)
  Acc_norm normalizes best_metric (MAE, lower-is-better) to [0,1].
  `active` = uncertain hypotheses; `expendable` candidates = status done|synthesized.
  Selection is deterministic (pick the highest score).

Hypothesis selection (Eq. 5-6):
    exploit: θ_h ~ Beta(α₀+Σw_supports, β₀+Σw_contradicts) — top K_HYPO
    explore: epistemic(h) = 1 − |2c−1|                              — top K_HYPO
    Q_t = dedup(exploit ∪ explore), |Q_t| ≤ 2·K_HYPO, over uncertain &
    not-yet-tested-on-this-ancestry hypotheses.
"""
from __future__ import annotations

import random
from typing import Any

from cac.expt.constants import (BETA_ALPHA0, BETA_BETA0, CONF_INIT, K_HYPO,
                                LAMBDA_ACC, LAMBDA_PARENT, TAU_MAX_SECONDS)
from cac.expt.mechanisms import conflicts, feasible, requires


def _acc_norm(best_metric: float | None, lo: float, hi: float) -> float:
    if best_metric is None:
        return 0.0
    if hi > lo:
        return (hi - best_metric) / (hi - lo)  # MAE: lower is better
    return 1.0


def select_parent(tree: "TrajectoryTree", index: dict[str, Any],
                  lambda_acc: float = LAMBDA_ACC, lambda_parent: float = LAMBDA_PARENT,
                  tau_max: float = TAU_MAX_SECONDS, verbose: bool = True) -> str | None:
    mat = tree.materialize()
    hyp = index.get("hypotheses", index)
    cand = {nid: r for nid, r in mat.items()
            if r.get("status") in ("done", "synthesized")}
    if not cand:
        print("no expandable node (status done|synthesized) — bootstrap or complete feedback first")
        return None

    accs = [float(r["best_metric"]) for r in cand.values() if r.get("best_metric") is not None]
    lo, hi = (min(accs), max(accs)) if accs else (0.0, 0.0)
    active = {h for h, m in hyp.items() if m.get("status", "uncertain") == "uncertain"}

    rows: list[tuple[float, float, float, float, str]] = []
    for nid, r in cand.items():
        acc_norm = _acc_norm(r.get("best_metric"), lo, hi)
        tau = min(float(r.get("train_seconds") or 0), tau_max)
        quality = lambda_acc * acc_norm + (1 - lambda_acc) * (1 - tau / tau_max)
        tested = set(r.get("tested_hypotheses") or [])
        for a in tree.ancestors(nid):
            tested.update(a.get("tested_hypotheses") or [])
        n_tested = len(active & tested)
        avail = 0.0 if not active else 1 - n_tested / len(active)
        score = lambda_parent * quality + (1 - lambda_parent) * avail
        rows.append((score, quality, avail, acc_norm, nid))

    rows.sort(key=lambda x: -x[0])
    if verbose:
        print(f"{'node':<24}{'score':>7}{'qual':>7}{'avail':>7}{'acc_n':>7}")
        for s, q, av, an, nid in rows:
            print(f"{nid:<24}{s:>7.3f}{q:>7.3f}{av:>7.3f}{an:>7.3f}")
    best = rows[0][4]
    if verbose:
        print(f"\nPARENT -> {best}")
    return best


def _parent_switches(mat: dict[str, Any], parent: str) -> frozenset:
    """use_* switches actually turned on in the parent's config.toml (flat TOML)."""
    import os
    import tomllib
    cfgp = os.path.join(mat[parent].get("path", ""), "config.toml")
    if not os.path.exists(cfgp):
        return frozenset()
    try:
        cfg = tomllib.load(open(cfgp, "rb"))
        return frozenset(k for k, v in cfg.items() if k.startswith("use_") and bool(v))
    except Exception:
        return frozenset()


def select_hypo(parent: str, index: dict[str, Any], seed: int | None = None,
                k_hypo: int = K_HYPO, verbose: bool = True,
                tree: "TrajectoryTree" | None = None,
                mandated: list[str] | None = None,
                max_adjoin: int = 1) -> list[str]:
    if tree is None:
        tree = __import__("cac.expt.node", fromlist=["TrajectoryTree"]).TrajectoryTree()
    hyp = index.get("hypotheses", index)
    mat = tree.materialize()
    if parent not in mat:
        raise KeyError(f"node not registered: {parent}")
    tested = set(mat[parent].get("tested_hypotheses") or [])
    for a in tree.ancestors(parent):
        tested.update(a.get("tested_hypotheses") or [])
    cand = {h: m for h, m in hyp.items()
            if m.get("status", "uncertain") == "uncertain" and h not in tested}

    # mandated hypotheses are pre-registered (--new) or explicitly booked (--book):
    # they are forced into the child regardless of the sample path. Validate them.
    mandated = [m for m in (mandated or []) if m and m in hyp]
    unknown = [m for m in (mandated or []) if m not in hyp]
    if unknown:
        raise KeyError(f"mandated hypothesis(es) not in index: {unknown}")

    if not cand and not mandated:
        print("(no uncertain untested-on-ancestry hypotheses)")
        return []

    rng = random.Random(seed)
    scored: list[tuple[str, float, float, Any]] = []
    for h, m in cand.items():
        sup = sum(float(e.get("strength", 1.0)) for e in m.get("log", [])
                  if e.get("evidence_type") == "supports")
        con = sum(float(e.get("strength", 1.0)) for e in m.get("log", [])
                  if e.get("evidence_type") in ("contradicts", "neutral"))
        theta = rng.betavariate(BETA_ALPHA0 + sup, BETA_BETA0 + con)
        c = float(m.get("confidence", CONF_INIT))
        scored.append((h, theta, 1 - abs(2 * c - 1), m))

    exploit = sorted(scored, key=lambda r: -r[1])[:k_hypo]
    explore = sorted(scored, key=lambda r: -r[2])[:k_hypo]
    seen: set[str] = set()
    qt: list[str] = []
    for h, *_ in exploit + explore:
        if h not in seen and h not in (mandated or []):
            seen.add(h)
            qt.append(h)

    # Composition feasibility: when a new hypothesis is explicitly booked, keep
    # at most ONE compatible adjoin (cap) and never co-compose hypotheses that
    # conflict on the same component/switch (registry).
    parent_on: frozenset = _parent_switches(mat, parent)
    if mandated:
        qt_paper = qt[:]
        qt = list(mandated)
        while qt_paper and len(qt) < len(mandated) + max_adjoin:
            h = qt_paper.pop(0)
            if any(conflicts(h, m) for m in qt):
                if verbose:
                    print(f"(drop {h} — conflicts with mandated composition)")
                continue
            if requires(h) and not (requires(h) <= parent_on):
                if verbose:
                    print(f"(drop {h} — parent lacks required switch(es) {sorted(requires(h) - parent_on)})")
                continue
            qt.append(h)
            break
    else:
        # paper's own autonomous selection: still refuse adjoins whose required
        # parent switches are absent (H0011 on a gate-less parent is infeasible).
        qt = [h for h in qt
              if not (requires(h) and not (requires(h) <= parent_on))]

    if verbose:
        print(f"{'hyp':<14}{'theta':>7}{'epist':>7}  text")
        for h, th, ep, m in sorted(scored, key=lambda r: -r[1]):
            print(f"{h:<14}{th:>7.3f}{ep:>7.3f}  {str(m.get('text', ''))[:76]}")
        print(f"\nQ_t (parent={parent})" + (" mandated=" + ",".join(mandated) if mandated else "") + f" -> {qt}")
    # Feasibility filter over the paper's own set too: drop conflicting pairs.
    qt = feasible(qt)
    return qt
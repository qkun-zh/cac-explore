"""Expt core tests — the paper's mechanics, no torch required.

Run: python -m pytest tests -q   (or)   python tests/test_all.py
"""
import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cac.expt.constants import CONF_CEIL, CONF_FLOOR, ETA  # noqa: E402
from cac.expt.gates import Calibration, Conformance  # noqa: E402
from cac.expt.hypothesis import Memory, apply_evidence, validate  # noqa: E402
from cac.expt.node import TrajectoryTree  # noqa: E402
from cac.expt.select import select_hypo, select_parent  # noqa: E402

GOOD = ("IF we add a coarse multi-scale exemplar summary IN the ExemplarEncoder "
        "THEN val MAE improves by at least 0.30 BECAUSE fine and coarse views "
        "together resolve scale-invariant exemplar appearance, helping the "
        "condenser separate objects from backgrounds. DISPROVED IF val MAE does "
        "not improve by at least 0.30")


def make_repo(root: str) -> Path:
    for d in ("tree", "memory", "journal", "src"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    t = TrajectoryTree(root)
    t.register("N0001_root", None, title="test root")
    t.update_metrics("N0001_root", best_metric=1.5, train_seconds=100, epochs=3,
                     tested=["H0001", "H0002"])
    t.set_status("N0001_root", "done")
    t.register("N0002_child", "N0001_root", title="test child")
    return t


def test_node_physical_nesting(tmp_path):
    make_repo(str(tmp_path))
    t = TrajectoryTree(str(tmp_path))
    mat = t.materialize()
    assert mat["N0001_root"]["parent_id"] is None
    assert mat["N0002_child"]["parent_id"] == "N0001_root"
    assert os.path.dirname(mat["N0002_child"]["path"]) == mat["N0001_root"]["path"]
    assert t.parent_of(mat["N0002_child"]["path"]) == mat["N0001_root"]["path"]
    assert [r["id"] for r in t.ancestors("N0002_child")] == ["N0001_root"]
    assert t.next_id().startswith("N0003_")


def test_node_duplicate_and_missing_parent(tmp_path):
    t = make_repo(str(tmp_path))
    try:
        t.register("N0001_root", None)
        assert False, "duplicate accepted"
    except ValueError:
        pass
    try:
        t.register("N0003_x", "N0999_nope")
        assert False, "missing parent accepted"
    except ValueError:
        pass


def test_node_expandable_metrics(tmp_path):
    t = make_repo(str(tmp_path))
    mat = t.materialize()
    assert mat["N0001_root"]["status"] == "done"
    assert mat["N0001_root"]["best_metric"] == 1.5
    assert list(mat["N0002_child"]["tested_hypotheses"]) == []


def test_validate_gate():
    errs, _ = validate(GOOD)
    assert not errs
    bad = "maybe the model improves"
    errs, _ = validate(bad)
    assert errs
    no_falsifier = ("IF we wiggle IN the head THEN MAE drops BECAUSE wiggling "
                    "shuffles the optimization landscape of the decoder enough "
                    "to generalize. " * 1)
    errs, _ = validate(no_falsifier)
    assert any("DISPROVED" in e for e in errs)


def test_apply_evidence_math():
    c = apply_evidence(0.5, "supports", 1.0, ETA)
    assert abs(c - 0.6) < 1e-9
    c = apply_evidence(0.5, "contradicts", 1.0, ETA)
    assert abs(c - 0.4) < 1e-9
    c = apply_evidence(0.5, "neutral", 1.0, ETA)
    assert abs(c - 0.5) < 1e-9
    c = apply_evidence(0.5, "supports", 0.8, ETA)
    assert abs(c - 0.58) < 1e-9


def seed_ledger(root: str) -> Memory:
    m = Memory(root)
    for i in range(3):
        m.create(f"H{i + 1:04d}", GOOD, source="N0001_root")
    m.evidence("H0001", "supports", 1.0, "N0001_root")
    m.evidence("H0001", "supports", 1.0, "N0001_root")  # decisive (c0=0.6)
    m.evidence("H0002", "contradicts", 0.5, "N0001_root")
    m.evidence("H0003", "neutral", 1.0, "N0001_root")
    return m


def test_ledger_append_and_math(tmp_path):
    m = seed_ledger(str(tmp_path))
    idx = m.build_index()["hypotheses"]
    assert abs(idx["H0001"]["confidence"] - 0.68) < 1e-9  # 0.5→0.6→0.68
    assert abs(idx["H0002"]["confidence"] - (0.5 - ETA * 0.5 * 0.5)) < 1e-9
    assert abs(idx["H0003"]["confidence"] - 0.5) < 1e-9  # neutral no-op
    assert idx["H0001"]["status"] == "uncertain"


def test_phantom_id_ban(tmp_path):
    m = Memory(str(tmp_path))
    try:
        m.evidence("H9999", "supports", 1.0, "N0001_root")
        assert False, "phantom evidence accepted"
    except ValueError:
        pass


def test_evidence_bounds_and_type(tmp_path):
    m = seed_ledger(str(tmp_path))
    for bad_type, bad_strength in (("notatype", 0.5), ("supports", 9.0), ("contradicts", -1.0)):
        try:
            m.evidence("H0001", bad_type, bad_strength, "N0001_root")
            assert False, f"{bad_type}/{bad_strength} accepted"
        except ValueError:
            pass


def test_conformance_green(tmp_path):
    m = seed_ledger(str(tmp_path))
    t = make_repo(str(tmp_path))
    t.materialize()
    report = Conformance(str(tmp_path)).report()
    assert not report["errors"], report["errors"]


def test_conformance_catches_drift(tmp_path):
    seed_ledger(str(tmp_path))
    t = make_repo(str(tmp_path))
    info_p = os.path.join(t.materialize()["N0002_child"]["path"], "info.json")
    info = json.load(open(info_p))
    info["parent"] = "N0001_root"  # consistent; now corrupt status
    info["status"] = "banana"
    json.dump(info, open(info_p, "w"))
    errors = Conformance(str(tmp_path)).report()["errors"]
    assert any("status" in e for e in errors)


def test_calibration_replay(tmp_path):
    m = seed_ledger(str(tmp_path))
    idx = m.build_index()
    rep = Calibration(str(tmp_path)).report(as_json=True)
    assert rep["tests"], rep
    hits = [t for t in rep["tests"] if t["hit"] is True]
    assert len(hits) == 2  # the two supports
    assert "reliability_error" in rep and "direction" in rep
    d = rep["direction"]
    assert d["decisive"] == 1 and d["correct"] == 1 and d["accuracy"] == 1.0
    for b in rep["bins"].values():
        assert set(b) == {"n", "confirm", "rate", "mean_conf"}
        if b["n"]:
            assert 0.0 <= b["rate"] <= 1.0


def test_select_parent_deterministic(tmp_path):
    seed_ledger(str(tmp_path))
    t = make_repo(str(tmp_path))
    idx = Memory(str(tmp_path)).build_index()
    a = select_parent(t, idx, verbose=False)
    b = select_parent(t, idx, verbose=False)
    assert a == b == "N0001_root"


def test_select_hypo_filters_tested(tmp_path):
    seed_ledger(str(tmp_path))
    t = make_repo(str(tmp_path))
    idx = Memory(str(tmp_path)).build_index()
    q = select_hypo("N0001_root", idx, seed=1, verbose=False, tree=t)
    assert isinstance(q, list)
    assert "H0001" in q or "H0002" in q or "H0003" in q


def test_select_hypo_mandated_caps_adjoin(tmp_path):
    m = seed_ledger(str(tmp_path))
    t = make_repo(str(tmp_path))
    idx = m.build_index()
    q = select_hypo("N0001_root", idx, seed=1, verbose=False, tree=t,
                    mandated=["H0001"])
    assert q[0] == "H0001", q
    assert len(q) <= 2, q


def test_select_hypo_feasibility_drops_conflict(tmp_path):
    # H0009 (content-conditioned channel gate) and H0011 (frozen-random input
    # to the SAME gate) conflict on the channel_gate component. Booking H0009
    # must NOT admit H0011 as an adjoin.
    m = Memory(str(tmp_path))
    for i in range(4):
        m.create(f"H{i:04d}", GOOD, source="N0001_root")
    m.create("H0009", GOOD, source="N0001_root")
    m.create("H0011", GOOD, source="N0001_root")
    t = make_repo(str(tmp_path))
    idx = m.build_index()
    q = select_hypo("N0001_root", idx, seed=1, verbose=False, tree=t,
                    mandated=["H0009"])
    assert "H0009" in q, q
    assert "H0011" not in q, f"conflicting adjoin admitted: {q}"
    assert len(q) <= 2, q


def _pytest_runner():
    qs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    import inspect
    import traceback
    passed = 0
    for q in qs:
        try:
            if len(inspect.signature(q).parameters):
                with tempfile.TemporaryDirectory() as d:
                    q(d)
            else:
                q()
            passed += 1
            print(f"ok   {q.__name__}")
        except Exception:
            print(f"FAIL {q.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(qs)} passed")
    return 0 if passed == len(qs) else 1


if __name__ == "__main__":
    sys.exit(_pytest_runner())
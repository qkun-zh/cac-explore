"""Experiment orchestration — the evolutionary machinery.

Single definitions of the discovery algorithm live in this package:
  TrajectoryTree  — the trajectory tree AS a directory tree (filesystem lineage)
  Memory          — hypothesis ledger (append-only) + format gate + Eq.1 confidence
  select_parent   — Eq. 2-4 parent selection (deterministic)
  select_hypo     — Eq. 5-6 dual hypothesis selection (Thompson exploit + epistemic explore)
  Calibration     — hypothesis-prediction calibration monitor
  Conformance     — anti-drift / anti-gaming guard (runs before every commit)

A single `tree/` (the directory tree) + a single `memory/` ledger live at the
repo root and are shared by all nodes.
"""
from __future__ import annotations

from cac.expt.constants import *  # noqa: F401,F403
from cac.expt.gates import Calibration, Conformance  # noqa: F401
from cac.expt.hypothesis import Memory, apply_evidence, validate  # noqa: F401
from cac.expt.node import TrajectoryTree  # noqa: F401
from cac.expt.select import select_hypo, select_parent  # noqa: F401
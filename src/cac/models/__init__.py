"""Model registry.

`build_model(cfg)` is the stable contract: given a flat config dict, returns an
nn.Module whose `forward(imgs, bboxes[, bboxes3])` returns
{"density": ..., optional "n_aux"}. The canonical champion lives in counter.py
and is what every evolved node builds on via its own node-local model.py.
"""
from __future__ import annotations

from typing import Any

from cac.models.counter import Counter, build_model  # noqa: F401

_REGISTRY = {"baseline": build_model}


def make_model(cfg: dict[str, Any]):
    name = cfg.get("model", "baseline")
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registry has {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)
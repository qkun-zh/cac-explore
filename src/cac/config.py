"""Config loading: read a TOML file into a flat dotted-key dict.

Interface: load_config(path: str) -> dict
  - Reads a TOML file (single source of truth).
  - Nested tables are flattened to dotted keys so consumers rely on flat
    `cfg.get(...)` only.
  - Values are returned as-native with no coercion.
Raises FileNotFoundError if the file is missing.
"""
from __future__ import annotations

import os
import tomllib
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    flat: dict[str, Any] = {}

    def _flood(prefix: str, node: dict[str, Any]) -> None:
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flood(key, v)
            else:
                flat[key] = v

    _flood("", raw)
    return flat
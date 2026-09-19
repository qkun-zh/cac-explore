"""HF hub helpers: env setup + optional token.

HF_HOME/HF_ENDPOINT are set for the server (persistent cache at /data/asset/hf,
mirror at hf-mirror.com). `hf_token` reads a token file if present, else None.
"""
from __future__ import annotations

import os
from pathlib import Path


def setup_hf_env() -> None:
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/data/asset/hf")
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def hf_token() -> str | None:
    for p in ("/tmp/hf_token.txt", str(Path.home() / ".cache/huggingface/token")):
        if os.path.exists(p):
            return open(p).read().strip()
    return None
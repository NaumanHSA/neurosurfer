"""Shared env-loading helpers used by every config section."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency on python-dotenv).

    Only sets keys that are not already present in the environment, so real env
    vars always win over the file.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

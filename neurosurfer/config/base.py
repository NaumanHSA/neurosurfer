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
        value = value.strip()
        # Quoted value: take it verbatim (a '#' inside quotes is data, not a comment).
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        else:
            # Unquoted: strip a trailing inline comment (" # ...") like standard dotenv,
            # so `KEY=val   # note` yields `val`, not `val   # note`.
            hash_idx = value.find(" #")
            if hash_idx != -1:
                value = value[:hash_idx].rstrip()
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

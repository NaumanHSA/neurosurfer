"""Sandbox environment sanitisation.

Strips sensitive env vars (API keys, secrets, tokens) from the child process
environment and overrides HOME / TMPDIR to point at the sandbox directory so
that well-behaved libraries (matplotlib, tempfile, requests cache, etc.) write
there by default rather than the user's home directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Keys that are safe to forward as-is.
_SAFE_KEYS = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "USER",
        "LOGNAME",
        "SHELL",
    }
)

# Patterns that indicate a sensitive env var — matched against the key name.
_SENSITIVE: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"api[_-]?key",
        r"secret",
        r"token",
        r"password",
        r"passwd",
        r"credential",
        r"private[_-]?key",
        r"auth",
        r"anthropic",
        r"openai",
        r"aws[_-]",
        r"azure[_-]",
        r"gcp[_-]",
        r"stripe",
        r"twilio",
        r"database[_-]?url",
        r"db[_-]?pass",
        r"smtp",
        r"sendgrid",
    ]
)


def build_sandbox_env(sandbox: Path) -> dict[str, str]:
    """Return a minimal, sanitised ``os.environ``-style dict for the child process.

    Only allow-listed or clearly harmless keys are forwarded; everything that
    matches a sensitive pattern is dropped.  HOME and TMPDIR are overridden to
    the sandbox directory.
    """
    env: dict[str, str] = {}

    for key, value in os.environ.items():
        if key in _SAFE_KEYS:
            env[key] = value
            continue
        if any(pat.search(key) for pat in _SENSITIVE):
            continue
        # Forward other keys that don't match any sensitive pattern
        # (e.g. DISPLAY, XDG_* on Linux, PATH extensions, etc.)
        env[key] = value

    # Pin filesystem-adjacent dirs to the sandbox
    sandbox_str = str(sandbox)
    env["HOME"] = sandbox_str
    env["TMPDIR"] = sandbox_str
    env["TEMP"] = sandbox_str
    env["TMP"] = sandbox_str

    # Deterministic, debug-friendly Python behaviour
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"

    return env

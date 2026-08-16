"""Module-level constants for web_search, overridable via environment variables."""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


MAX_RESULTS: int = _env_int("WEB_SEARCH_MAX_RESULTS", 5)
FETCH_TOP_K: int = _env_int("WEB_SEARCH_FETCH_TOP_K", 3)
BUDGET_TOKENS: int = _env_int("WEB_SEARCH_BUDGET_TOKENS", 3000)
CHUNK_TOKENS: int = _env_int("WEB_SEARCH_CHUNK_TOKENS", 200)
FETCH_TIMEOUT: float = _env_float("WEB_SEARCH_TIMEOUT", 10.0)
DEFAULT_ENGINE: str = os.environ.get("WEB_SEARCH_ENGINE", "ddg").lower()

MAX_PAGE_BYTES: int = 2_000_000
_UA: str = (
    "Mozilla/5.0 (compatible; neurosurfer/0.2; +https://github.com/NaumanHSA/neurosurfer)"
)

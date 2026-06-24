"""Pluggable search engine backends."""

from __future__ import annotations

from .base import BaseEngine, EngineResult
from .ddg import DuckDuckGoEngine
from .serpapi import SerpApiEngine

__all__ = [
    "BaseEngine",
    "EngineResult",
    "DuckDuckGoEngine",
    "SerpApiEngine",
    "get_engine",
]


def get_engine(name: str) -> BaseEngine:
    """Instantiate a search engine by name.

    Raises :class:`ValueError` for unknown names so callers get a clear message.
    """
    if name == "ddg":
        return DuckDuckGoEngine()
    if name == "serpapi":
        return SerpApiEngine()
    raise ValueError(
        f"Unknown search engine {name!r}. Available: 'ddg' (DuckDuckGo), 'serpapi' (SerpAPI/Google)."
    )

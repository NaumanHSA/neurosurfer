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


def get_engine(name: str, api_key: str | None = None) -> BaseEngine:
    """Instantiate a search engine by name.

    *api_key* is passed through for engines that need one; each falls back to its
    own environment variable when it is None, so an engine that has always been
    configured through the environment keeps working untouched.

    Raises :class:`ValueError` for unknown names so callers get a clear message.
    """
    if name == "ddg":
        return DuckDuckGoEngine()
    if name == "serpapi":
        return SerpApiEngine(api_key=api_key)
    raise ValueError(
        f"Unknown search engine {name!r}. Available: 'ddg' (DuckDuckGo), 'serpapi' (SerpAPI/Google)."
    )

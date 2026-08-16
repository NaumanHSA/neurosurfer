"""Web search with pluggable engines and query-aware, budget-capped result injection.

Engine selection (per-call or via env var):
    ddg      — DuckDuckGo (free, no key required)
    serpapi  — Google via SerpAPI (requires SERPAPI_API_KEY env var)

The active engine defaults to the WEB_SEARCH_ENGINE env var (default: 'ddg').
Override per-call via ``WebSearchArgs.engine``.

Backward-compatible public API (all names importable from this package directly):
    WebSearchTool, WebSearchArgs          — the tool class and arg model
    extract_body, _normalize_text         — HTML extraction helpers (used by browse.py)
    chunk_text                            — paragraph-boundary chunker
    rank_chunks, select_within_budget     — BM25 ranking (from agents.lexical)
    BaseEngine, EngineResult              — engine ABC and result dataclass
    DuckDuckGoEngine, SerpApiEngine       — concrete engine classes
    get_engine                            — factory: get_engine('ddg' | 'serpapi')
    config                                — module exposed for test patching:
                                           ``ws.config.BUDGET_TOKENS = N``
"""

from __future__ import annotations

from . import config  # expose for test patching: monkeypatch.setattr(ws.config, "BUDGET_TOKENS", N)
from .config import (
    _UA,
    BUDGET_TOKENS,
    CHUNK_TOKENS,
    DEFAULT_ENGINE,
    FETCH_TIMEOUT,
    FETCH_TOP_K,
    MAX_PAGE_BYTES,
    MAX_RESULTS,
)
from .engines import BaseEngine, DuckDuckGoEngine, EngineResult, SerpApiEngine, get_engine
from .extractor import _fetch_page, _normalize_text, _store_full_text, extract_body
from .ranker import chunk_text, rank_chunks, select_within_budget
from .tool import WebSearchArgs, WebSearchTool

__all__ = [
    # Tool
    "WebSearchTool",
    "WebSearchArgs",
    # Extraction (browse.py imports extract_body from here)
    "extract_body",
    "_normalize_text",
    "_fetch_page",
    "_store_full_text",
    # Ranking
    "chunk_text",
    "rank_chunks",
    "select_within_budget",
    # Engine abstraction
    "BaseEngine",
    "EngineResult",
    "DuckDuckGoEngine",
    "SerpApiEngine",
    "get_engine",
    # Config (re-exported for backward compat; patch via ws.config.BUDGET_TOKENS)
    "config",
    "MAX_RESULTS",
    "FETCH_TOP_K",
    "BUDGET_TOKENS",
    "CHUNK_TOKENS",
    "FETCH_TIMEOUT",
    "DEFAULT_ENGINE",
    "MAX_PAGE_BYTES",
    "_UA",
]

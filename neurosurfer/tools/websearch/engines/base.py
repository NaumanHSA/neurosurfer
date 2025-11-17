from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple



@dataclass
class EngineResult:
    """
    One normalized result item from a search engine.

    Engines should populate at least:
      - title
      - url
      - snippet

    Optional:
      - score (relevance, rank, etc.)
    """
    title: str
    url: str
    snippet: str
    score: Optional[float] = None


@dataclass
class EngineSearchMeta:
    """
    Metadata about a search engine call.

    total_results:
        Estimated total results (if available).
    provider:
        Name/identifier of the provider (e.g. "serpapi", "tavily").
    extra:
        Engine-specific metadata.
    """
    total_results: Optional[int] = None
    provider: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class SearchEngine(ABC):
    """
    Abstract base class for web search engines.

    Concrete engines (SerpApiEngine, TavilyEngine, etc.) must implement
    the `search()` method and expose a `name` attribute.
    """

    name: str

    @abstractmethod
    def search(
        self,
        *,
        query: str,
        hl: str,
        max_results: int,
        location: Optional[str],
        gl: str,
    ) -> Tuple[List[EngineResult], EngineSearchMeta, Dict[str, Any]]:
        """
        Execute a search and return:

            (results, meta, raw)

        where:
          - results: list[EngineResult]
          - meta: EngineSearchMeta
          - raw: engine-specific raw payload (dict or any JSON-like object)
        """
        raise NotImplementedError



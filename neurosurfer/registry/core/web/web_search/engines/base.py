"""Abstract search engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EngineResult:
    title: str
    url: str
    snippet: str


class BaseEngine(ABC):
    """Pluggable web search backend.

    Subclasses implement :meth:`search`; optionally override :meth:`is_available`
    to signal that required dependencies or credentials are missing.
    """

    name: str

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[EngineResult]:
        """Return up to ``max_results`` results for ``query``."""

    def is_available(self) -> bool:
        """Return True if the engine's runtime dependencies / credentials are present."""
        return True

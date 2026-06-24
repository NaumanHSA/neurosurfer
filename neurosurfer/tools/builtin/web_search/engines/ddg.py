"""DuckDuckGo search engine (free, no API key required)."""

from __future__ import annotations

from .base import BaseEngine, EngineResult


def _import_ddgs():
    """Return the DDGS class from whichever package is installed, else None."""
    try:
        from ddgs import DDGS  # type: ignore

        return DDGS
    except Exception:  # noqa: BLE001
        try:
            from duckduckgo_search import DDGS  # type: ignore

            return DDGS
        except Exception:  # noqa: BLE001
            return None


class DuckDuckGoEngine(BaseEngine):
    """DuckDuckGo text search via the ``ddgs`` or ``duckduckgo_search`` package."""

    name = "ddg"

    def search(self, query: str, max_results: int) -> list[EngineResult]:
        DDGS = _import_ddgs()
        if DDGS is None:
            raise RuntimeError(
                "Web search needs the 'ddgs' package. "
                "Install with: pip install 'neurosurfer[search]'"
            )
        results: list[EngineResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url") or ""
                if not url:
                    continue
                results.append(
                    EngineResult(
                        title=(r.get("title") or url).strip(),
                        url=url.strip(),
                        snippet=(r.get("body") or r.get("description") or "").strip(),
                    )
                )
        return results

    def is_available(self) -> bool:
        return _import_ddgs() is not None

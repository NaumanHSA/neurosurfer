from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None  # will be checked in __init__

from .base import EngineResult, EngineSearchMeta, SearchEngine


class SerpApiEngine(SearchEngine):
    """
    SerpAPI-backed implementation of SearchEngine (Google engine).
    """
    name = "serpapi"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        api_key:
            SerpAPI key. If None, uses SERPAPI_API_KEY from the environment.
        endpoint:
            Optional override for SerpAPI endpoint, defaults to
            'https://serpapi.com/search.json'.
        timeout:
            HTTP request timeout in seconds for SerpAPI calls.
        """
        if requests is None:
            raise ImportError(
                "The 'requests' package is required for SerpApiEngine. "
                "Install it with `pip install requests`."
            )

        self.endpoint = endpoint or "https://serpapi.com/search.json"
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SerpApiEngine requires an API key. "
                "Pass `api_key=...` or set SERPAPI_API_KEY in the environment."
            )

        self.timeout = int(timeout)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search(
        self,
        *,
        query: str,
        hl: str,
        max_results: int,
        location: Optional[str],
        gl: str,
    ) -> Tuple[List[EngineResult], EngineSearchMeta, Dict[str, Any]]:
        params: Dict[str, Any] = {
            "engine": "google",
            "q": query,
            "num": max_results,
            "hl": hl,
            "gl": gl,
            "api_key": self.api_key,
        }

        if location:
            params["location"] = location

        resp = requests.get(self.endpoint, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Unexpected response from SerpAPI: expected dict, got {type(data)!r}"
            )

        organic = data.get("organic_results", []) or []
        organic = organic[:max_results]

        results: List[EngineResult] = []
        for item in organic:
            title = item.get("title") or "Untitled result"
            url = item.get("link") or ""
            snippet = item.get("snippet") or ""
            score = None  # SerpAPI doesn't expose a score directly

            results.append(
                EngineResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    score=score,
                )
            )

        info = data.get("search_information", {}) or {}
        total_results = info.get("total_results")

        meta = EngineSearchMeta(
            total_results=total_results,
            provider=self.name,
            extra={"search_information": info},
        )

        return results, meta, data

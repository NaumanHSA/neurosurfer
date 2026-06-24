"""SerpAPI search engine — proxies Google results via the SerpAPI service."""

from __future__ import annotations

import os

import httpx

from .base import BaseEngine, EngineResult

_SERPAPI_BASE = "https://serpapi.com/search"


class SerpApiEngine(BaseEngine):
    """Google search results via SerpAPI (https://serpapi.com).

    Requires ``SERPAPI_API_KEY`` to be set in the environment, or passed
    explicitly as ``api_key``. Uses ``httpx`` (sync client) — no extra deps.
    """

    name = "serpapi"

    def __init__(self, api_key: str | None = None, timeout: float = 10.0) -> None:
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY", "")
        self.timeout = timeout

    def search(self, query: str, max_results: int) -> list[EngineResult]:
        if not self.api_key:
            raise RuntimeError(
                "SerpAPI requires an API key. "
                "Set the SERPAPI_API_KEY environment variable or pass api_key= to SerpApiEngine()."
            )
        params: dict[str, object] = {
            "q": query,
            "api_key": self.api_key,
            "num": max_results,
            "engine": "google",
            "output": "json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(_SERPAPI_BASE, params=params)
            resp.raise_for_status()

        data: dict = resp.json()
        results: list[EngineResult] = []
        for r in data.get("organic_results", [])[:max_results]:
            url = r.get("link", "")
            if not url:
                continue
            results.append(
                EngineResult(
                    title=r.get("title", url).strip(),
                    url=url.strip(),
                    snippet=r.get("snippet", "").strip(),
                )
            )
        return results

    def is_available(self) -> bool:
        return bool(self.api_key or os.environ.get("SERPAPI_API_KEY", ""))

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None  # We'll raise a clear error in __init__ if missing.

from dataclasses import dataclass

from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn


@dataclass
class WebSearchResult:
    """
    Normalized search result item (provider-agnostic).
    """
    title: str
    url: str
    snippet: str
    score: Optional[float] = None


class WebSearchTool(BaseTool):
    """
    Web search tool backed by SerpAPI (Google engine).

    This tool calls SerpAPI's `/search.json` endpoint and returns:
      - A short text `summary`
      - A normalized list of `results` (title, url, snippet, score)

    It is designed to be LLM-friendly and provider-agnostic on the output side,
    so you can swap providers in the future without changing the agent prompts.

    Example SerpAPI response (truncated):

        {
          "search_metadata": {...},
          "search_parameters": {...},
          "search_information": {...},
          "organic_results": [
            {
              "position": 1,
              "title": "...",
              "link": "https://example.com",
              "snippet": "...",
              ...
            },
            ...
          ]
        }

    We primarily use:
      - query       => from the tool input
      - organic_results[*].title / link / snippet
      - search_information.total_results (for a nicer summary, if present)
    """

    spec = ToolSpec(
        name="web_search",
        description="Search the web via SerpAPI (Google engine) and return URLs and snippets.",
        when_to_use=(
            "Use this tool when you need up-to-date information, facts, or references "
            "from the internet. Ideal for research, checking current data, and "
            "finding external sources."
        ),
        inputs=[
            ToolParam(
                name="query",
                type="string",
                description="The web search query.",
                required=True,
            ),
            ToolParam(
                name="num_results",
                type="integer",
                description="How many top results to return (default 5, max 10).",
                required=False,
            ),
            ToolParam(
                name="location",
                type="string",
                description=(
                    "Optional location string for localized search, e.g. "
                    "'Seattle-Tacoma, WA, Washington, United States'. "
                    "If omitted, SerpAPI defaults will be used."
                ),
                required=False,
            ),
            ToolParam(
                name="hl",
                type="string",
                description="Interface language (e.g. 'en'). Defaults to 'en'.",
                required=False,
            ),
            ToolParam(
                name="gl",
                type="string",
                description="Geolocation / country code (e.g. 'us'). Defaults to 'us'.",
                required=False,
            ),
            ToolParam(
                name="include_raw",
                type="boolean",
                description=(
                    "If true, include the raw SerpAPI payload in `extras['raw']` "
                    "for debugging or advanced use."
                ),
                required=False,
            ),
        ],
        returns=ToolReturn(
            type="object",
            description=(
                "JSON object with `query`, `summary`, and `results` "
                "(each result has `title`, `url`, `snippet`, `score`)."
            ),
        ),
    )

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 10,
        max_results: int = 10,
        default_location: Optional[str] = None,
    ) -> None:
        """
        Initialize the SerpAPI-backed web search tool.

        Parameters
        ----------
        api_key:
            SerpAPI key. If None, uses SERPAPI_API_KEY from the environment.
        endpoint:
            Optional override for SerpAPI endpoint, defaults to
            'https://serpapi.com/search.json'.
        timeout:
            HTTP request timeout in seconds.
        max_results:
            Upper bound on `num_results` provided at call time.
        default_location:
            Optional default `location` string for geo-targeted searches.
        """
        super().__init__()

        if requests is None:
            raise ImportError(
                "The 'requests' package is required for WebSearchTool. "
                "Install it with `pip install requests`."
            )

        self.endpoint = endpoint or "https://serpapi.com/search.json"
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "WebSearchTool (SerpAPI) requires an API key. "
                "Pass `api_key=...` or set SERPAPI_API_KEY in the environment."
            )

        self.timeout = int(timeout)
        self.max_results = int(max_results)
        self.default_location = default_location

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        query: str,
        num_results: int = 5,
        location: Optional[str] = None,
        hl: str = "en",
        gl: str = "us",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> ToolResponse:
        """
        Execute the web search via SerpAPI.

        Parameters
        ----------
        query:
            Search query string.
        num_results:
            Desired number of results. Clamped to [1, self.max_results].
        location:
            Optional SerpAPI `location` string. If not provided, `default_location`
            is used (if set), otherwise SerpAPI's own default.
        hl:
            Interface language (e.g. 'en').
        gl:
            Geolocation / country code (e.g. 'us').
        include_raw:
            If True, include raw SerpAPI JSON in `extras['raw']`.

        Returns
        -------
        ToolResponse
            observation: dict with keys:
                - `query`: str
                - `summary`: str
                - `results`: List[{title, url, snippet, score}]
                - `provider`: "serpapi"
                - `elapsed_ms`: int
        """
        t0 = time.time()

        k = max(1, min(int(num_results or 5), self.max_results))
        loc = location or self.default_location

        raw = self._search_serpapi(
            query=query,
            max_results=k,
            location=loc,
            hl=hl,
            gl=gl,
        )

        organic = raw.get("organic_results", []) or []
        normalized_results: List[WebSearchResult] = []

        for item in organic[:k]:
            title = item.get("title") or "Untitled result"
            url = item.get("link") or ""
            snippet = item.get("snippet") or ""
            # SerpAPI doesn't give a direct relevance score; keep None for now.
            score = None

            normalized_results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    score=score,
                )
            )

        # Build a human-friendly summary
        info = raw.get("search_information", {}) or {}
        total_results = info.get("total_results")
        if total_results is not None:
            total_str = f"~{total_results:,} results"
        else:
            total_str = "some results"

        if not normalized_results:
            summary = f"SerpAPI search for {query!r} returned no organic results."
        else:
            lines = [f"Top {len(normalized_results)} results out of {total_str} for: {query!r}"]
            for i, r in enumerate(normalized_results, start=1):
                lines.append(f"{i}. {r.title} — {r.url}")
            summary = "\n".join(lines)

        observation: Dict[str, Any] = {
            "query": query,
            "summary": summary,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "score": r.score,
                }
                for r in normalized_results
            ],
            "provider": "serpapi",
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

        extras: Dict[str, Any] = {}
        if include_raw:
            extras["raw"] = raw

        return ToolResponse(
            final_answer=False,  # usually part of a bigger reasoning chain
            observation=observation,
            extras=extras,
        )

    # ------------------------------------------------------------------ #
    # SerpAPI HTTP implementation
    # ------------------------------------------------------------------ #
    def _search_serpapi(
        self,
        *,
        query: str,
        max_results: int,
        location: Optional[str],
        hl: str,
        gl: str,
    ) -> Dict[str, Any]:
        """
        Call SerpAPI's `/search.json` endpoint and return the raw JSON.

        We use `engine=google` by default. You can adjust the query params
        as needed (e.g. device, safe, etc.).
        """
        params: Dict[str, Any] = {
            "engine": "google",
            "q": query,
            "num": max_results,
            "hl": hl,
            "gl": gl,
            "api_key": self.api_key,
        }

        # Optional location
        if location:
            params["location"] = location

        # You can customize other SerpAPI params here if needed:
        # params["safe"] = "active"
        # params["device"] = "desktop"
        # params["google_domain"] = "google.com"

        resp = requests.get(self.endpoint, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, dict):
            raise RuntimeError(
                f"Unexpected response from SerpAPI: expected dict, got {type(data)!r}"
            )

        return data

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from neurosurfer.models.chat_models import BaseChatModel

try:
    import requests
except ImportError:
    requests = None  # We'll raise a clear error in __init__ if missing.

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # We'll raise a clear error in __init__ if crawling is enabled.

from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """
    Normalized search result item (provider-agnostic).

    Extended to optionally include crawled page content.
    """
    title: str
    url: str
    snippet: str
    score: Optional[float] = None

    # Crawled content (if enabled)
    content: Optional[str] = None          # Cleaned plain text
    content_length: Optional[int] = None   # len(content) if available
    error: Optional[str] = None            # Crawl / parse error if any


class WebSearchTool(BaseTool):
    """
    Web search + optional crawl tool backed by SerpAPI (Google engine).

    1. Calls SerpAPI `/search.json` to get organic results.
    2. Optionally crawls the top N result URLs in parallel.
    3. Extracts clean text from HTML using BeautifulSoup.
    4. Returns LLM-friendly JSON with search metadata and content.

    Output (`results` in ToolResponse):

        {
          "query": str,
          "summary": str,
          "results": [
            {
              "title": str,
              "url": str,
              "snippet": str,
              "score": float | null,
              "content": str | null,
              "content_length": int | null,
              "error": str | null
            },
            ...
          ],
          "provider": "serpapi",
          "elapsed_ms": int
        }
    """

    spec = ToolSpec(
        name="web_search",
        description=(
            "Search the web via SerpAPI (Google engine). Optionally crawls the "
            "top results and returns page text content for LLM consumption."
        ),
        when_to_use=(
            "Use this tool when you need up-to-date information or external web "
            "content. It can fetch both search result snippets and the actual "
            "page text, making it suitable for research and RAG-style workflows."
        ),
        inputs=[
            ToolParam(
                name="query",
                type="string",
                description="The web search query.",
                required=True,
            ),
            ToolParam(
                name="hl",
                type="string",
                description="Interface language (e.g. 'en'). Defaults to 'en'.",
                required=False,
            ),
        ],
        returns=ToolReturn(
            type="object",
            description=(
                "JSON object with `query`, `summary`, and `results`. Each result "
                "includes `title`, `url`, `snippet`, optional `content` "
                "(page text), and `error` if crawling failed."
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
        gl: str = "us",
        include_raw: bool = False,
        # Crawling configuration
        enable_crawl: bool = True,
        max_crawl_results: Optional[int] = None,   # None => same as max_results
        crawl_timeout: int = 10,
        max_concurrent_crawls: int = 4,
        allowed_content_types: Optional[Sequence[str]] = None,
        max_content_chars: int = 20000,
        user_agent: Optional[str] = None,
        summarize_search_results: bool = False,
        llm: Optional[BaseChatModel] = None,
        preferred_domains: Optional[Sequence[str]] = None,
        prefer_preferred_domains: bool = True,
    ) -> None:
        """
        Initialize the SerpAPI-backed web search + crawl tool.

        Parameters
        ----------
        api_key:
            SerpAPI key. If None, uses SERPAPI_API_KEY from the environment.
        endpoint:
            Optional override for SerpAPI endpoint, defaults to
            'https://serpapi.com/search.json'.
        timeout:
            HTTP request timeout for the SERP API in seconds.
        max_results:
            Upper bound on number of search results to consider.
        default_location:
            Optional default `location` string for geo-targeted SERP searches.
        gl:
            Default `gl` string (geo location) for SERP.
        include_raw:
            If True, include raw SerpAPI payload in `extras['raw_serpapi']`.
        enable_crawl:
            If True, crawl and extract text from the top N URLs.
        max_crawl_results:
            Max number of results to crawl; if None, uses `max_results`.
        crawl_timeout:
            Per-URL crawl timeout in seconds.
        max_concurrent_crawls:
            Max number of parallel crawl workers.
        allowed_content_types:
            Iterable of allowed Content-Type prefixes (e.g. ["text/html"]).
            If None, defaults to ("text/html", "application/xhtml+xml").
        max_content_chars:
            Maximum number of characters to keep from page text.
        user_agent:
            Optional custom User-Agent header for crawling.
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
        self.include_raw = include_raw
        self.gl = gl

        # Crawl-related config
        self.enable_crawl = bool(enable_crawl)
        self.max_crawl_results = int(max_crawl_results) if max_crawl_results is not None else None
        self.crawl_timeout = int(crawl_timeout)
        self.max_concurrent_crawls = max(1, int(max_concurrent_crawls))
        self.max_content_chars = int(max_content_chars)

        if allowed_content_types is None:
            self.allowed_content_types = ("text/html", "application/xhtml+xml", "text/plain")
        else:
            self.allowed_content_types = tuple(allowed_content_types)

        self.user_agent = user_agent or "Mozilla/5.0 (compatible; NeurosurferWebSearch/1.0)"
        self.summarize_search_results = bool(summarize_search_results)
        self.llm = llm

        # NEW:
        self.preferred_domains = tuple(
            d.lower() for d in (preferred_domains or ("wikipedia.org", "medium.com"))
        )
        self.prefer_preferred_domains = bool(prefer_preferred_domains)

        if self.enable_crawl and BeautifulSoup is None:
            raise ImportError(
                "BeautifulSoup (beautifulsoup4) is required for HTML parsing when "
                "enable_crawl=True. Install it with `pip install beautifulsoup4`."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        query: str,
        hl: str = "en",
        **_: Any,
    ) -> ToolResponse:
        """
        Execute the web search via SerpAPI and optionally crawl result URLs.

        Parameters
        ----------
        query:
            Search query string.
        hl:
            Interface language (e.g. 'en').

        Returns
        -------
        ToolResponse
            results: dict with keys:
                - `query`: str
                - `summary`: str
                - `results`: List[result dicts]
                - `provider`: "serpapi"
                - `elapsed_ms`: int
        """
        t0 = time.time()

        raw = self._search_serpapi(
            query=query,
            max_results=self.max_results,
            location=self.default_location,
            hl=hl,
            gl=self.gl,
        )

        organic = raw.get("organic_results", []) or []
        organic = organic[: self.max_results]

        if not organic:
            summary = f"SerpAPI search for {query!r} returned no organic results."
            results_payload: Dict[str, Any] = {
                "query": query,
                "summary": summary,
                "results": [],
                "provider": "serpapi",
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
            extras: Dict[str, Any] = {}
            if self.include_raw:
                extras["raw_serpapi"] = raw

            return ToolResponse(
                final_answer=False,
                results=results_payload,
                extras=extras,
            )

        # Build normalized results with optional crawling
        normalized_results: List[WebSearchResult] = []

        if self.enable_crawl:
            normalized_results = self._build_results_with_crawl(organic)
        else:
            for item in organic:
                normalized_results.append(
                    self._build_result_from_item(
                        item=item,
                        crawl=False,
                    )
                )

        # Build a human-friendly summary
        info = raw.get("search_information", {}) or {}
        total_results = info.get("total_results")
        if total_results is not None:
            total_str = f"~{total_results:,} results"
        else:
            total_str = "some results"

        lines = [f"Top {len(normalized_results)} results out of {total_str} for: {query!r}"]
        for i, r in enumerate(normalized_results, start=1):
            lines.append(f"{i}. {r.title} — {r.url}")
        summary = "\n".join(lines)

        results_dict: Dict[str, Any] = {
            "query": query,
            "summary": summary,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "score": r.score,
                    "content": r.content,
                    "content_length": r.content_length,
                    "error": r.error,
                }
                for r in normalized_results
            ],
            "provider": "serpapi",
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

        extras: Dict[str, Any] = {}
        if self.include_raw:
            extras["raw_serpapi"] = raw

        return ToolResponse(
            final_answer=False,  # usually part of a bigger reasoning chain
            results=results_dict,
            extras=extras,
        )

    # ------------------------------------------------------------------ #
    # Result building + crawling
    # ------------------------------------------------------------------ #
    def _build_results_with_crawl(self, organic: List[Dict[str, Any]]) -> List[WebSearchResult]:
        """
        Crawl top N results in parallel and return normalized WebSearchResult list
        in the original SERP order.

        If preferred_domains is configured, we first try to crawl those. If none
        are found, we fall back to the usual top-N.
        """
        num_to_crawl = self.max_crawl_results or len(organic)
        num_to_crawl = min(num_to_crawl, len(organic))

        # --- NEW: pick which indices to crawl ---
        preferred_indices: List[int] = []
        if self.prefer_preferred_domains and self.preferred_domains:
            for idx, item in enumerate(organic):
                url = item.get("link") or ""
                if self._is_preferred_domain(url):
                    preferred_indices.append(idx)

        if preferred_indices:
            indices_to_crawl = preferred_indices[:num_to_crawl]
        else:
            indices_to_crawl = list(range(num_to_crawl))

        results: List[Optional[WebSearchResult]] = [None] * len(organic)
        futures = {}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=self.max_concurrent_crawls) as executor:
            for idx, item in enumerate(organic):
                crawl = idx in indices_to_crawl
                future = executor.submit(self._build_result_from_item, item=item, crawl=crawl)
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.exception("Error building result for index %s: %s", idx, e)
                    item = organic[idx]
                    fallback = self._build_result_from_item(item=item, crawl=False)
                    fallback.error = f"Internal error while crawling: {e}"
                    results[idx] = fallback

        return [r for r in results if r is not None]

    def _build_result_from_item(self, *, item: Dict[str, Any], crawl: bool) -> WebSearchResult:
        """
        Normalize a single SerpAPI organic result item and optionally crawl its URL.
        """
        title = item.get("title") or "Untitled result"
        url = item.get("link") or ""
        snippet = item.get("snippet") or ""

        result = WebSearchResult(
            title=title,
            url=url,
            snippet=snippet,
            score=None,
        )

        if crawl and url:
            crawl_res = self._crawl_url(url)
            result.content = crawl_res.get("content")
            result.content_length = (
                len(result.content) if result.content is not None else None
            )
            result.error = crawl_res.get("error")

        return result

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

        We use `engine=google` by default.
        """
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

        return data

    # ------------------------------------------------------------------ #
    # URL crawling + HTML parsing
    # ------------------------------------------------------------------ #
    def _crawl_url(self, url: str) -> Dict[str, Any]:
        """
        Fetch page content and extract clean text.

        Returns a dict:
            {
                "content": str | None,
                "error": str | None,
            }
        """
        if not url.startswith("http"):
            return {"content": None, "error": "Invalid URL"}

        try:
            resp = requests.get(
                url,
                timeout=self.crawl_timeout,
                headers={"User-Agent": self.user_agent},
            )
        except Exception as e:
            logger.warning("Failed to fetch URL %s: %s", url, e)
            return {"content": None, "error": f"Request failed: {e}"}

        content_type = resp.headers.get("Content-Type", "")
        if not any(content_type.startswith(prefix) for prefix in self.allowed_content_types):
            return {
                "content": None,
                "error": f"Unsupported content type: {content_type}",
            }

        try:
            resp.raise_for_status()
        except Exception as e:
            return {"content": None, "error": f"HTTP error: {e}"}

        html = resp.text

        try:
            text = self._html_to_text(html, url=url)
        except Exception as e:
            logger.warning("Failed to parse HTML for %s: %s", url, e)
            return {"content": None, "error": f"HTML parse error: {e}"}

        if text and len(text) > self.max_content_chars:
            text = text[: self.max_content_chars]

        return {"content": text, "error": None}

    def _html_to_text(self, html: str, url: Optional[str] = None) -> str:
        """
        Convert HTML to cleaned plain text using BeautifulSoup.

        If the URL belongs to a known domain (e.g. Wikipedia, Medium), use
        a domain-specific content extractor to avoid menus / sidebars / language lists.
        """
        if BeautifulSoup is None:
            raise RuntimeError(
                "BeautifulSoup is not available but HTML parsing was requested."
            )

        soup = BeautifulSoup(html, "html.parser")

        # Domain-specific handling
        domain = self._get_domain(url or "")

        if "wikipedia.org" in domain:
            # Wikipedia: main article content lives here
            content = soup.find("div", id="bodyContent") or soup.find("div", id="mw-content-text")
            if content:
                soup = content

        elif "medium.com" in domain:
            # Medium: main content is usually inside <article>
            article = soup.find("article")
            if article:
                soup = article

        # Strip boilerplate tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        return text

    # ------------------------------------------------------------------ #
    # Domain helpers
    # ------------------------------------------------------------------ #
    def _get_domain(self, url: str) -> str:
        from urllib.parse import urlparse

        try:
            netloc = urlparse(url).netloc.lower()
        except Exception:
            return ""
        # strip "www."
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc

    def _is_preferred_domain(self, url: str) -> bool:
        if not self.preferred_domains:
            return False
        domain = self._get_domain(url)
        return any(domain.endswith(pref) for pref in self.preferred_domains)
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Type

try:
    import requests
except ImportError:
    requests = None  # checked when crawling is enabled

from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
from neurosurfer.models.chat_models import BaseChatModel

from .engines.base import EngineResult, EngineSearchMeta, SearchEngine
from .engines.serpapi import SerpApiEngine
from .utils import (
    build_result_from_engine_result,
    build_results_with_crawl,
    summarize_with_llm,
)
from .config import WebSearchConfig, DOMAIN_CONTENT_CONFIG_DEFAULT


logger = logging.getLogger(__name__)


# Engine registry
ENGINE_REGISTRY: Dict[str, Type[SearchEngine]] = {
    "serpapi": SerpApiEngine,
}

class WebSearchTool(BaseTool):
    """
    Unified web search + optional crawl + optional LLM summarization.

    This tool:
      1. Uses a pluggable engine (SerpAPI now; Tavily, etc. later) to get SERP results.
      2. Optionally crawls the top results, with smart domain-specific extraction.
      3. Optionally asks an LLM to create a long, refined summary of the results.

    Parameters common across engines:
      - engine: which backend to use ("serpapi", ...)
      - engine_kwargs: dict with engine-specific config (api_key, endpoint, ...)
      - max_results, location, gl: SERP-level knobs
      - enable_crawl, max_crawl_results, max_concurrent_crawls, etc.
      - domain_content_config, preferred_domains: smarter content extraction
      - llm: optional LLM instance for summarization
    """

    spec = ToolSpec(
        name="web_search",
        description=(
            "Search the web using a pluggable backend (e.g. SerpAPI). "
            "Optionally crawls the top results, extracts page content, and "
            "summarizes it with an LLM."
        ),
        when_to_use=(
            "Use this tool when you need up-to-date information, external web "
            "content, or detailed summaries combining multiple sources. The "
            "tool can return raw results or a refined LLM summary."
        ),
        inputs=[
            ToolParam(
                name="query",
                type="string",
                description="The web search query.",
                required=True
            ),
            ToolParam(
                name="hl",
                type="string",
                description="Interface language (e.g. 'en'). Defaults to 'en'.",
                required=False
            )
        ],
        returns=ToolReturn(
            type="object",
            description=(
                "JSON object with keys: `query`, `summary`, `results`, `provider`, "
                "`elapsed_ms`, and optionally `llm_summary` if summarization is enabled."
            ),
        ),
    )

    def __init__(
        self,
        *,
        config: WebSearchConfig = WebSearchConfig(),
        llm: Optional[BaseChatModel] = None,    # Optional LLM for summarization when summarize=True
    ) -> None:
        super().__init__()

        # ---------------- Engine selection ----------------
        self.config = config
        engine_name = self.config.engine.lower()
        if engine_name not in ENGINE_REGISTRY:
            raise ValueError(
                f"Unknown web search engine {self.config.engine!r}. "
                f"Available engines: {', '.join(sorted(ENGINE_REGISTRY.keys()))}"
            )

        engine_cls = ENGINE_REGISTRY[engine_name]
        self.engine: SearchEngine = engine_cls(**(self.config.engine_kwargs or {}))

        if self.config.enable_crawl and requests is None:
            raise ImportError(
                "The 'requests' package is required for crawling. "
                "Install it with `pip install requests`."
            )

        # ---------------- Domain config ----------------
        # Merge user overrides into default config
        self.config.domain_content_config.update(DOMAIN_CONTENT_CONFIG_DEFAULT)

        # Preferred domains: by default, all known domain configs
        if self.config.preferred_domains is None:
            self.config.preferred_domains = tuple(self.config.domain_content_config.keys())
        # ---------------- LLM ----------------
        self.llm = llm

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        query: str,
        hl: str = "en",
        **kwargs,
    ) -> ToolResponse:
        """
        Execute the web search via the configured engine, optionally crawl,
        and optionally summarize with an LLM.
        """
        t0 = time.time()

        # 1) Engine-level search
        engine_results, meta, raw = self.engine.search(
            query=query,
            hl=hl,
            max_results=self.config.max_results,
            location=self.config.location,
            gl=self.config.gl,
        )

        if not engine_results:
            summary = f"{self.engine.name} search for {query!r} returned no results."
            results_payload: Dict[str, Any] = {
                "query": query,
                "summary": summary,
                "results": [],
                "provider": meta.provider or self.engine.name,
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
            extras: Dict[str, Any] = {}
            if self.config.include_raw:
                extras[f"raw_{meta.provider or self.engine.name}"] = raw

            return ToolResponse(
                final_answer=False,
                results=results_payload,
                extras=extras,
            )

        # 2) Normalize results + optional crawling
        if self.config.enable_crawl:
            normalized_results = build_results_with_crawl(
                config=self.config,
                engine_results=engine_results
            )
        else:
            normalized_results = [
                build_result_from_engine_result(
                    config=self.config,
                    item=r,
                    crawl=False
                )
                for r in engine_results
            ]

        # 3) Build human-friendly summary
        if meta.total_results is not None:
            total_str = f"~{meta.total_results:,} results"
        else:
            total_str = "some results"

        lines = [
            f"Top {len(normalized_results)} results out of {total_str} for: {query!r}"
        ]
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
            "provider": meta.provider or self.engine.name,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

        # 4) Optional LLM summarization
        if self.config.summarize and self.llm is not None:
            llm_summary = summarize_with_llm(
                llm=self.llm,
                results_dict=results_dict,
                stream=self.config.stream_summary,
            )
            results_dict["llm_summary"] = llm_summary

        extras: Dict[str, Any] = {}
        if self.config.include_raw:
            extras[f"raw_{meta.provider or self.engine.name}"] = raw

        return ToolResponse(
            final_answer=False,
            results=results_dict,
            extras=extras,
        )

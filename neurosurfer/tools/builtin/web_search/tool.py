"""WebSearchTool — pluggable engine, BM25-ranked, budget-capped result injection."""

from __future__ import annotations

import asyncio
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from ....llm.tokens import estimate_text_tokens
from ...base import Tool, ToolContext, ToolResult
from . import config as _config
from .engines import get_engine
from .extractor import _fetch_page, _store_full_text, extract_body
from .ranker import chunk_text, rank_chunks, select_within_budget


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=2, description="The search query.")
    max_results: int = Field(
        default=_config.MAX_RESULTS,
        ge=1,
        le=10,
        description="Number of search results to list.",
    )
    fetch: bool = Field(
        default=True,
        description="Fetch and extract the top result pages (off = snippets only).",
    )
    engine: Literal["ddg", "serpapi"] | None = Field(
        default=None,
        description=(
            "Search engine to use. Defaults to WEB_SEARCH_ENGINE env var (default: 'ddg'). "
            "Options: 'ddg' (DuckDuckGo, free) or 'serpapi' (Google via SerpAPI, needs SERPAPI_API_KEY)."
        ),
    )


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for current information. Supports DuckDuckGo (free, default) "
        "and SerpAPI/Google (requires SERPAPI_API_KEY). Returns result titles, URLs and "
        "snippets, and — unless fetch=false — the most relevant extracted text from the "
        "top pages, ranked against your query and capped to a token budget. "
        "Cite source URLs as markdown links in your answer."
    )
    input_model = WebSearchArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    def is_enabled(self) -> bool:
        try:
            engine = get_engine(_config.DEFAULT_ENGINE)
            return engine.is_available()
        except ValueError:
            return False

    def progress_message(self, args: dict) -> str:
        return f"Searching the web for {args.get('query') or '…'}…"

    async def call(self, args: WebSearchArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        engine_name = args.engine or _config.DEFAULT_ENGINE
        try:
            engine = get_engine(engine_name)
        except ValueError as e:
            return ToolResult.error(str(e))

        try:
            hits_raw = await asyncio.to_thread(engine.search, args.query, args.max_results)
        except RuntimeError as e:
            return ToolResult.error(str(e))
        except Exception as e:  # noqa: BLE001
            return ToolResult.error(
                f"Web search failed ({engine_name}): {type(e).__name__}: {e}. "
                "Try again or rephrase your query."
            )

        if not hits_raw:
            return ToolResult.ok(
                f'No results for "{args.query}" via {engine_name}. '
                "DuckDuckGo can be rate-limited from datacenter/VPN IPs; try rephrasing or retry later."
            )

        # Convert EngineResult dataclasses → dicts for rendering helpers
        hits = [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in hits_raw]

        sections: list[str] = [self._render_snippets(args.query, hits)]

        if args.fetch:
            extracted = await self._fetch_and_extract(hits[: _config.FETCH_TOP_K])
            if extracted:
                sections.append(self._render_content(args.query, extracted))

        sections.append(
            "REMINDER: cite the source URLs above as markdown links in your answer."
        )
        return ToolResult.ok("\n\n".join(sections))

    # ── rendering helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _render_snippets(query: str, hits: list[dict[str, str]]) -> str:
        lines = [f'Search results for "{query}":', ""]
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. {h['title']}\n   {h['url']}")
            if h["snippet"]:
                lines.append(f"   {h['snippet']}")
        return "\n".join(lines)

    async def _fetch_and_extract(
        self, hits: list[dict[str, str]]
    ) -> list[tuple[str, str]]:
        """Fetch + extract body text for each hit. Returns [(url, body), ...]."""
        async with httpx.AsyncClient(headers={"User-Agent": _config._UA}) as client:
            htmls = await asyncio.gather(
                *(_fetch_page(client, h["url"]) for h in hits)
            )
        out: list[tuple[str, str]] = []
        for h, html in zip(hits, htmls, strict=False):
            body = extract_body(html) if html else ""
            if body:
                out.append((h["url"], body))
        return out

    @staticmethod
    def _render_content(query: str, pages: list[tuple[str, str]]) -> str:
        # Pool chunks across all fetched pages, tagged with their source URL.
        tagged: list[tuple[str, str]] = []
        for url, body in pages:
            for chunk in chunk_text(body):
                tagged.append((url, chunk))

        total = sum(estimate_text_tokens(c) for _, c in tagged)
        header = "Extracted page content"

        if total <= _config.BUDGET_TOKENS:
            # Fits under budget: inject everything grouped by source.
            blocks = [f"--- {url} ---\n{body}" for url, body in pages]
            return f"{header}:\n\n" + "\n\n".join(blocks)

        # Over budget: rank pooled chunks against the query and keep only the top ones.
        chunks = [c for _, c in tagged]
        ranked = rank_chunks(query, chunks)
        keep = set(select_within_budget(chunks, ranked, _config.BUDGET_TOKENS))
        blocks = [f"[{tagged[i][0]}]\n{tagged[i][1]}" for i in sorted(keep)]
        stored = [f"- {url}: {_store_full_text(url, body)}" for url, body in pages]
        return (
            f"{header} (ranked excerpt — most query-relevant chunks of "
            f"~{total} tokens, capped to ~{_config.BUDGET_TOKENS}):\n\n"
            + "\n\n".join(blocks)
            + "\n\nFull extracted text saved (use read_file for more):\n"
            + "\n".join(stored)
        )

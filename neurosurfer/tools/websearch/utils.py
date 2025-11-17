from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .engines.base import EngineResult
from .extractor import get_domain, html_to_text
from .config import WebSearchConfig, DOMAIN_CONTENT_CONFIG_DEFAULT
from neurosurfer.models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# WebSearchResult (post-crawl, LLM-ready)
# --------------------------------------------------------------------------- #
@dataclass
class WebSearchResult:
    """
    Final normalized result, ready for LLM use.

    Combines:
      - engine-level info (title, url, snippet, score)
      - optional crawled content and length
      - crawl error, if any
    """
    title: str
    url: str
    snippet: str
    score: Optional[float] = None

    # Crawled content
    content: Optional[str] = None
    content_length: Optional[int] = None
    error: Optional[str] = None


# ------------------------------------------------------------------ #
# Result building + crawling
# ------------------------------------------------------------------ #
def build_results_with_crawl(
    config: WebSearchConfig,
    engine_results: List[EngineResult],
) -> List[WebSearchResult]:
    """
    Crawl top N results in parallel and return WebSearchResult list
    in the original SERP order.

    If preferred_domains are configured, we first try to crawl those. If none
    are found, we fall back to the usual top-N.
    """
    num_to_crawl = config.max_crawl_results or len(engine_results)
    num_to_crawl = min(num_to_crawl, len(engine_results))

    # Pick which indices to crawl
    preferred_indices: List[int] = []
    if config.preferred_domains:
        for idx, item in enumerate(engine_results):
            if is_preferred_domain(item.url, preferred_domains=config.preferred_domains):
                preferred_indices.append(idx)
    indices_to_crawl = preferred_indices[:num_to_crawl] if preferred_indices else list(range(num_to_crawl))

    results: List[Optional[WebSearchResult]] = [None] * len(engine_results)
    futures: Dict[Any, int] = {}
    with ThreadPoolExecutor(max_workers=config.max_concurrent_crawls) as executor:
        for idx, item in enumerate(engine_results):
            crawl = idx in indices_to_crawl
            future = executor.submit(
                build_result_from_engine_result,
                config=config,
                item=item,
                crawl=crawl
            )
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.exception("Error building result for index %s: %s", idx, e)
                item = engine_results[idx]
                fallback = build_result_from_engine_result(
                    config=config,
                    item=item,
                    crawl=False,
                )
                fallback.error = f"Internal error while crawling: {e}"
                results[idx] = fallback

    return [r for r in results if r is not None]

def build_result_from_engine_result(
    config: WebSearchConfig,
    item: EngineResult,
    *,
    crawl: bool,
) -> WebSearchResult:
    result = WebSearchResult(
        title=item.title,
        url=item.url,
        snippet=item.snippet,
        score=item.score,
    )
    if crawl and item.url:
        crawl_res: Dict[str, Any] = crawl_url(url=item.url, config=config, content_char_limit=config.content_char_limit)
        result.content = crawl_res.get("content")
        result.content_length = len(result.content) if result.content is not None else None
        result.error = crawl_res.get("error")
    return result

# ------------------------------------------------------------------ #
# URL crawling + HTML parsing
# ------------------------------------------------------------------ #
def crawl_url(url: str, *, config: WebSearchConfig, content_char_limit: int) -> Dict[str, Any]:
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

    if requests is None:
        return {"content": None, "error": "requests is not available for crawling"}

    try:
        resp = requests.get(
            url,
            timeout=config.crawl_timeout,
            headers={"User-Agent": config.user_agent},
        )
    except Exception as e:
        logger.warning("Failed to fetch URL %s: %s", url, e)
        return {"content": None, "error": f"Request failed: {e}"}

    content_type = resp.headers.get("Content-Type", "")
    if not any(content_type.startswith(prefix) for prefix in config.allowed_content_types):
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
        text = html_to_text(
            html=html,
            url=url,
            domain_content_config=DOMAIN_CONTENT_CONFIG_DEFAULT,
        )
    except Exception as e:
        logger.warning("Failed to parse HTML for %s: %s", url, e)
        return {"content": None, "error": f"HTML parse error: {e}"}

    if text and len(text) > content_char_limit:
        text = text[:content_char_limit]

    return {"content": text, "error": None}


def is_preferred_domain(url: str, preferred_domains: Sequence[str]) -> bool:
    domain = get_domain(url)
    if not domain or not preferred_domains:
        return False
    domain = domain.lower()
    return any(domain == d or domain.endswith(d) for d in preferred_domains)

# ------------------------------------------------------------------ #
# Optional LLM summarization
# ------------------------------------------------------------------ #
def summarize_with_llm(
    llm: BaseChatModel,
    *,
    results_dict: Dict[str, Any],
    stream: bool,
) -> str:
    """
    Use the configured LLM to produce a long, detailed summary of the search results.
    """
    import json

    system_prompt = (
        "You are a helpful research assistant. "
        "Given the following web search results (including page content), "
        "write a clear, detailed, well-structured answer for the user."
    )

    user_prompt = (
        "Here are the search results as JSON. "
        "Use them to answer the user's query.\n\n"
        f"User query:\n{results_dict['query']}\n\n"
        "Search results JSON:\n"
        f"{json.dumps(results_dict['results'], ensure_ascii=False, indent=2)}"
    )

    if not stream:
        # Non-streaming: assume .ask returns a full string or a structured object
        out = llm.ask(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stream=False,
        )
        if isinstance(out, str):
            return out
        # Fallback if the model returns a more complex object
        return str(out)

    # Streaming mode: accumulate chunks into a single string
    stream_response = llm.ask(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stream=True,
    )

    parts: List[str] = []
    for chunk in stream_response:
        # Assuming OpenAI-style delta; adjust if your BaseModel differs
        try:
            delta = getattr(chunk.choices[0], "delta", None)
            content = getattr(delta, "content", None) or ""
        except Exception:
            content = str(chunk)
        parts.append(content)
    return "".join(parts)

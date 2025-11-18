from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List, Literal
import requests
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .engines.base import EngineResult
from .extractor import get_domain, html_to_text
from .config import WebSearchConfig, LimitStrategy
from .templates import SUMMARIZE_SYSTEM_PROMPT, SUMMARIZE_USER_PROMPT
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
        crawl_res: Dict[str, Any] = crawl_url(url=item.url, config=config)
        result.content = crawl_res.get("content")
        result.content_length = len(result.content) if result.content is not None else None
        result.error = crawl_res.get("error")
    return result

# ------------------------------------------------------------------ #
# URL crawling + HTML parsing
# ------------------------------------------------------------------ #
def crawl_url(url: str, *, config: WebSearchConfig) -> Dict[str, Any]:
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
            domain_content_config=config.domain_content_config,
        )
    except Exception as e:
        logger.warning("Failed to parse HTML for %s: %s", url, e)
        return {"content": None, "error": f"HTML parse error: {e}"}

    # Apply content limit if configured
    if config.content_words_limit is not None and config.content_words_limit > 0:
        text = limit_content_length(
            content=text,
            max_words=config.content_words_limit,
            strategy=config.content_limit_strategy,
        )
    return {"content": text, "error": None}


def is_preferred_domain(url: str, preferred_domains: Sequence[str]) -> bool:
    domain = get_domain(url)
    if not domain or not preferred_domains:
        return False
    domain = domain.lower()
    return any(domain == d or domain.endswith(d) for d in preferred_domains)

def limit_content_length(
    content: str,
    max_words: int,
    strategy: LimitStrategy = "distributive",
    distributive_segment_size: int = 100,
) -> str:
    """
    Truncate `content` to at most `max_words` words using a given strategy.

    Strategies
    ----------
    - "first":       Keep the first `max_words` words.
    - "last":        Keep the last `max_words` words.
    - "middle":      Keep a centered window of `max_words` words.
    - "head_tail":   Split budget between start and end, join with "...".
    - "distributive":Pick several short chunks distributed across the text,
                     joined with "...".
    """
    words = content.split()
    num_words = len(words)

    if max_words <= 0 or num_words <= max_words:
        return content

    strategy = strategy.lower()
    if strategy == "first":
        return " ".join(words[:max_words])

    if strategy == "last":
        return " ".join(words[-max_words:])

    if strategy == "middle":
        # Centered window
        start = max((num_words - max_words) // 2, 0)
        end = start + max_words
        return " ".join(words[start:end])

    if strategy == "head_tail":
        # Half from the start, half from the end
        head_count = max_words // 2
        tail_count = max_words - head_count
        head = words[:head_count]
        tail = words[-tail_count:] if tail_count > 0 else []
        if not tail:
            return " ".join(head)
        return " ".join(head) + " ... " + " ".join(tail)

    if strategy == "distributive":
        # Very small budgets: just fall back to first
        if max_words < 3:
            return " ".join(words[:max_words])

        # Ensure positive segment size
        distributive_segment_size = max(1, distributive_segment_size)

        # How many segments can we fit if each is ~distributive_segment_size words?
        # e.g. max_words=900, segment_size=100 → 9 segments
        segment_count = max(1, max_words // distributive_segment_size)

        # Safety: don't have more segments than max_words (1 word per segment min)
        segment_count = min(segment_count, max_words)

        # Now recompute exact words per segment so we use (almost) full budget
        words_per_segment = max(1, max_words // segment_count)

        # If even a single segment is almost the whole thing, just do "first"
        if words_per_segment >= num_words:
            return " ".join(words[:max_words])

        # Evenly spaced starting indices across the usable range
        usable_range = max(0, num_words - words_per_segment)
        if segment_count == 1 or usable_range == 0:
            return " ".join(words[:max_words])

        step = max(1, usable_range // (segment_count - 1))
        starts = [min(i * step, usable_range) for i in range(segment_count)]

        segments = [
            " ".join(words[s : s + words_per_segment])
            for s in starts
            if s < num_words
        ]
        # In rare cases we might have fewer segments (e.g., very short text);
        # that's fine — we still respect max_words.
        return " ... ".join(segments)




# ------------------------------------------------------------------ #
# Optional LLM summarization
# ------------------------------------------------------------------ #
def summarize_with_llm(
    llm: BaseChatModel,
    *,
    results_dict: Dict[str, Any],
) -> str:
    """
    Use the configured LLM to produce a long, detailed summary of the search results.
    """
    if "rag_content" in results_dict: 
        results_dict.pop("results")

    user_prompt = SUMMARIZE_USER_PROMPT.format(
        query=results_dict['query'],
        json=json.dumps(results_dict, ensure_ascii=False, indent=2)
    )
    # Streaming mode: accumulate chunks into a single string
    return llm.ask(
        system_prompt=SUMMARIZE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        stream=False,
    ).choices[0].message.content
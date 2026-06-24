#!/usr/bin/env python
"""Manual harness for the web_search tool.

Runs a live DuckDuckGo search through the real WebSearchTool and prints the
result the model would actually receive — so you can eyeball snippet quality,
extraction, BM25 ranking and the budget cap for a given query.

Run inside the project's conda env:

    conda run -n LLMs python scripts/try_web_search.py "python asyncio event loop"

    # snippets only (skip page fetch/extraction)
    conda run -n LLMs python scripts/try_web_search.py "rust borrow checker" --no-fetch

    # tune the knobs for one run (these mirror the WEB_SEARCH_* env vars)
    conda run -n LLMs python scripts/try_web_search.py "llm quantization" \
        --max-results 8 --top-k 4 --budget 2000

Requires the optional search extra:  pip install 'neurosurfer[search]'
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Allow running directly from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Try the web_search tool against a live query.")
    p.add_argument("query", help="The search query.")
    p.add_argument("--max-results", type=int, help="Number of search results to list.")
    p.add_argument("--top-k", type=int, help="Pages to fetch + extract.")
    p.add_argument("--budget", type=int, help="Token budget for injected page content.")
    p.add_argument("--chunk", type=int, help="Target tokens per rankable chunk.")
    p.add_argument("--no-fetch", action="store_true", help="Snippets only; skip page fetch.")
    return p.parse_args()


class _NullIO:
    """A no-op IOHandler; web_search never prompts, so nothing here is exercised."""

    async def ask(self, *a, **k):  # noqa: ANN002, ANN003
        return ""

    async def request_plan_approval(self, *a, **k):  # noqa: ANN002, ANN003
        return (True, "")

    async def request_shell_approval(self, *a, **k):  # noqa: ANN002, ANN003
        return True

    async def request_write_approval(self, *a, **k):  # noqa: ANN002, ANN003
        return "once"

    def notify(self, *a, **k):  # noqa: ANN002, ANN003
        pass


async def _main() -> int:
    args = _parse_args()

    # Knobs are read from the environment at import time, so set before importing.
    if args.max_results is not None:
        os.environ["WEB_SEARCH_MAX_RESULTS"] = str(args.max_results)
    if args.top_k is not None:
        os.environ["WEB_SEARCH_FETCH_TOP_K"] = str(args.top_k)
    if args.budget is not None:
        os.environ["WEB_SEARCH_BUDGET_TOKENS"] = str(args.budget)
    if args.chunk is not None:
        os.environ["WEB_SEARCH_CHUNK_TOKENS"] = str(args.chunk)

    from neurosurfer.llm.tokens import estimate_text_tokens
    from neurosurfer.tools.base import ToolContext
    from neurosurfer.tools.builtin.web_search import (
        BUDGET_TOKENS,
        FETCH_TOP_K,
        MAX_RESULTS,
        WebSearchTool,
    )

    tool = WebSearchTool()
    if not tool.is_enabled():
        print(
            "web_search is disabled: the 'ddgs' package is not installed.\n"
            "Install it with:  pip install 'neurosurfer[search]'",
            file=sys.stderr,
        )
        return 1

    print(
        f"query={args.query!r}  max_results={MAX_RESULTS}  "
        f"top_k={FETCH_TOP_K}  budget={BUDGET_TOKENS}  fetch={not args.no_fetch}\n"
        + "=" * 72
    )

    ctx = ToolContext(cwd=Path.cwd(), io=_NullIO())
    result = await tool.run({"query": args.query, "fetch": not args.no_fetch}, ctx)

    print(result.content)
    print("=" * 72)
    print(
        f"is_error={result.is_error}  chars={len(result.content)}  "
        f"~tokens={estimate_text_tokens(result.content)}"
    )
    return 1 if result.is_error else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

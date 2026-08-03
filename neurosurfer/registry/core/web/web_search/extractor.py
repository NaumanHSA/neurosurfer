"""HTML body extraction, text normalisation, page fetching and full-text storage."""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

import httpx

from . import config as _config

# Tags whose text is never content; removed wholesale before extraction.
_DROP_TAGS = (
    "script", "style", "noscript", "nav", "header", "footer", "aside",
    "form", "svg", "math", "button", "figure", "table",
)
# CSS selectors for site furniture / markup noise (citations, edit links, math).
_DROP_SELECTORS = (
    ".mwe-math-element", ".reference", ".mw-editsection", "sup.reference",
    ".noprint", ".toc", "#toc", ".navbox", ".sidebar", ".infobox",
)
# Block-level tags: each becomes one line; inline children are joined by spaces.
_BLOCK_TAGS = (
    "p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre", "dd", "dt", "figcaption",
)

# Citation / edit markers: [1], [12], [edit], [citation needed], [note 3], [a].
_NOISE_MARKER_RE = re.compile(
    r"\[\s*(?:\d+|[a-z]|edit|citation needed|note\s*\d*|clarification needed)\s*\]",
    re.IGNORECASE,
)


def extract_body(html: str) -> str:
    """Extract readable body text from an HTML document.

    Extraction priority:
    1. BeautifulSoup block-walker (primary) — strips structural noise, preserves
       inline elements on one line (no sentence fragmentation from ``<a>``/``<sup>``).
    2. trafilatura (fallback) — used when the block-walker finds nothing; excellent
       for complex real-world articles where BeautifulSoup yields too little.
    3. Crude tag-strip — last resort when neither bs4 nor trafilatura is installed.
    """
    if not html:
        return ""

    # --- Step 1: BeautifulSoup block-walker (primary) ---
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(list(_DROP_TAGS)):
            tag.decompose()
        for sel in _DROP_SELECTORS:
            for el in soup.select(sel):
                el.decompose()

        root = soup.find("article") or soup.find("main") or soup.body or soup

        # Walk leaf block elements; inline children are flattened with spaces so
        # sentences like "<p>BM25 is a <a>bag-of-words</a> model</p>" stay on one line.
        blocks: list[str] = []
        for el in root.find_all(_BLOCK_TAGS):
            if el.find(_BLOCK_TAGS):
                continue  # skip container blocks; their leaf children are handled separately
            txt = el.get_text(separator=" ", strip=True)
            if txt:
                blocks.append(txt)

        text = "\n\n".join(blocks) if blocks else root.get_text(separator=" ", strip=True)
        result = _normalize_text(text)
        if result:
            return result
    except Exception:  # noqa: BLE001 - bs4 or lxml missing / parse failure
        pass

    # --- Step 2: trafilatura (fallback for complex pages where bs4 finds nothing) ---
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        if extracted:
            return _normalize_text(extracted)
    except Exception:  # noqa: BLE001 - trafilatura optional / may fail on some pages
        pass

    # --- Step 3: last resort — crude tag strip (no bs4 or trafilatura) ---
    return _normalize_text(re.sub(r"(?s)<[^>]+>", " ", html))


def _normalize_text(text: str) -> str:
    """Tidy extracted text into clean paragraphs.

    Strips citation/edit markers, collapses intra-line whitespace, drops stray
    single-symbol fragment lines (leftover from inline markup / math), and
    squeezes runs of blank lines to a single paragraph break.
    """
    if not text:
        return ""
    text = _NOISE_MARKER_RE.sub("", text)

    cleaned: list[str] = []
    blank = False
    for raw in text.splitlines():
        line = re.sub(r"[​‌‍﻿]", "", raw)
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([.,;:!?)])", r"\1", line)
        if not line:
            if cleaned and not blank:
                cleaned.append("")
                blank = True
            continue
        if not re.search(r"[A-Za-z0-9]", line):
            continue
        if len(line) == 1 or (len(line) <= 2 and not line.isalnum()):
            continue
        cleaned.append(line)
        blank = False
    return "\n".join(cleaned).strip()


async def _fetch_page(client: httpx.AsyncClient, url: str) -> str:
    """Fetch a URL and return raw HTML, or '' on failure / non-HTML / oversize."""
    if not url.lower().startswith(("http://", "https://")):
        return ""
    try:
        resp = await client.get(url, timeout=_config.FETCH_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
    except Exception:  # noqa: BLE001
        return ""
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return ""
    if len(resp.content) > _config.MAX_PAGE_BYTES:
        return resp.text[:_config.MAX_PAGE_BYTES]
    return resp.text


def _store_full_text(url: str, text: str) -> Path:
    """Persist the full extracted page text to a temp file; return its path."""
    cache = Path(tempfile.gettempdir()) / "neurosurfer_websearch"
    cache.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    path = cache / f"{name}.txt"
    try:
        path.write_text(f"Source: {url}\n\n{text}", encoding="utf-8")
    except OSError:
        pass
    return path

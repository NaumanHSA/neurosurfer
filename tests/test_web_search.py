"""Offline tests for the web_search tool's pure helpers (no network).

Covers HTML body extraction, chunking, query-aware ranking, budget selection,
and the threshold logic in result rendering.
"""

from __future__ import annotations

from neurosurfer.tools.builtin.web_search import (
    WebSearchTool,
    _normalize_text,
    chunk_text,
    extract_body,
    rank_chunks,
    select_within_budget,
)


# ── extract_body ───────────────────────────────────────────────────────────────
def test_extract_body_strips_boilerplate():
    html = """
    <html><head><title>T</title><style>.x{}</style></head>
    <body>
      <nav>home about contact</nav>
      <script>var x = 1;</script>
      <article><p>The real content lives here.</p>
      <p>And a second paragraph.</p></article>
      <footer>copyright 2026</footer>
    </body></html>
    """
    body = extract_body(html)
    assert "real content lives here" in body
    assert "second paragraph" in body
    # boilerplate dropped
    assert "var x" not in body
    assert "copyright" not in body
    assert "home about contact" not in body


# ── _normalize_text ──────────────────────────────────────────────────────────
def test_normalize_text_strips_markers_and_fragments():
    raw = "\n".join(
        [
            "Real sentence one [1].",
            "  ",
            "·",  # pure-symbol line → dropped
            "[edit]",  # marker-only → dropped
            "Q",  # stray single char → dropped
            "Real sentence two [citation needed].",
        ]
    )
    out = _normalize_text(raw)
    assert "Real sentence one." in out  # marker removed, no space before period
    assert "Real sentence two." in out
    assert "[1]" not in out and "citation needed" not in out
    assert "·" not in out
    assert "\nQ\n" not in out and out != "Q"  # lone single char dropped
    # collapsed: no triple blank lines
    assert "\n\n\n" not in out


# ── chunk_text ───────────────────────────────────────────────────────────────
def test_chunk_text_splits_on_paragraphs():
    text = "\n\n".join(f"Paragraph number {i} with some filler words." for i in range(20))
    chunks = chunk_text(text, chunk_tokens=20)
    assert len(chunks) > 1
    # round-trips content (no paragraph dropped)
    joined = " ".join(chunks)
    assert "Paragraph number 0" in joined
    assert "Paragraph number 19" in joined


# ── rank_chunks ──────────────────────────────────────────────────────────────
def test_rank_chunks_surfaces_relevant_first():
    chunks = [
        "unrelated cooking recipe about pasta and tomatoes",
        "the python asyncio event loop schedules coroutines",
        "weather forecast sunny with a chance of rain",
    ]
    ranked = rank_chunks("python asyncio event loop", chunks)
    assert ranked[0] == 1  # the asyncio chunk ranks first


# ── select_within_budget ─────────────────────────────────────────────────────
def test_select_within_budget_caps_and_orders():
    chunks = [f"chunk {i} " * 50 for i in range(10)]  # each well over a tiny budget
    ranked = list(range(10))
    selected = select_within_budget(chunks, ranked, budget_tokens=120)
    assert selected  # at least one
    assert selected == sorted(selected)  # original document order
    assert len(selected) < len(chunks)  # budget actually capped it


# ── _render_content threshold ────────────────────────────────────────────────
def test_render_content_injects_small_as_is():
    pages = [("https://ex.com/a", "Short body about apples.\n\nSecond paragraph.")]
    out = WebSearchTool._render_content("apples", pages)
    assert "https://ex.com/a" in out
    assert "Short body about apples" in out
    assert "ranked excerpt" not in out  # under budget → no truncation path


def test_render_content_ranks_when_over_budget(monkeypatch):
    import neurosurfer.tools.builtin.web_search as ws

    # Patch on the config module so _render_content (which reads _config.BUDGET_TOKENS) picks it up.
    monkeypatch.setattr(ws.config, "BUDGET_TOKENS", 50)
    relevant = "quantum entanglement links particle spins instantly. " * 5
    filler = "lorem ipsum dolor sit amet filler text. " * 200
    body = filler + "\n\n" + relevant + "\n\n" + filler
    out = ws.WebSearchTool._render_content("quantum entanglement spins", [("https://ex.com/q", body)])
    assert "ranked excerpt" in out
    assert "quantum entanglement" in out
    assert "read_file for more" in out  # full text stored + path surfaced

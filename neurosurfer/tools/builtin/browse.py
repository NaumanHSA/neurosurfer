"""Headless-browser tool — open a JS-rendered page and return its readable text.

Optional: needs Playwright (``pip install 'neurosurfer[browser]'`` then
``playwright install chromium``). The tool hides itself from the pool when
Playwright is unavailable (``is_enabled`` → False), mirroring ``web_search``.

Network egress is **gated by the Task's ``network_policy``** (enforced in
``core/permissions.py``). Readable-text extraction reuses ``web_search.extract_body``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult
from .web_search import extract_body

DEFAULT_TIMEOUT_MS = 30_000
MAX_OUTPUT_CHARS = 30_000


def _playwright_available() -> bool:
    try:
        import playwright.async_api  # type: ignore  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - optional dependency
        return False


class BrowseArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL to open in a headless browser.")
    wait_ms: int = Field(
        default=0, ge=0, le=15_000, description="Extra wait after load for JS to settle."
    )
    timeout_ms: int = Field(default=DEFAULT_TIMEOUT_MS, ge=1000, le=120_000)


class BrowseTool(Tool):
    name = "browse"
    description = (
        "Open a URL in a headless browser (renders JavaScript) and return the page's "
        "readable text. Use this instead of `http` when a page needs JS to show its "
        "content. Subject to the task's network policy — you may be asked to approve it."
    )
    input_model = BrowseArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    def is_concurrency_safe(self, args: BaseModel) -> bool:
        # Launching a browser is heavy and stateful — never run several at once.
        return False

    def is_enabled(self) -> bool:
        return _playwright_available()

    async def call(self, args: BrowseArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        url = args.url.strip()
        if not url.lower().startswith(("http://", "https://")):
            return ToolResult.error(f"Unsupported URL scheme: {url!r}. Use http:// or https://.")
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except Exception:  # noqa: BLE001
            return ToolResult.error(
                "Headless browsing needs Playwright. Install with: "
                "pip install 'neurosurfer[browser]' && playwright install chromium"
            )

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                try:
                    page = await browser.new_page()
                    await page.goto(url, timeout=args.timeout_ms, wait_until="domcontentloaded")
                    if args.wait_ms:
                        await page.wait_for_timeout(args.wait_ms)
                    html = await page.content()
                finally:
                    await browser.close()
        except Exception as e:  # noqa: BLE001 - surface browser failures to the model
            return ToolResult.error(f"Browse failed: {type(e).__name__}: {e}")

        body = extract_body(html) or ""
        if not body:
            return ToolResult.ok(f"Loaded {url} but found no readable text.")
        if len(body) > MAX_OUTPUT_CHARS:
            body = body[:MAX_OUTPUT_CHARS] + "\n… [content truncated]"
        return ToolResult.ok(f"Readable content from {url}:\n\n{body}")

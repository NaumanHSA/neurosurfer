"""HTTP/API request tool — call a web endpoint and return the response.

Network egress is **gated by the Task's ``network_policy``** (default: ask per
request), enforced in ``core/permissions.py`` — not here. Uses ``httpx`` (a core
dependency), so the tool is always available. Responses are size- and char-capped
so a large payload never blows a small-context model's window.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult

MAX_RESPONSE_BYTES = 5_000_000
MAX_OUTPUT_CHARS = 30_000
DEFAULT_TIMEOUT = 30.0
_READ_METHODS = {"GET", "HEAD"}
_TEXTUAL = ("json", "text", "xml", "html", "javascript", "csv", "yaml")

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]


class HttpArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL to request.")
    method: HttpMethod = Field(default="GET", description="HTTP method.")
    headers: dict[str, str] = Field(default_factory=dict, description="Request headers.")
    params: dict[str, str] = Field(default_factory=dict, description="URL query parameters.")
    json_body: Any | None = Field(
        default=None, description="JSON body (sent as application/json)."
    )
    body: str | None = Field(
        default=None, description="Raw text body (ignored when json_body is set)."
    )
    timeout: float = Field(default=DEFAULT_TIMEOUT, ge=1, le=300)


class HttpTool(Tool):
    name = "http"
    description = (
        "Make an HTTP request to a URL and return the status, key response headers, and "
        "body (JSON pretty-printed; long text truncated). Use it to call REST/JSON APIs or "
        "fetch raw resources. Subject to the task's network policy — you may be asked to "
        "approve the request."
    )
    input_model = HttpArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return getattr(args, "method", "GET") in _READ_METHODS

    async def call(self, args: HttpArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        url = args.url.strip()
        if not url.lower().startswith(("http://", "https://")):
            return ToolResult.error(f"Unsupported URL scheme: {url!r}. Use http:// or https://.")

        headers = dict(args.headers)
        content: bytes | None = None
        if args.json_body is not None:
            try:
                content = json.dumps(args.json_body).encode("utf-8")
            except (TypeError, ValueError) as e:
                return ToolResult.error(f"json_body is not JSON-serialisable: {e}")
            headers.setdefault("Content-Type", "application/json")
        elif args.body is not None:
            content = args.body.encode("utf-8")

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.request(
                    args.method,
                    url,
                    headers=headers or None,
                    params=args.params or None,
                    content=content,
                    timeout=args.timeout,
                )
        except httpx.TimeoutException:
            return ToolResult.error(f"Request timed out after {args.timeout}s: {args.method} {url}")
        except httpx.HTTPError as e:
            return ToolResult.error(f"Request failed: {type(e).__name__}: {e}")

        return ToolResult.ok(self._render(args.method, url, resp))

    @staticmethod
    def _render(method: str, url: str, resp: httpx.Response) -> str:
        ctype = resp.headers.get("content-type", "")
        head = (
            f"{method} {url}\n"
            f"→ {resp.status_code} {resp.reason_phrase}\n"
            f"Content-Type: {ctype or '(none)'}"
        )
        raw = resp.content or b""
        if len(raw) > MAX_RESPONSE_BYTES:
            return f"{head}\n\n[response too large: {len(raw)} bytes — not shown]"
        if ctype and not any(t in ctype for t in _TEXTUAL):
            return f"{head}\n\n[{len(raw)} bytes of non-text content not shown]"

        text = resp.text
        if "json" in ctype:
            try:
                text = json.dumps(resp.json(), indent=2, ensure_ascii=False)
            except ValueError:
                pass
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "\n… [body truncated]"
        return f"{head}\n\n{text or '(empty body)'}"

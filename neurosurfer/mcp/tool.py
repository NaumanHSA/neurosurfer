"""``McpTool`` — wraps one tool exposed by an MCP server as a neurosurfer :class:`Tool`.

Unlike built-in tools, an MCP tool's input contract is a **raw JSON Schema** supplied
by the server, not a pydantic model. So this adapter overrides :attr:`schema` to pass
that schema through verbatim and :meth:`parse_args` to skip the pydantic round-trip —
the server is the authoritative validator. MCP *tool annotations* (``readOnlyHint`` /
``destructiveHint`` / ``idempotentHint`` / ``openWorldHint``) map onto the behaviour
flags the agent loop uses for scheduling and permission gating.

Calls go through the live :class:`mcp.ClientSession` owned by the :class:`McpManager`.
Transport failures are *returned* as ``ToolResult(is_error=True)``, never raised, so the
agent loop self-corrects — same contract as every other tool.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..llm.types import ToolSchema
from ..tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Tool as MCPToolDef


class _PassthroughArgs(BaseModel):
    """Permissive args holder.

    MCP validates inputs server-side, so we accept any extra keys here and forward
    the original dict unchanged. We keep a pydantic instance only so the rest of the
    engine (``is_read_only(args)``, permission checks) has the object it expects.
    """

    model_config = {"extra": "allow"}


class McpTool(Tool):
    is_mcp = True
    input_model = _PassthroughArgs

    def __init__(
        self,
        *,
        session: ClientSession,
        server_name: str,
        remote_name: str,
        exposed_name: str,
        description: str,
        input_schema: dict[str, Any],
        annotations: Any = None,
    ) -> None:
        self._session = session
        self._server_name = server_name
        self._remote_name = remote_name  # name the server knows the tool by
        self.name = exposed_name  # name the agent calls (possibly prefixed)
        self.description = description or f"{remote_name} (via {server_name})"
        self._input_schema = input_schema or {"type": "object", "properties": {}}
        self._ann = annotations
        # Capture the event loop this session was created on. When called from a
        # different loop (e.g. a graph-executor thread or a notebook cell's
        # asyncio.run()), we bridge back here automatically — the user never sees it.
        try:
            self._home_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._home_loop = None

    # ── schema: pass the server's JSON Schema through unchanged ─────────────────
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self._input_schema,
        )

    def parse_args(self, raw: dict[str, Any]) -> BaseModel:
        # No pydantic coercion — keep exactly what the model produced; the server
        # validates. We still hand back a BaseModel for the loop's type contract.
        return _PassthroughArgs.model_validate(raw or {})

    # ── behaviour flags from MCP annotations ────────────────────────────────────
    def _hint(self, attr: str, default: bool) -> bool:
        val = getattr(self._ann, attr, None) if self._ann is not None else None
        return default if val is None else bool(val)

    def is_read_only(self, args: BaseModel) -> bool:
        return self._hint("readOnlyHint", False)

    def is_destructive(self, args: BaseModel) -> bool:
        # Only meaningful for non-read-only tools per the MCP spec; default True
        # (conservative) unless the server says otherwise.
        if self.is_read_only(args):
            return False
        return self._hint("destructiveHint", True)

    def is_concurrency_safe(self, args: BaseModel) -> bool:
        # Safe to parallelise only if read-only and not touching an open world.
        return self.is_read_only(args) and not self._hint("openWorldHint", True)

    def progress_message(self, args: dict[str, Any]) -> str:
        return f"{self._server_name}: {self._remote_name}…"

    # ── call the remote tool ────────────────────────────────────────────────────
    async def _do_call(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        """The actual MCP round-trip — must run on self._home_loop."""
        payload = _to_payload(args)
        try:
            result = await self._session.call_tool(self._remote_name, payload)
        except Exception as e:  # noqa: BLE001 - surface as a tool error, never raise
            return ToolResult.error(
                f"MCP call to '{self._remote_name}' on '{self._server_name}' failed: "
                f"{type(e).__name__}: {e}"
            )
        text = _flatten_content(getattr(result, "content", None))
        if not text and getattr(result, "structuredContent", None) is not None:
            text = str(result.structuredContent)
        return ToolResult(content=text or "(no content)", is_error=bool(getattr(result, "isError", False)))

    async def call(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        # Fast path: no home loop recorded, or same loop — call directly.
        if self._home_loop is None:
            return await self._do_call(args, ctx)
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is not None and current is not self._home_loop:
            # Called from a different loop (graph executor thread, asyncio.run() in
            # a worker thread, etc.). Route the coroutine back to the session's home
            # loop so the transport sees the right event loop.
            fut = asyncio.run_coroutine_threadsafe(self._do_call(args, ctx), self._home_loop)
            return await asyncio.wrap_future(fut)
        return await self._do_call(args, ctx)

    @classmethod
    def from_def(
        cls,
        *,
        session: ClientSession,
        server_name: str,
        definition: MCPToolDef,
        exposed_name: str,
    ) -> McpTool:
        return cls(
            session=session,
            server_name=server_name,
            remote_name=definition.name,
            exposed_name=exposed_name,
            description=definition.description or "",
            input_schema=dict(definition.inputSchema or {}),
            annotations=getattr(definition, "annotations", None),
        )


def _to_payload(args: BaseModel) -> dict[str, Any]:
    if isinstance(args, _PassthroughArgs):
        # extra="allow" stashes the real fields in __pydantic_extra__.
        extra = getattr(args, "__pydantic_extra__", None)
        if extra:
            return dict(extra)
    return args.model_dump(exclude_none=False)


def _flatten_content(content: Any) -> str:
    """Render MCP tool-result content blocks into a single text string.

    Text blocks are concatenated; non-text blocks (images, embedded resources) are
    described compactly so the model knows something was returned without us inventing
    a binary channel (that arrives with the Vision phase).
    """
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(getattr(block, "text", "") or "")
        elif btype == "image":
            mime = getattr(block, "mimeType", "image")
            parts.append(f"[image content: {mime} — not yet supported as input]")
        elif btype == "resource":
            res = getattr(block, "resource", None)
            uri = getattr(res, "uri", "") if res is not None else ""
            parts.append(f"[resource: {uri}]")
        else:
            parts.append(f"[{btype or 'unknown'} content]")
    return "\n".join(p for p in parts if p)

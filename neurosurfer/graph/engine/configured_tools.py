"""Tools with their author-set configuration applied.

Sibling of :mod:`bound_tools`, and the difference between them is the whole
reason both exist:

- a **bound argument** is a value the model must never supply — a credential —
  so `BoundTool` removes it from the schema and substitutes it;
- a **setting** is a value the model was never asked for in the first place. It
  is not in `input_model`, so there is nothing to remove. What it does instead is
  change *where the call lands*.

## What this fixes

`write_file` resolved `path` against `ToolContext.cwd`, and in the gateway that
is the process's working directory — the neurosurfer checkout. So an agent asked
to write a report wrote into the server's own source tree, and no surface
anywhere could have said otherwise: `path` is the model's to choose, and a
directory is not a parameter.

Now a node says `tool_settings: {write_file: {root: /home/nomi/reports}}` and the
tool is handed a context whose `cwd` **is** that directory. `write_file` itself is
unchanged — it already resolves through `resolve_path(ctx.cwd, …)`, which is
exactly the seam this needs.

## Confinement

The root is a boundary, not a default. Every argument the tool declares in
`path_inputs` is resolved under it and refused outside it, before `call` runs —
so `../../.env` and `/etc/passwd` fail the same way, with a message naming the
directory the step is allowed to work in.

Refused *as a tool result*, never as an exception: an agent that asked for an
out-of-scope path should read why and try a path inside the root, which is the
whole self-correction contract tool errors are built on. A raised exception would
end the node instead.

**Declared, not sniffed.** `search` takes a `pattern` and a `path`; only one is a
place on disk. A wrapper that guessed from the value would confine whatever
happened to look path-shaped, which is a rule that changes with the data.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from neurosurfer.tools.base import Tool, ToolContext, ToolPool, ToolResult
from neurosurfer.tools.utils import PathOutsideRoot, resolve_within

__all__ = ["ConfiguredTool", "configure_pool", "tool_settings_for"]


class ConfiguredTool(Tool):
    """*inner*, run inside the directory its author configured."""

    def __init__(self, inner: Tool, settings: dict[str, Any]) -> None:
        self._inner = inner
        self._settings = dict(settings)
        self.name = inner.name
        self.description = inner.description
        self.input_model = inner.input_model
        self.is_mcp = getattr(inner, "is_mcp", False)

    @property
    def root(self) -> Path | None:
        """The configured directory, or None when this tool has no root setting."""
        field = getattr(self._inner, "root_setting", "") or ""
        raw = str(self._settings.get(field) or "").strip() if field else ""
        return Path(raw).expanduser() if raw else None

    # ── the call ────────────────────────────────────────────────────────────
    async def call(self, args: Any, ctx: ToolContext) -> ToolResult:
        root = self.root
        if root is None:
            return await self._inner.call(args, ctx)

        # Checked before the call, so a refusal costs nothing and names the
        # argument that caused it rather than whatever the tool failed on later.
        for field in getattr(self._inner, "path_inputs", frozenset()):
            raw = getattr(args, field, None)
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                resolve_within(root, raw)
            except PathOutsideRoot as e:
                return ToolResult.error(f"{self.name}: {e}")

        # A relative path now lands under the root, because the root *is* the cwd.
        # `dataclasses.replace` rather than mutation: the context is shared with
        # every other tool on the step, and scoping one must not scope them all.
        return await self._inner.call(args, replace(ctx, cwd=root))

    # ── everything else is the inner tool's business ────────────────────────
    @property
    def schema(self) -> Any:
        return self._inner.schema

    def parse_args(self, raw: dict[str, Any]) -> Any:
        return self._inner.parse_args(raw)

    def is_read_only(self, args: Any) -> bool:
        return self._inner.is_read_only(args)

    def is_concurrency_safe(self, args: Any) -> bool:
        return self._inner.is_concurrency_safe(args)

    def is_destructive(self, args: Any) -> bool:
        return self._inner.is_destructive(args)

    def is_enabled(self) -> bool:
        return self._inner.is_enabled()

    def progress_message(self, args: dict[str, Any]) -> str:
        return self._inner.progress_message(args)

    def __getattr__(self, item: str) -> Any:
        # Manifest facets (`capabilities`, `runtime`, `secret_inputs`,
        # `settings_model`, …) come from the real tool.
        return getattr(self._inner, item)


def tool_settings_for(node: Any, tool_name: str) -> dict[str, Any]:
    """What *node* configured for *tool_name*. Empty when it configured nothing.

    Keyed by tool because a node holds several — an agent with `write_file` and
    `read_file` legitimately writes to one directory and reads from another, and
    a single flat settings dict could not say so.
    """
    settings = getattr(node, "tool_settings", None) or {}
    if not isinstance(settings, dict):
        return {}
    value = settings.get(tool_name)
    return dict(value) if isinstance(value, dict) else {}


def configure_pool(pool: ToolPool, settings: dict[str, dict[str, Any]]) -> ToolPool:
    """*pool* with each tool's own settings applied.

    Only tools that were actually configured are wrapped, so a step holding four
    tools and configuring one leaves the other three exactly as they were.
    """
    if not settings:
        return pool
    out: list[Tool] = []
    for tool in pool.all():
        own = settings.get(tool.name)
        out.append(ConfiguredTool(tool, own) if isinstance(own, dict) and own else tool)
    return ToolPool(out)

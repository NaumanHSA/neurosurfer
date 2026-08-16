"""Tools with some arguments already decided, for `react` nodes.

A `tool` node states its whole call in `tool_args`, and that is the one place a
`${SECRET}` is substituted. A `react` node states none of it — its model composes
each call at run time — so until now **a react node could not use a credential at
all**.

That gap was invisible for a long time and then became the main event. Asked to
audit a SQL Server database, the planner made two correct decisions:

- composing SQL needs a model, so the step is a `react` node;
- the step needs a connection string, so it declares `secrets:`.

and the validator answered *"declares secrets … but is a 'react' node, which has
no tool_args to use them in"*. Both decisions were right and the engine could not
honour them together — which rules out the commonest integration shape there is:
query a database, call an authenticated API, search a private index.

So `tool_args` on a react node means **bound arguments**: values the engine merges
into every call the model makes of that tool, rather than the whole call.

    kind: react
    tools: [sql]
    secrets: [AIA_DB_URL]
    tool_args: {dsn: "${AIA_DB_URL}"}   # bound; the model composes only `query`

Two properties make this safe rather than merely convenient:

- **The bound parameter is removed from the schema the model sees.** It is not
  hidden, or defaulted, or overridden after the fact — the model is never offered
  it, so it cannot supply one, cannot hallucinate one, and cannot be persuaded to
  echo one back. A credential stays as invisible here as it is in a `tool` node.
- **Bound values win.** If a model somehow produces the key anyway (a
  passthrough-schema MCP tool accepts anything), the bound value replaces it
  rather than merging under it.
"""

from __future__ import annotations

from typing import Any

from neurosurfer.tools.base import Tool, ToolContext, ToolPool, ToolResult, ToolSchema

__all__ = ["BoundTool", "bind_pool"]


class BoundTool(Tool):
    """*inner* with some arguments already supplied and hidden from the model."""

    def __init__(self, inner: Tool, bound: dict[str, Any]) -> None:
        self._inner = inner
        self._bound = dict(bound)
        self.name = inner.name
        self.description = inner.description
        self.input_model = inner.input_model
        self.is_mcp = getattr(inner, "is_mcp", False)

    # ── what the model is offered ───────────────────────────────────────────
    @property
    def schema(self) -> ToolSchema:
        base = self._inner.schema
        raw = dict(base.input_schema or {})
        props = {k: v for k, v in (raw.get("properties") or {}).items()
                 if k not in self._bound}
        required = [r for r in (raw.get("required") or []) if r not in self._bound]
        return ToolSchema(
            name=base.name,
            description=base.description,
            input_schema={**raw, "properties": props, "required": required},
        )

    # ── what actually gets called ───────────────────────────────────────────
    def parse_args(self, raw: dict[str, Any]) -> Any:
        # Bound last: a model that produced the key anyway does not get to win.
        return self._inner.parse_args({**(raw or {}), **self._bound})

    async def call(self, args: Any, ctx: ToolContext) -> ToolResult:
        return await self._inner.call(args, ctx)

    # ── everything else is the inner tool's business ────────────────────────
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
        # Manifest facets (`capabilities`, `runtime`, `secret_inputs`, …) and
        # anything else a caller reads off the tool come from the real one.
        return getattr(self._inner, item)


def bind_pool(pool: ToolPool, bound: dict[str, Any]) -> ToolPool:
    """*pool* with *bound* applied to every tool that declares those parameters.

    Only tools whose schema actually names a bound key are wrapped, so binding
    `dsn` for a node holding both `sql` and `write_file` leaves the second
    alone rather than passing it an argument it has never heard of.
    """
    if not bound:
        return pool
    out: list[Tool] = []
    for tool in pool.all():
        try:
            props = set((tool.schema.input_schema or {}).get("properties") or {})
        except Exception:  # noqa: BLE001 - an unreadable schema binds nothing
            props = set()
        applicable = {k: v for k, v in bound.items() if k in props}
        out.append(BoundTool(tool, applicable) if applicable else tool)
    return ToolPool(out)

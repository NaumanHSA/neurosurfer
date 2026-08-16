"""What a workflow needs supplied before it can run.

A registry entry publishes its configuration — `SQLITE_DB_PATH (required)`,
`SQLITE_TIMEOUT (optional)` — and until now that was read once during resolution
and thrown away. A registered workflow therefore recorded nothing about what had
to be set for it, and the omission surfaced as a connection error on the first
node: the least informative place and the latest possible moment.

**Derived, never stored.** The same reasoning as the capability manifest: a copy
in the package would be a second source of truth that drifts the moment a server
is reconfigured. Both facts this reads — the `${VAR}`s in a server's config and
the `secrets:` on a node — are already authoritative somewhere, so this asks them.

Enabled servers are included whether or not the workflow names one of their tools,
because starting a run connects every enabled server (`ensure_mcp_tools`); one of
them missing a value fails the run regardless of which tools it was going to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Requirement", "missing_requirements", "workflow_requirements"]


@dataclass(frozen=True)
class Requirement:
    """One stored value this workflow needs, and who is asking for it."""

    name: str
    #: `node:<id>` or `server:<name>` — enough to answer "why does it want this".
    source: str
    satisfied: bool = False
    #: Why a value that IS set still cannot be used — "is not a SQLAlchemy
    #: connection URL". Set is not the same as usable, and conflating them let a
    #: hostname left over from an abandoned naming scheme satisfy a requirement
    #: silently: the user was never asked, and the run failed on a malformed URL.
    problem: str = ""
    #: What a person needs in order to supply it, from the tool that wants it.
    help: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source,
                "satisfied": self.satisfied, "problem": self.problem,
                "help": self.help}


def _node_secret_refs(node: Any) -> list[str]:
    """Stored values this node will actually reach for.

    Read from the `${NAME}` references inside `tool_args` — the only place one is
    ever substituted — rather than from the node's `secrets:` list.

    The two differ, and the difference is the point. `secrets:` is what the author
    *declared*; `tool_args` is what the workflow will *use*. A live build declared
    seven names on one node (`…_DRIVER`, `…_ENCRYPTION`, `…_TRUST_CERTIFICATE`),
    invented from an imagined connection model, and the user was asked to supply
    all seven before a workflow that referenced none of them could be tested.

    `secrets:` is still the authorisation: a reference the node did not declare is
    left unexpanded by the executor, so it would not be filled even if supplied,
    and asking for it would be asking for something that cannot help. Intersecting
    the two means we ask for exactly what will be substituted.
    """
    declared = {str(s) for s in (getattr(node, "secrets", None) or ())}
    if not declared:
        return []
    from neurosurfer.graph.engine.secrets import secret_refs

    def _walk(value: Any) -> list[str]:
        if isinstance(value, str):
            return secret_refs(value)
        if isinstance(value, dict):
            return [n for v in value.values() for n in _walk(v)]
        if isinstance(value, (list, tuple)):
            return [n for v in value for n in _walk(v)]
        return []

    used = _walk(getattr(node, "tool_args", None) or {})
    return [n for n in dict.fromkeys(used) if n in declared]


def _secret_checker(node: Any) -> Any:
    """`(secret_name) -> (problem, help)` for the tools this node calls.

    The tool is the only thing that knows what a usable value looks like — a
    schema says `dsn` is a string, not that the string must parse as a connection
    URL. Which parameter a secret feeds is read from `tool_args`: `${DB_URL}`
    written into `dsn` means `DB_URL` is checked as a `dsn`.
    """
    names = list(getattr(node, "tools", None) or ())
    if not names:
        return None
    try:
        from neurosurfer.tools.registry import all_tools

        tools = [t for t in all_tools() if t.name in names]
    except Exception:  # noqa: BLE001 - a checkless requirement is the old behaviour
        return None
    if not tools:
        return None

    # secret name → the parameter it is written into, from `${NAME}` in tool_args.
    from neurosurfer.graph.engine.secrets import secret_refs

    slots: dict[str, str] = {}
    for param, value in (getattr(node, "tool_args", None) or {}).items():
        if isinstance(value, str):
            for ref in secret_refs(value):
                slots.setdefault(ref, str(param))

    def _check(secret_name: str) -> tuple[str, str]:
        from neurosurfer.mcp.credentials import lookup

        value = lookup(secret_name)[0]
        if not value:
            return "", _help_for(tools, slots.get(secret_name))
        param = slots.get(secret_name)
        for tool in tools:
            if param and param not in (getattr(tool, "secret_inputs", None) or ()):
                continue
            try:
                problem = tool.check_secret(param or secret_name, value)
            except Exception:  # noqa: BLE001 - a faulty check never blocks a run
                continue
            if problem:
                return problem, _help_for([tool], param)
        return "", ""

    return _check


def _help_for(tools: list[Any], param: str | None) -> str:
    """The first tool that says how to obtain the value for *param*."""
    for tool in tools:
        if param and param not in (getattr(tool, "secret_inputs", None) or ()):
            continue
        text = str(getattr(tool, "credential_help", "") or "").strip()
        if text:
            return text
    return ""


def _nodes(graph: Any) -> list[Any]:
    """Every node, including those nested inside loop/map bodies.

    A body is where the work that actually calls something usually lives, so a
    checklist stopping at the top level would miss exactly the nodes with
    credentials.
    """
    out: list[Any] = []
    stack = list(getattr(graph, "nodes", []) or [])
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(getattr(node, "body", None) or [])
    return out


def workflow_requirements(pkg: Any, *, store: Any = None,
                          available: dict[str, str] | None = None) -> list[Requirement]:
    """Every stored value *pkg* needs, marked with whether it is already set.

    Satisfaction goes through the same lookup a connection uses, so a value in the
    gateway's environment counts exactly as much here as it does there — a
    checklist that ignored `.env` would ask for things that already work.
    """
    from neurosurfer.mcp.credentials import lookup

    seen: dict[str, Requirement] = {}

    def _have(name: str) -> bool:
        if available is not None:
            return name in available
        return bool(lookup(name)[1])

    def add(name: str, source: str, *, checker: Any = None) -> None:
        if not name or name in seen:
            return
        have = _have(name)
        problem, help_text = "", ""
        if checker is not None:
            problem, help_text = checker(name)
            if problem:
                # Set, and unusable. Reported as unsatisfied so the run refuses
                # and the prompt asks again — silently accepting it is how a
                # stale value reaches a driver and fails as someone else's error.
                have = False
        seen[name] = Requirement(name=name, source=source, satisfied=have,
                                 problem=problem, help=help_text)

    for node in _nodes(getattr(pkg, "graph", None)):
        refs = _node_secret_refs(node)
        if not refs:
            continue
        checker = _secret_checker(node)
        for name in refs:
            add(name, f"node:{node.id}", checker=checker)

    try:
        from neurosurfer.config.mcp import McpStore

        for cfg in (store or McpStore.default()).list():
            if not cfg.enabled:
                continue
            for name in cfg.required_vars():
                add(name, f"server:{cfg.name}")
    except Exception:  # noqa: BLE001 - a store fault must not break a listing
        pass

    return sorted(seen.values(), key=lambda r: r.name)


def missing_requirements(pkg: Any, *, store: Any = None,
                         available: dict[str, str] | None = None) -> list[Requirement]:
    """Only the ones that are not set — what a run should refuse on."""
    return [r for r in workflow_requirements(pkg, store=store, available=available)
            if not r.satisfied]

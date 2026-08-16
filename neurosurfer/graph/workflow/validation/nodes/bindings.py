"""Binding a tool's arguments to what an upstream step actually produces.

## The shape these rules exist for

    agent  (mode: structured, output_schema: {sql: string})
      ↓
    query  (kind: tool, tool_args: {query: "{agent[sql]}"})

The agent decides the *arguments*; the tool node still always runs. That is the
deterministic half of tool use, and the half a graph can be checked for — unlike
an attached tool, where the model composes the call at run time and there is
nothing to check until it does.

## What is deliberately **not** checked

**A parameter the model was never meant to supply.** A credential (`dsn`, an API
key) comes from the secrets store, and a value bound literally in `tool_args` is
already decided. Neither should ever be reported as "missing from the agent's
output" — the agent was never going to produce them, and saying so would push an
author toward putting a password in a prompt. `_decided_elsewhere` is that
exclusion, and it is the first thing every rule here applies.

**An optional parameter nobody set.** That is what optional means. The severity
split below is about a binding that *will not resolve*, never about absence:
a required parameter bound to a field that does not exist cannot be called at
all, where an optional one merely goes missing.
"""

from __future__ import annotations

import re
from typing import Any

from ..models import Severity, ValidationIssue
from ..registry import node_rule
from ..tool_schema import tool_input_schema

#: `{agent[sql]}` and `{agent.sql}` mean the same thing — see
#: `engine.templates._Reachable`. Both spellings have to be recognised here or a
#: rule would check one graph and quietly skip an identical one.
_REACH = re.compile(r"\{([A-Za-z_]\w*)(?:\[([^\]\{\}]+)\]|\.(\w+))\}")


def _reaches(text: str) -> list[tuple[str, str]]:
    """Every `(node_id, field)` a template string reaches into."""
    out: list[tuple[str, str]] = []
    for m in _REACH.finditer(text or ""):
        root, subscript, attr = m.group(1), m.group(2), m.group(3)
        field = (subscript or attr or "").strip().strip("'\"")
        # `{nodes.agent}` names a namespace, not a field of a node.
        if root in {"nodes", "inputs", "vars", "state"}:
            continue
        if field:
            out.append((root, field))
    return out


def _arg_strings(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Every string inside `tool_args`, with the argument name it sits under."""
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for k, v in value.items():
            out.extend(_arg_strings(v, path or str(k)))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(_arg_strings(v, path))
        return out
    return []


def _credential_params(tool_name: str) -> set[str]:
    """Parameters the tool says carry a credential."""
    try:
        from neurosurfer.tools.registry import all_tools  # noqa: PLC0415

        tool = next((t for t in all_tools() if t.name == tool_name), None)
        return set(getattr(tool, "secret_inputs", None) or ())
    except Exception:  # noqa: BLE001 - unreadable registry excludes nothing
        return set()


def _decided_elsewhere(node: Any, tool_name: str) -> set[str]:
    """Arguments no upstream step is expected to produce.

    A credential comes from the secrets store; a literal in `tool_args` is
    already settled. Both are correct configurations, and neither is the agent's
    job — so neither may be reported against the agent's output.
    """
    settled = _credential_params(tool_name)
    for name, text in _arg_strings(getattr(node, "tool_args", None) or {}):
        if "{" not in text:  # a literal value, decided by the author
            settled.add(name)
        if "${" in text:  # `${NAME}` — a stored secret
            settled.add(name)
    return settled


def _declared_fields(schema: Any) -> set[str] | None:
    """The property names of a JSON Schema, or None if it does not declare any."""
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    return set(props) if isinstance(props, dict) else None


@node_rule(kinds=("tool", "react", "base"), severity=Severity.ERROR)
def a_binding_reaches_into_a_shaped_output(node, ctx, report) -> None:
    """Reaching into a step that never says what shape it returns.

    `{agent[sql]}` asks a step for one field of its answer. A step with no output
    shape answers in prose, and prose has no fields — so the placeholder resolves
    to nothing and is passed through to the tool verbatim.
    """
    for arg, text in _arg_strings(getattr(node, "tool_args", None) or {}):
        # The field is deliberately unused here: this rule asks only whether the
        # *source* declares a shape at all. Which field was named is the next
        # rule's question, and answering both here would report one mistake twice.
        for root, _field in _reaches(text):
            source = ctx.by_id.get(root)
            if source is None:
                continue  # an unknown id is `_check_tool_args_templates`'s to report
            if _declared_fields(getattr(source, "output_schema", None)) is not None:
                continue
            report.add(ValidationIssue(
                severity=Severity.ERROR,
                kind="binding.source_has_no_shape",
                node_id=node.id,
                subject=arg,
                message=(
                    f"This step takes '{arg}' from a field of '{root}', but "
                    f"'{root}' does not describe what it returns."
                ),
                suggestion=(
                    f"Set '{root}' to return a described shape, or take its whole "
                    f"answer instead."
                ),
                detail=f"tool_args.{arg} = {text!r}; '{root}' has no output_schema",
            ))


@node_rule(kinds=("tool", "react", "base"), severity=Severity.ERROR)
def a_binding_names_a_field_that_exists(node, ctx, report) -> None:
    """Reaching for a field the upstream step does not produce.

    Severity follows the *parameter*, not the field: a required argument bound to
    a field that does not exist cannot be sent at all, so the call is impossible.
    An optional one simply goes missing, which is worth saying and not worth
    blocking a run for.
    """
    tools = list(getattr(node, "tools", None) or [])
    if not tools:
        return
    schema = tool_input_schema(tools[0]) or {}
    required = {str(r) for r in (schema.get("required") or [])}
    accepted = _declared_fields(schema)
    settled = _decided_elsewhere(node, tools[0])

    for arg, text in _arg_strings(getattr(node, "tool_args", None) or {}):
        if arg in settled:
            continue  # a credential or a literal — never the agent's to produce
        # An argument the tool does not take at all is one problem, and
        # `every_bound_argument_belongs_to_a_tool` states it better. Reporting
        # where its value *would have* come from as well gives two messages for
        # one mistake, and the second is about a value nothing will ever read.
        if accepted is not None and arg not in accepted:
            continue
        for root, field in _reaches(text):
            source = ctx.by_id.get(root)
            if source is None:
                continue
            fields = _declared_fields(getattr(source, "output_schema", None))
            if fields is None or field in fields:
                continue  # unshaped is the rule above's; a hit is fine
            blocking = arg in required
            have = ", ".join(sorted(fields)) or "nothing"
            report.add(ValidationIssue(
                severity=Severity.ERROR if blocking else Severity.WARNING,
                kind="binding.unknown_field",
                node_id=node.id,
                subject=arg,
                message=(
                    f"This step takes '{arg}' from '{field}' of '{root}', which "
                    f"'{root}' does not return."
                    + ("" if blocking else " It will be left out.")
                ),
                suggestion=f"'{root}' returns: {have}.",
                detail=f"tool_args.{arg} = {text!r}",
            ))


@node_rule(kinds=("tool", "react", "base"), severity=Severity.WARNING)
def every_bound_argument_belongs_to_a_tool(node, ctx, report) -> None:
    """An argument no tool on this step accepts.

    The engine already does the right thing at run time — `bind_pool` applies a
    bound value only to tools whose schema names it, so a `path` meant for
    `write_file` is not handed to `sql`. What it cannot do is tell you that a name
    matched *nothing*, which is a value you configured and no tool will ever see.
    """
    tools = list(getattr(node, "tools", None) or [])
    args = getattr(node, "tool_args", None) or {}
    if not tools or not isinstance(args, dict):
        return

    accepted: set[str] = set()
    unreadable = False
    for name in tools:
        schema = tool_input_schema(name)
        if schema is None:
            unreadable = True  # an MCP tool we cannot read accepts anything
            continue
        accepted |= _declared_fields(schema) or set()
    if unreadable:
        return

    for arg in args:
        if arg in accepted:
            continue
        report.add(ValidationIssue(
            severity=Severity.WARNING,
            kind="binding.unused_argument",
            node_id=node.id,
            subject=str(arg),
            message=(
                f"'{arg}' is set here but no tool on this step takes it, so it "
                f"will be ignored."
            ),
            suggestion=f"Remove it, or check the name against: {', '.join(tools)}.",
        ))

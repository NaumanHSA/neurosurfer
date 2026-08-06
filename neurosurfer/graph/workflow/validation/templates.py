"""Placeholders, and everything else resolved against a node's runtime scope.

``{summary}`` in a prompt is rendered against the scope the executor will build
at that point in the graph. A name that resolves to nothing is left as written —
so nine literal characters reach the model, the run completes, and the only sign
of the fault is a warning on stdout. These checks rebuild that scope statically
so it lands before the run instead of inside it.

**One rule, not several, because the scope is cumulative.** A container body sees
its parent's names plus its own bindings, so the walk has to descend level by
level; a per-node rule would have to recompute the scope each time, which is how
two checks come to disagree about what is in it. The per-node checks that need a
scope — secrets, tool arguments, an output node's value — hang off that walk.
"""

from __future__ import annotations

import difflib
import re
import string
from typing import Any, NamedTuple

from .context import body_nodes as _body_nodes
from .models import Severity, ValidationIssue, ValidationReport
from .registry import graph_rule
from .tool_schema import tool_input_schema as _tool_input_schema

#: The fields whose `{placeholders}` are checked.
#:
#: `instructions` was **missing** from this tuple, so the one field new nodes
#: actually set was the one field never template-validated: a typo in `goal`
#: was an error, the same typo in `instructions` was silence. Survivable while
#: every graph input was recited to every node anyway — the value still reached
#: the model, just not where the author put it. Not survivable now that a
#: placeholder is the only way a node sees graph state.
_TEMPLATE_FIELDS = ("instructions", "purpose", "goal", "expected_result")

_FORMATTER = string.Formatter()

# The real format-spec grammar (fill/align/sign/width/precision/type). Prose after
# a colon â€” `{status: ok}` â€” means the author meant the brace literally, not that
# they wanted a variable formatted.
_FORMAT_SPEC_RE = re.compile(r"^[<>=^+\- #0-9,._bcdeEfFgGnosxX%]*$")

#: Namespaces the template renderer understands, mirroring the expression
#: language: `{nodes.x}` / `{inputs.x}` / `{vars.x}` name the same values as
#: `{x}`. See `engine.templates.with_namespaces`.
_TEMPLATE_NAMESPACES = frozenset({"nodes", "inputs", "vars"})

class _Placeholder(NamedTuple):
    """One `{...}` in a template.

    ``root`` is the name ``str.format`` looks up — the part before any ``.attr``
    or ``[key]``. ``literal`` marks a placeholder that is not a variable reference
    at all (`{}`, `{0}`, `{"a": 1}`): those raise too, but the fix is to escape
    the brace rather than to wire something up.
    """

    root: str
    raw: str
    literal: bool


class _Scope(NamedTuple):
    """The names a node's templates can resolve, split by where they come from.

    Mirrors `interp_scope = {**graph_inputs, **dependency_results, **state.vars}`
    in the executor. ``maybe`` holds names that exist in the graph but are only
    set when some *other* branch ran — real at runtime sometimes, so a warning
    rather than an error.
    """

    inputs: frozenset[str]
    vars: frozenset[str]
    deps: frozenset[str]
    maybe: frozenset[str]


def _parse_placeholders(text: str) -> list[_Placeholder] | None:
    """Every placeholder in *text*, or ``None`` if the braces don't balance."""
    try:
        fields = [(name, spec) for _lit, name, spec, _conv in _FORMATTER.parse(text)
                  if name is not None]
    except ValueError:
        return None  # a lone '{' or '}' — format() raises before looking anything up

    out: list[_Placeholder] = []
    for name, spec in fields:
        parts = name.split(".")
        # `{nodes.summarise}` is the expression language's way of naming the same
        # thing `{summarise}` names, and the renderer accepts both. Unwrap it here
        # so validation agrees: checking the literal root would report `nodes` as
        # an unknown variable, which is how a build spent fifteen rounds being
        # told the correct name and writing the namespaced one again.
        if len(parts) > 1 and parts[0] in _TEMPLATE_NAMESPACES:
            parts = parts[1:]
        root = parts[0].split("[")[0]
        prose_spec = bool(spec) and not _FORMAT_SPEC_RE.match(spec)
        out.append(_Placeholder(
            root=root,
            raw=name if not spec else f"{name}:{spec}",
            literal=not root.isidentifier() or prose_spec,
        ))
    return out


def _interpolation_mode(node) -> str:
    """How the executor renders this kind's prompt fields.

    ``full``   — base/react nodes: graph inputs + dependency outputs + `writes` vars.
    ``inputs`` — a `routes` router classifies with `text.format(**state.inputs)`,
                 so only graph inputs are in scope (upstream outputs reach it as
                 appended context instead).
    ``none``   — function/python/tool/loop/map/subgraph/input nodes and `cases`
                 routers never format these fields; a `{var}` there is inert.
    """
    if node.kind in {"base", "react"}:
        return "full"
    if node.kind == "router" and node.routes:
        return "inputs"
    return "none"


def _container_bindings(node) -> frozenset[str]:
    """Names a container makes reachable inside its body.

    Loop and map bodies run through a child executor seeded with
    ``{**state.inputs, "index": …, item_var: …}`` (plus ``feedback`` for loops),
    **and** with the iteration scope, which `render_scope` now folds into every
    node's template scope — so ``iteration`` and ``acc`` resolve in a template
    today. They used to reach expressions only, and this docstring used to say
    so; the engine widened, and a validator that still refused them would be
    reporting an error about a placeholder that renders correctly at run time.

    Reachable is not the same as *recited*. A node is told what its task text
    names, so `{item}` reaches a model only when the author writes it — and
    these names still resolve either way, which is all a template rule cares
    about. What a model is shown is `ManagerAgent.compose_user_prompt`'s
    business, not this file's.
    """
    if node.kind == "loop":
        # `acc` unconditionally: the loop puts the results so far on the scope
        # every iteration, whether or not `accumulate` gives them a var name.
        return frozenset({"index", "iteration", "feedback", "acc", node.item_var})
    if node.kind == "map":
        return frozenset({"index", node.item_var})
    return frozenset()  # a subgraph passes the parent inputs through unchanged


def _transitive_deps(by_id: dict) -> dict[str, set[str]]:
    """Every node reachable upward through `depends_on`, per node id."""
    resolved: dict[str, set[str]] = {}

    def walk(nid: str, seen: frozenset[str]) -> set[str]:
        if nid in resolved:
            return resolved[nid]
        if nid in seen or nid not in by_id:  # a cycle is caught at load time
            return set()
        acc: set[str] = set()
        for dep in by_id[nid].depends_on or []:
            acc.add(dep)
            acc |= walk(dep, seen | {nid})
        resolved[nid] = acc
        return acc

    for nid in by_id:
        walk(nid, frozenset())
    return resolved


def _check_templates(graph, report: ValidationReport) -> None:
    _check_template_level(
        nodes=graph.nodes,
        inherited_inputs=frozenset(i.name for i in graph.inputs),
        inherited_vars=frozenset(),
        inherited_maybe=frozenset(),
        report=report,
    )


def _check_template_level(
    *,
    nodes,
    inherited_inputs: frozenset[str],
    inherited_vars: frozenset[str],
    inherited_maybe: frozenset[str],
    report: ValidationReport,
) -> None:
    """Check one graph level (the top graph, or one container's body).

    Bodies share the parent's `vars` dict, so writes visible when the container
    started are visible inside it; node *ids*, which arrive via `dependency_results`,
    are scoped to their own level and are not inherited.
    """
    by_id = {n.id: n for n in nodes}
    ancestors = _transitive_deps(by_id)
    # Writes from anywhere at or below this level. A node only sees these if the
    # writer actually ran, which the DAG doesn't guarantee off the dependency path.
    level_writes = {n.writes for n in nodes if n.writes}
    level_writes |= {n.writes for n in _body_nodes(nodes) if n.writes}
    level_writes |= {n.accumulate for n in nodes if n.accumulate}
    level_writes.discard(None)

    for node in nodes:
        guaranteed = set(inherited_vars) | {
            by_id[a].writes for a in ancestors[node.id]
            if a in by_id and by_id[a].writes
        }
        scope = _Scope(
            inputs=inherited_inputs,
            vars=frozenset(guaranteed),
            deps=frozenset(node.depends_on or []),
            maybe=frozenset((level_writes | inherited_maybe) - guaranteed),
        )
        _check_node_templates(node, scope, frozenset(by_id), report)

        if node.body:
            _check_template_level(
                nodes=node.body,
                inherited_inputs=inherited_inputs | _container_bindings(node),
                inherited_vars=frozenset(guaranteed),
                # `accumulate` is only set from the second iteration on.
                inherited_maybe=scope.maybe | ({node.accumulate} if node.accumulate else set()),
                report=report,
            )


def _check_node_templates(
    node, scope: _Scope, node_ids: frozenset[str], report: ValidationReport
) -> None:
    mode = _interpolation_mode(node)
    available = scope.inputs if mode == "inputs" else scope.inputs | scope.vars | scope.deps

    _check_node_secrets(node, report)
    # The executor hands a tool node `{**graph_inputs, **dependency_results,
    # **tool_args}`, so a required parameter can legitimately arrive from the
    # scope rather than from tool_args — checking tool_args alone would reject
    # working workflows.
    _check_tool_args(node, scope.inputs | scope.vars | scope.deps, report)
    _check_tool_args_templates(node, scope.inputs | scope.vars | scope.deps, report)
    _check_output_value(node, scope.inputs | scope.vars | scope.deps, report)

    for field_name in _TEMPLATE_FIELDS:
        text = getattr(node, field_name, None)
        if not text or "{" not in text:
            continue
        parsed = _parse_placeholders(text)

        if parsed is None:
            if mode != "none":
                report.add(ValidationIssue(
        severity=Severity.WARNING,
                    kind="template_braces",
                    node_id=node.id,
                    message=(
                        f"{field_name} has an unbalanced '{{' or '}}', so the whole "
                        f"field is passed to the model unrendered"
                    ),
                    suggestion="double any literal brace ('{{' / '}}')",
                ))
            continue

        names = [p for p in parsed if not p.literal]
        literals = [p for p in parsed if p.literal]

        if mode == "none":
            # Braces are harmless where nothing formats them; a name-shaped one
            # still means someone expected a value and will not get one.
            if names:
                report.add(ValidationIssue(
        severity=Severity.WARNING,
                    kind="template_var",
                    node_id=node.id,
                    message=(
                        f"{field_name} references {_join(p.raw for p in names)}, but a "
                        f"'{node.kind}' node's {field_name} is never interpolated — it "
                        f"reaches the model (or the user) with the braces intact"
                    ),
                    subject=names[0].root,
                ))
            continue

        if literals:
            # Not a fault on its own — `render_template` passes these through as
            # written, which is usually what the author wanted. Said anyway, because
            # the alternative reading is that someone expected a substitution here.
            report.add(ValidationIssue(
        severity=Severity.WARNING,
                kind="template_braces",
                node_id=node.id,
                message=(
                    f"{field_name} contains {_join(p.raw for p in literals)}, which is "
                    f"not a variable reference — it reaches the model as written"
                ),
                suggestion="double the braces ('{{' / '}}') if that was the intent",
            ))

        for placeholder in names:
            _check_placeholder(node, field_name, placeholder, scope, available, node_ids, mode, report)


def _check_node_secrets(node, report: ValidationReport) -> None:
    """A secret must not be referenced from anything that becomes a prompt.

    `${NAME}` is only expanded inside `tool_args`, so one written into a goal
    would reach the model verbatim as `${DB_PASSWORD}` — harmless in itself, and
    exactly the mistake that means the author believed it would be filled in. The
    dangerous version of that belief is the one where it *is* filled in, so the
    reference is refused where it can never work rather than tolerated.
    """
    from neurosurfer.graph.engine.secrets import secret_refs

    declared = set(getattr(node, "secrets", None) or ())
    for field_name in _TEMPLATE_FIELDS:
        text = getattr(node, field_name, None)
        if not text:
            continue
        for ref in secret_refs(text):
            report.add(ValidationIssue(
        severity=Severity.ERROR,
                kind="secret_in_prompt",
                node_id=node.id,
                message=(
                    f"{field_name} references ${{{ref}}}. A stored value is only "
                    f"available to `tool_args`, never to text that reaches a model — "
                    f"a secret in a prompt is a secret in the trace"
                ),
                subject=ref,
                suggestion=(
                    "pass it to a tool via tool_args on a node declaring "
                    f"secrets: [{ref}]"
                ),
            ))

    # `tool` states its whole call in tool_args; `react` binds *some* of it and
    # lets its model compose the rest. Both reach a secret the same way, and a
    # react node needing a credential is the commonest integration there is —
    # query a database, call an authenticated API. Warning about it used to send
    # a build round a loop it could not leave, because both of the decisions that
    # produced it were correct.
    if declared and node.kind not in {"tool", "react"}:
        report.add(ValidationIssue(
        severity=Severity.WARNING,
            kind="secret_unusable",
            node_id=node.id,
            message=(
                f"declares secrets {_join(sorted(declared))} but is a "
                f"'{node.kind}' node, which has no tool_args to use them in"
            ),
            subject=sorted(declared)[0],
        ))
    elif declared and node.kind == "react" and not getattr(node, "tool_args", None):
        report.add(ValidationIssue(
        severity=Severity.WARNING,
            kind="secret_unbound",
            node_id=node.id,
            message=(
                f"declares secrets {_join(sorted(declared))} but binds nothing in "
                f"tool_args, so no tool call will receive them"
            ),
            subject=sorted(declared)[0],
            suggestion=(
                f"add tool_args binding the credential to the parameter that "
                f"takes it, e.g. tool_args: {{dsn: '${{{sorted(declared)[0]}}}'}} — "
                f"the model then composes only the rest of the call"
            ),
        ))


def _tool_arg_strings(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Every string inside `tool_args`, with a path naming where it sits.

    Recurses because a model nests `tool_args` as readily as it writes a flat one,
    and a placeholder buried in a list is passed to the tool just the same.
    """
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for k, v in value.items():
            out += _tool_arg_strings(v, f"{path}.{k}" if path else str(k))
        return out
    if isinstance(value, list):
        out = []
        for i, v in enumerate(value):
            out += _tool_arg_strings(v, f"{path}[{i}]")
        return out
    return []


def _check_tool_args_templates(node, available: frozenset[str] | set[str],
                               report: ValidationReport) -> None:
    """`tool_args` placeholders that resolve to nothing.

    `_TEMPLATE_FIELDS` covers `purpose`/`goal`/`expected_result` and has never
    covered `tool_args`, so this was invisible to validation for the life of the
    mechanism. It matters more here than there: a prompt with a stray `{x}` is a
    slightly worse prompt a model reads past, but `tool_args` goes **straight to a
    tool** — a `tool` node makes no model call at all, and a `react` node's bound
    args are supplied verbatim on every call.

    The build that earned this wrote `content: "{output of generate_markdown_table}"`
    — a *description* of the value where a reference belongs. Nothing resolved it,
    the renderer left it as written, `write_file` wrote it, and the build, the run
    and all three nodes reported success over a 35-byte file containing exactly
    that string.
    """
    args = getattr(node, "tool_args", None)
    if not args:
        return

    # `${NAME}` is a secret, expanded by a separate later pass. `str.Formatter`
    # sees the `{NAME}` inside it, so without this every declared credential
    # reports as unresolved — saying "your secret did not resolve" in precisely
    # the case where everything works.
    declared = set(getattr(node, "secrets", None) or ())

    for where, text in _tool_arg_strings(args):
        if "{" not in text:
            continue
        parsed = _parse_placeholders(text)
        if parsed is None:
            continue  # unbalanced braces — `format()` leaves the text alone anyway
        for p in parsed:
            if p.root in available or p.root in declared:
                continue

            # A prose placeholder usually *contains* the name it meant, which is a
            # far stronger signal than edit distance: `{output of generate_markdown_table}`
            # holds `generate_markdown_table` outright. Fall back to difflib for a
            # plain misspelling.
            contained = sorted(
                (a for a in available if a and a in p.root), key=len, reverse=True
            )
            close = difflib.get_close_matches(p.root, sorted(available), n=1, cutoff=0.7)
            hint = contained[0] if contained else (close[0] if close else None)

            field = f"tool_args.{where}" if where else "tool_args"
            report.add(ValidationIssue(
        severity=Severity.ERROR,
                kind="tool_args_template",
                node_id=node.id,
                message=(
                    f"{field} contains {{{p.raw}}}, which resolves to nothing — the "
                    f"tool receives the literal text '{{{p.raw}}}' as this argument "
                    f"(in scope: {_join(sorted(available)) or 'nothing'})"
                ),
                suggestion=(
                    f"did you mean '{{{hint}}}'? a placeholder must be a name, not a "
                    f"description of the value"
                    if hint else
                    "reference a graph input or an upstream node id, or double the "
                    "brace ('{{' / '}}') if it is meant literally"
                ),
                subject=p.root,
            ))


def _check_tool_args(node, available: frozenset[str], report: ValidationReport) -> None:
    """A `tool` node must actually supply what the tool it calls requires.

    A tool node makes **no model call** — `tool_args` is the whole instruction, and
    nothing composes what is missing. Until this check existed a node could declare
    `tools: [query_sql]` with no `tool_args` at all and a careful prose goal that
    nothing would ever read; it validated, registered, and failed at runtime with
    whatever the tool's own service says about a call with no arguments. One live
    build shipped seven such nodes.

    *available* is what the executor will pass alongside `tool_args`: the graph's
    inputs and this node's dependency outputs, both keyed by name. A required
    parameter satisfied from there is genuinely satisfied, so it is not an error.
    """
    if node.kind != "tool" or not (node.tools or []):
        return
    schema = _tool_input_schema(node.tools[0])
    if not schema:
        return  # an MCP tool whose schema we could not read; nothing to check against
    required = [str(r) for r in (schema.get("required") or [])]
    if not required:
        return
    supplied = set((getattr(node, "tool_args", None) or {}).keys()) | available
    missing = [r for r in required if r not in supplied]
    if not missing:
        return
    report.add(ValidationIssue(
        severity=Severity.ERROR,
        kind="tool_args",
        node_id=node.id,
        message=(
            f"This step runs '{node.tools[0]}' but never says what to give it "
            f"for {_join(missing)}, and there is no model here to work it out."
        ),
        subject=missing[0],
        # Plain, because this is the message somebody meets the first time they
        # drag a tool onto the canvas and press Run. It was still written in
        # field names — `add tool_args`, `` `react` node with `tools: [...]` `` —
        # while the rule id sat in the swept list, and nothing caught that because
        # no test graph had ever provoked it. The `detail` below is where the
        # field name belongs.
        suggestion=(
            f"Set {_join(missing)} on this step — or, if the value has to be "
            f"worked out (a query to compose, a phrase to search), make this an "
            f"agent step that can call the tool, so a model can write it."
        ),
        detail=f"tool_args is missing {', '.join(missing)}",
    ))

    # A credential parameter left unbound is worth its own sentence: the fix is a
    # stored value, not a literal, and saying so here is what stops the next
    # attempt writing a password into the graph.
    try:
        from neurosurfer.tools.registry import all_tools

        tool = next((t for t in all_tools() if t.name == node.tools[0]), None)
        secret_params = set(getattr(tool, "secret_inputs", None) or ())
    except Exception:  # noqa: BLE001
        secret_params = set()
    for name in (p for p in missing if p in secret_params):
        report.add(ValidationIssue(
        severity=Severity.ERROR,
            kind="tool_args",
            node_id=node.id,
            message=f"This step needs a stored credential for '{name}', and none is set.",
            subject=name,
            suggestion=(
                "Pick one of your saved credentials for it. The value is supplied "
                "when the step runs and never reaches a prompt or a trace."
            ),
            detail=(
                f"declare `secrets: [SOME_NAME]` and write "
                f"`tool_args: {{{name}: '${{SOME_NAME}}'}}`"
            ),
        ))


def _check_output_value(node, available: frozenset[str] | set[str],
                        report: ValidationReport) -> None:
    """An `output` node's `value` template, checked the way `tool_args` is.

    An error rather than a warning, and for the same reason: this text *is* the
    answer handed back to whoever called the workflow. A leftover `{name}` in a
    prompt is a slightly worse prompt a model reads past; a leftover `{name}` in
    the return value is a defect the caller receives as content, and the executor
    refuses it at run time — so validation should say so first.
    """
    if node.kind != "output":
        return

    if node.value is None:
        # Nothing to interpolate: the node passes its dependency through. That is
        # only well-defined if it *has* one.
        if not node.depends_on:
            report.add(ValidationIssue(
        severity=Severity.ERROR,
                kind="output_empty",
                node_id=node.id,
                message=(
                    "This step has nothing to return — nothing is connected to "
                    "it and no fixed text is set."
                ),
                suggestion="Connect the step whose result is the answer.",
                detail="output node has neither `depends_on` nor `value`",
            ))
        return

    if "{" not in node.value:
        return
    parsed = _parse_placeholders(node.value)
    if parsed is None:
        report.add(ValidationIssue(
        severity=Severity.WARNING,
            kind="template_braces",
            node_id=node.id,
            message="value has an unbalanced '{' or '}', so it is returned unrendered",
            suggestion="double any literal brace ('{{' / '}}')",
        ))
        return

    for p in parsed:
        if p.literal or p.root in available:
            continue
        contained = sorted(
            (a for a in available if a and a in p.root), key=len, reverse=True
        )
        close = difflib.get_close_matches(p.root, sorted(available), n=1, cutoff=0.7)
        hint = contained[0] if contained else (close[0] if close else None)
        report.add(ValidationIssue(
        severity=Severity.ERROR,
            kind="output_template",
            node_id=node.id,
            message=(
                f"value contains {{{p.raw}}}, which resolves to nothing — the caller "
                f"receives the literal text '{{{p.raw}}}' as the workflow's answer "
                f"(in scope: {_join(sorted(available)) or 'nothing'})"
            ),
            suggestion=(
                f"did you mean '{{{hint}}}'?"
                if hint else
                "reference a graph input or an upstream node id, or double the "
                "brace ('{{' / '}}') if it is meant literally"
            ),
            subject=p.root,
        ))


def _check_placeholder(
    node,
    field_name: str,
    placeholder: _Placeholder,
    scope: _Scope,
    available: frozenset[str] | set[str],
    node_ids: frozenset[str],
    mode: str,
    report: ValidationReport,
) -> None:
    root = placeholder.root
    if root in available:
        return

    where = f"{field_name} references {{{placeholder.raw}}}"

    # Checked before `maybe`: a router never reads `vars`, so "it might resolve if
    # that branch ran first" is not true for one — the reference is dead either way.
    if mode == "inputs" and (
        root in node_ids or root in scope.vars or root in scope.deps or root in scope.maybe
    ):
        # The name is real, just not reachable from a router: `_route_by_classification`
        # formats with graph inputs alone. Upstream outputs are already appended to the
        # classifier prompt, so the reference is redundant rather than merely wrong.
        report.add(ValidationIssue(
        severity=Severity.ERROR,
            kind="template_var",
            node_id=node.id,
            message=(
                f"{where}, but a 'routes' router interpolates graph inputs only "
                f"(available: {_join(sorted(scope.inputs)) or 'none'})"
            ),
            suggestion=(
                "drop it — the router is already given its upstream nodes' output as "
                "context — or reference a graph input"
            ),
            subject=root,
        ))
        return

    if root in scope.maybe:
        report.add(ValidationIssue(
        severity=Severity.WARNING,
            kind="template_var",
            node_id=node.id,
            message=(
                f"{where}, which is written by a node this one does not depend on — "
                f"it resolves only when that branch happens to run first"
            ),
            suggestion=f"depend on the node that writes '{root}' to make it certain",
            subject=root,
        ))
        return

    if root in node_ids:
        report.add(ValidationIssue(
        severity=Severity.ERROR,
            kind="template_var",
            node_id=node.id,
            message=(
                f"{where}, which is a node id this node does not depend on — only "
                f"declared dependencies are in scope"
            ),
            suggestion=f"add '{root}' to depends_on",
            subject=root,
        ))
        return

    match = difflib.get_close_matches(root, sorted(available), n=1, cutoff=0.7)
    report.add(ValidationIssue(
        severity=Severity.ERROR,
        kind="template_var",
        node_id=node.id,
        message=(
            f"{where}, which nothing provides "
            f"(in scope: {_join(sorted(available)) or 'nothing'})"
        ),
        suggestion=(
            f"did you mean '{{{match[0]}}}'?" if match
            else "declare it as a graph input, or use an upstream node's id or `writes` name"
        ),
        subject=root,
    ))


def _join(names) -> str:
    return ", ".join(f"'{n}'" for n in names)


# ── rule ─────────────────────────────────────────────────────────────────────


@graph_rule(severity=Severity.ERROR)
def placeholders_resolve(graph, ctx, report) -> None:
    """Every `{name}` a node interpolates resolves in the scope it will run in."""
    _check_templates(graph, report)

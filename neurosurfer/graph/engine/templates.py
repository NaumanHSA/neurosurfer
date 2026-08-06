"""Prompt templates, and the renderer that fills a node's `{vars}`."""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any

_FORMATTER = string.Formatter()


class _Reachable:
    """A value whose fields can be reached as `[key]` **or** `.key`.

    `string.Formatter` resolves `{x[k]}` through `__getitem__` and `{x.k}` through
    `getattr`, so the two spellings worked on opposite kinds of value: a node
    returning a dict answered only the first, and a node with an `output_schema` —
    whose output is a pydantic model — only the second.

    Which one a node returns is not visible on the canvas, and it changes the
    moment somebody sets an output shape. So an author had to know an invisible
    fact to pick the syntax, and got a placeholder left as written when they
    guessed wrong — silent, because an unresolved placeholder is passed through
    verbatim by design.

    Wrapping makes the spellings equivalent. Formatting delegates to the value, so
    `{x}` on its own is unchanged.
    """

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        object.__setattr__(self, "_value", value)

    def __getitem__(self, key: Any) -> Any:
        value = self._value
        try:
            return _reachable(value[key])
        except (TypeError, KeyError, IndexError):
            pass
        try:
            return _reachable(getattr(value, str(key)))
        except AttributeError as exc:  # let the renderer leave it as written
            raise KeyError(key) from exc

    def __getattr__(self, name: str) -> Any:
        return self[name]

    # Formatting, stringifying and truthiness all belong to the wrapped value —
    # the wrapper exists only to widen field access.
    def __format__(self, spec: str) -> str:
        return format(self._value, spec)

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return repr(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)


def _reachable(value: Any) -> Any:
    """Wrap only what has fields to reach into; leave scalars exactly as they are."""
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    return _Reachable(value)


class _ReachableScope(Mapping):
    """The template scope, with every value field-reachable both ways."""

    def __init__(self, inner: Mapping[str, Any]) -> None:
        self._inner = inner

    def __getitem__(self, key: str) -> Any:
        return _reachable(self._inner[key])

    def __iter__(self):
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)


def node_instruction(node: Any, default: str = "") -> str:
    """The one line saying what *node* should do, whichever way it was written.

    ``_build_system_prompt`` already knows this precedence — ``instructions``
    wins outright, the older ``purpose``/``goal`` are still read so graphs on
    disk keep running. But two kinds build their prompt by hand instead of going
    through it, and when ``instructions`` arrived both were missed:

    * an **LLM router** classified with ``purpose or goal``, so a router whose
      instruction was written in the studio — which writes ``instructions`` and
      clears the older three — fell back to ``"Route for node <id>"`` and
      classified with no instruction at all;
    * an **input** node asked ``purpose or goal``, so a person was asked
      ``"Input needed for 'input'"`` for a question that had been written.

    Neither failed loudly: one silently classified worse, the other silently
    asked a worse question. Hence one function rather than the same expression
    written at each site, which is how they came to disagree in the first place.
    """
    for text in (
        getattr(node, "instructions", None),
        getattr(node, "purpose", None),
        getattr(node, "goal", None),
    ):
        if text and str(text).strip():
            return str(text).strip()
    return default


class _Namespace:
    """Attribute access over a mapping, so `{nodes.step_id}` renders.

    The engine speaks two languages that look alike and are not: expressions say
    `nodes.summarise` / `inputs.topic`, templates say `{summarise}` / `{topic}`.
    A model moving between them confuses the two constantly — one build spent
    fifteen validation rounds writing `{nodes.query_inquiry_activity_result}`,
    being told the correct name, and writing it again.

    Accepting the namespaced form costs nothing (`{nodes.x}` could never have
    meant anything else) and removes the confusion rather than describing it in a
    better error message.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = values

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as e:  # unresolved → left as written, as any miss is
            raise AttributeError(name) from e


def with_namespaces(
    scope: dict[str, Any],
    *,
    inputs: Mapping[str, Any] | None = None,
    nodes: Mapping[str, Any] | None = None,
    variables: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """*scope* plus `inputs.` / `nodes.` / `vars.` namespaces for templates.

    A real key always wins: a graph input genuinely called `nodes` keeps working,
    since taking that name away would be a silent behaviour change for a
    convenience.
    """
    out = dict(scope)
    for key, values in (("inputs", inputs), ("nodes", nodes), ("vars", variables)):
        if values and key not in out:
            out[key] = _Namespace(values)
    return out


def render_scope(
    inputs: Mapping[str, Any] | None = None,
    *,
    nodes: Mapping[str, Any] | None = None,
    variables: Mapping[str, Any] | None = None,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything a node's `{placeholders}` may resolve against — built once.

    Four sites used to assemble this by hand and **all four disagreed**: a
    base/react node saw graph inputs, dependency outputs and `writes` vars; an
    output node saw the same; a `tool` node saw no vars; and a `routes` router
    saw graph inputs and nothing else. None of them saw the container scope.

    That last omission is why a `map` body had to be handed the *parent's whole
    input dict* — it was the only door `{item}` could arrive through, since the
    iteration scope was not part of any node's template scope. Widening here is
    what lets the container stop doing that.

    Layered most-general to most-local, so **the innermost name wins**: a graph
    input called `item` does not shadow the item a `map` is currently on.
    """
    flat: dict[str, Any] = {}
    for layer in (inputs, nodes, variables, scope):
        if layer:
            flat.update(layer)
    return with_namespaces(flat, inputs=inputs, nodes=nodes, variables=variables)


def recited_names(*texts: str | None) -> frozenset[str]:
    """Names whose **whole value** the given templates already state.

    A node whose instruction is ``"Summarise this review: {item}"`` has the
    review in its system prompt. Printing `item: …` underneath it is the same
    text a second time — the shape Phase 6 found stating one value three times
    under three headings that disagreed about what it was.

    Only a bare ``{name}`` counts. ``{reviews[0]}``, ``{doc.title}`` and
    ``{n:>4}`` each state *part* of a value or a formatting of it, so the value
    itself has not been recited and hiding it would remove something the reader
    does not have.
    """
    out: set[str] = set()
    for text in texts:
        if not text:
            continue
        try:
            parts = list(_FORMATTER.parse(text))
        except ValueError:  # unpaired brace — no placeholders to speak of
            continue
        for _literal, field, spec, conversion in parts:
            if field and field.isidentifier() and not spec and not conversion:
                out.add(field)
    return frozenset(out)


def render_template(text: str, scope: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Fill every `{name}` that resolves in *scope*; leave the rest exactly as written.

    ``str.format`` is all-or-nothing: one unknown name raises, and the caller is left
    holding the raw template — so a typo in *one* placeholder used to cost a node
    *every* value in the same field. Rendering per placeholder means a miss costs only
    itself. For text where nothing is missing the output is byte-identical to
    ``text.format(**scope)``; this only changes what happens on the failure path.

    Returns the rendered text and the placeholders that could not be resolved, so the
    caller can say precisely what was left behind instead of "this template failed".
    """
    scope = _ReachableScope(scope)
    try:
        parts = list(_FORMATTER.parse(text))
    except ValueError:
        # An unpaired '{' or '}' — the template can't be split into placeholders at
        # all, so there is no partial rendering to do. Hand the text back untouched.
        return text, []

    out: list[str] = []
    unresolved: list[str] = []
    for literal, field, spec, conversion in parts:
        out.append(literal)
        if field is None:
            continue
        try:
            # A nested spec (`{x:{width}}`) is itself a template; if its own
            # placeholders don't resolve, `format` below raises and we fall through
            # to leaving the whole thing literal.
            resolved_spec = (
                render_template(spec, scope)[0] if spec and "{" in spec else spec
            )
            value, _ = _FORMATTER.get_field(field, (), scope)
            value = _FORMATTER.convert_field(value, conversion)
            out.append(format(value, resolved_spec or ""))
        except Exception:  # noqa: BLE001 - any lookup/format failure leaves it as-is
            out.append(_as_written(field, spec, conversion))
            unresolved.append(field)
    return "".join(out), unresolved


def _as_written(field: str, spec: str | None, conversion: str | None) -> str:
    """Rebuild the placeholder source so an unresolved one passes through unchanged."""
    text = "{" + field
    if conversion:
        text += "!" + conversion
    if spec:
        text += ":" + spec
    return text + "}"


MANAGER_SYSTEM_PROMPT = """You are a workflow orchestrator for a multi-agent system.

Your ONLY job is to write the INSTRUCTIONS for the next node’s agent:
- What to do (task)
- Output format and strict output contract (especially for structured mode)
- Constraints (length, bullet vs paragraph, etc.)
- Tool usage guidance (if any tools are allowed)

IMPORTANT RULES:
- Do NOT restate, summarize, rewrite, or paraphrase dependency outputs.
- Do NOT include the dependency outputs themselves.
- Do NOT describe the graph, nodes, or orchestration mechanics.
- Assume dependency outputs will be appended verbatim after your instructions.

Output requirements:
Return ONLY the instruction prompt text that will be prepended before dependency context.
No markdown fences. No JSON wrapper. No extra commentary.
""".strip()


COMPOSE_NEXT_AGENT_PROMPT_TEMPLATE = """You are preparing instructions for the next agent node.

ORIGINAL USER REQUEST:
{user_intent}

NODE SPEC:
- PURPOSE: {purpose}
- GOAL: {goal}
- EXPECTED OUTPUT: {expected}
- MODE: {mode}
- TOOLS AVAILABLE: {tools}

{extra_inputs}
{dependency_section}
Write concise instructions for the next agent. The instructions MUST:
1) Open with "The user wants to: <restate user request>" so the agent never loses context.
2) State TASK in 1-3 lines, grounded in the original user request above.
3) State OUTPUT_CONTRACT: exact format rules matching MODE/EXPECTED_OUTPUT.
4) If MODE=structured: output STRICT JSON ONLY, no markdown fences, no extra text.

Dependency outputs (if any) will be appended verbatim after your instructions.
Return ONLY the instruction text — no preamble, no meta-commentary.
""".strip()


#: The system prompt for a node that states its job in one field.
#:
#: Preferred over the three-field template below whenever ``node.instructions``
#: is set. The framing around it stays, because it is the part an author should
#: not have to repeat on every node: that this step sits inside a larger
#: workflow, and how to behave while running.
NODE_SYSTEM_TEMPLATE = """You are a specialized agent in a larger workflow.

Your task:
{instructions}

General behaviour:
- Be precise and concise unless the task requires extended output.
- Use clear structure (headings/bullets) when helpful.
- If you are calling tools, interpret their outputs carefully and explain your reasoning.
"""


#: The three-field system prompt, for nodes written before ``instructions``.
#:
#: Kept, not deprecated-and-deleted: every workflow already registered names its
#: job under ``purpose`` / ``goal`` / ``expected_result``, and a graph that ran
#: yesterday has to run today. New nodes should set ``instructions`` instead —
#: see ``GraphNode.instructions`` for why one field replaced three.
DEFAULT_NODE_SYSTEM_TEMPLATE = """You are a specialized agent in a larger workflow.

Your role:
- PURPOSE: {purpose}
- GOAL: {goal}
- EXPECTED_RESULT: {expected_result}

General behaviour:
- Be precise and concise unless the task requires extended output.
- Use clear structure (headings/bullets) when helpful.
- If you are calling tools, interpret their outputs carefully and explain your reasoning.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
)

from neurosurfer.llm.types import Usage
from neurosurfer.tracing import TraceResult


class NodeMode(StrEnum):
    AUTO = "auto"
    TEXT = "text"
    STRUCTURED = "structured"
    JSON = "json"
    TOOL = "tool"

# ---------------------------
# Graph-level input spec
# ---------------------------
class GraphInput(BaseModel):
    """
    Specification for a top-level graph input.

    Normalized form:
      name: str
      type: str     (string|integer|float|boolean|object|array|file|image, or synonyms)
      required: bool
      description: Optional[str]

    ``file`` and ``image`` are still *strings* at runtime — the value a node
    receives is a path, exactly as if someone had typed one. What the type
    changes is how it is asked for: a button that uploads, rather than a box you
    paste a path into and hope the gateway's working directory agrees with yours.
    Everything downstream of the input — templates, `read_file`, a vision node —
    is unchanged.
    """

    name: str = Field(..., description="Input name (key expected at runtime).")
    type: str = Field(
        default="string",
        description=(
            "Logical type: string|integer|float|boolean|object|array|file|image "
            "(or 'str', 'int', etc.)."
        ),
    )
    required: bool = Field(
        default=True,
        description="If True, this key must be present in runtime inputs.",
    )
    description: str | None = Field(
        default=None,
        description="Optional human description of the input.",
    )
    #: For `file`/`image`: what the picker should offer, in the HTML `accept`
    #: form (".csv,.txt" or "image/*"). Advisory — the server still validates.
    accept: str | None = Field(
        default=None,
        description="File picker filter for file/image inputs, e.g. '.csv' or 'image/*'.",
    )
    #: For `file`/`image`: refuse anything larger, in bytes.
    max_bytes: int | None = Field(
        default=None,
        description="Maximum upload size in bytes for file/image inputs.",
    )
    #: A closed set of allowed values. Renders as a select, which removes a whole
    #: class of typo that a free-text box invites.
    enum: list[str] = Field(
        default_factory=list,
        description="Allowed values for a string input; renders as a choice.",
    )

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, v: str) -> str:
        v = v.strip()
        lower = v.lower()
        if lower in {"str", "string", "text"}:
            return "string"
        if lower in {"int", "integer"}:
            return "integer"
        if lower in {"float", "number"}:
            return "float"
        if lower in {"bool", "boolean"}:
            return "boolean"
        if lower in {"dict", "object"}:
            return "object"
        if lower in {"list", "array"}:
            return "array"
        if lower in {"file", "path", "filepath", "document"}:
            return "file"
        if lower in {"image", "picture", "photo"}:
            return "image"
        return lower

    model_config = dict(extra="ignore")


# ---------------------------
# Node policy (per-node AgentConfig overrides)
# ---------------------------
class NodePolicy(BaseModel):
    """
    Per-node policy that can override some AgentConfig settings and add
    node-level execution constraints (e.g., timeout).

    YAML example:
        nodes:
          - id: research
            policy:
              retries: 1
              timeout_s: 30
              max_new_tokens: 180
              temperature: 0.2
              allow_input_pruning: false
              repair_with_llm: true
              strict_tool_call: true
    """
    max_new_tokens: int | None = Field(default=None, description="Override AgentConfig.max_new_tokens for this node only.")
    temperature: float | None = Field(default=None, description="Override AgentConfig.temperature for this node only.")
    retries: int | None = Field(default=None, description="Override AgentConfig.retry.max_route_retries for this node.")
    timeout_s: int | None = Field(
        default=None,
        description=(
            "Soft timeout for this node in seconds. Execution isn't forcibly "
            "cancelled but the node will be marked as errored if exceeded."
        ),
    )
    # Direct AgentConfig-like overrides
    allow_input_pruning: bool | None = None
    repair_with_llm: bool | None = None
    strict_tool_call: bool | None = None
    strict_json: bool | None = None
    max_json_repair_attempts: int | None = None    # for malformed JSON repairs

    skip_special_tokens: bool | None = None
    return_stream_by_default: bool | None = None
    log_internal_thoughts: bool | None = None

    class Config:
        extra = "ignore"  # ignore unknown keys under 'policy'


# ---------------------------
# Graph node & spec
# ---------------------------
# base/react/function/python/tool are the classic node kinds. Control-flow kinds:
#   router (1d) selects a branch; loop (1e) iterates a body until a condition;
#   map (1f) fans a body out over a collection; subgraph (1h) runs a nested body
#   once (composition); input (1i) pauses for a human value; output (1j) declares
#   what the graph returns.
_VALID_NODE_KINDS = {
    "base", "react", "function", "python", "tool",
    "router", "loop", "map", "subgraph", "input", "output",
}

#: Kinds that end a graph rather than continuing it. Nothing may depend on one —
#: see ``_no_dependents_on_terminal`` on :class:`GraphSpec`.
_TERMINAL_NODE_KINDS = {"output"}


class RouterCase(BaseModel):
    """One branch of a :class:`GraphNode` router: if ``when`` is truthy, select ``to``.

    ``when`` is a restricted expression (see ``engine.expressions``) evaluated against
    the live workflow state. Cases are tried in order; the first match wins. A router
    with no matching case falls back to the node's ``default``.
    """

    when: str | None = Field(
        default=None,
        description="Predicate expression over state; None/empty means 'always match' (a catch-all).",
    )
    to: str = Field(description="Target node id to activate when this case matches.")
    label: str | None = Field(
        default=None,
        description="Optional human/LLM-facing label for this branch (used by LLM routers).",
    )

    model_config = dict(extra="ignore")


class GraphNode(BaseModel):
    # populate_by_name lets YAML use the alias `as:` while Python uses `item_var`.
    model_config = ConfigDict(populate_by_name=True)

    id: str
    #: A human label for this step, shown wherever the node is drawn.
    #:
    #: Purely presentational — nothing in the engine reads it. It exists because
    #: `id` has two jobs it cannot both do well: it is the reference every
    #: `depends_on`, template and trace line uses, so renaming it breaks the graph,
    #: and it is also the only thing a canvas had to show. So ids stay stable and
    #: terse while the label is free to say "Count passengers by survival status".
    #: Absent means "call it by its kind" — the studio falls back to the kind's own
    #: label rather than inventing one.
    name: str | None = None
    description: str | None = None
    kind: str = Field(default="base", description="Node kind: base | react | function | python | tool | router")
    #: What this step should do, in one field.
    #:
    #: The engine asked for three — ``purpose`` (who you are), ``goal`` (what to
    #: do) and ``expected_result`` (what to hand back) — which is a taxonomy, and
    #: a taxonomy is a thing an author has to learn before they can write a
    #: sentence. In practice the split was guessed at: the same instruction
    #: turned up under a different heading depending on who wrote the node, and
    #: the model saw one system prompt either way.
    #:
    #: ``instructions`` is that one system prompt, written directly. When it is
    #: set the other three are ignored; when it is absent they are still read and
    #: composed, so every workflow already on disk keeps running unchanged.
    #: See ``Executor._build_system_prompt``.
    instructions: str | None = None
    #: Switched off by hand: the node stays in the graph and does not run.
    #:
    #: Distinct from deleting it, which loses the configuration, and from a
    #: ``when`` guard, which is a decision the *run* makes. This one is a
    #: decision the author made, and it survives in the file so it can be undone.
    #: Pruned exactly like a false guard — a normal not-taken branch, so
    #: dependents still run if any other incoming branch is live.
    disabled: bool = False
    #: Superseded by ``instructions``. Still read, never required.
    purpose: str | None = None
    goal: str | None = None
    expected_result: str | None = None
    tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    mode: NodeMode = Field(default=NodeMode.AUTO)
    #: The shape this node's output is validated against, in one of two forms.
    #:
    #: A **string** is an import path to a pydantic model already in code
    #: (``my_module:ResultModel``) — right for a shape shared across workflows.
    #: A **dict** is a JSON Schema written on the node itself, which is what the
    #: studio writes: an author picking "structured" needs somewhere to say what
    #: the structure is, and putting a Python file on the server was not it.
    #:
    #: Both resolve to a pydantic model before the call — see
    #: ``GraphExecutor._load_output_schema_if_needed``. The string form is first
    #: in the union so a plain path keeps deserialising as a path.
    output_schema: str | dict[str, Any] | None = None
    # Per-node LLM selection. `provider` names a configured profile (cheap model to
    # classify, strong one to write, a local one for anything private); `model`
    # rebinds whichever provider is chosen to a specific model. Both fall back to the
    # provider the run was started with.
    provider: str | None = Field(
        default=None,
        description="Name of a configured provider profile to run this node on.",
    )
    model: str | None = None
    rag: bool | None = False

    # ── control flow (Phase 1c/1d) ──────────────────────────────────────────
    # Activation guard: the node runs only if this expression is truthy against the
    # live state. A falsy guard *prunes* the node (a normal not-taken branch), which
    # is distinct from an error-skip: dependents still run if any other incoming
    # branch is live (OR-join). None means "always active" (classic DAG behaviour).
    when: str | None = Field(
        default=None,
        description="Conditional-edge guard: restricted expression; node runs only if truthy.",
    )
    # Store this node's output under state.vars.<writes> so later predicates can read
    # it by a stable name regardless of node id.
    writes: str | None = Field(
        default=None,
        description="Optional variable name to store this node's output under (state.vars.<name>).",
    )
    # Error/fallback routing (Phase 1g): on failure, route to this handler node
    # instead of AND-skipping dependents. The error text is exposed as
    # state.vars.<id>__error for the handler to read.
    on_error: str | None = Field(
        default=None,
        description="Node id to activate if this node errors (fallback branch).",
    )
    # Router node (kind='router'). Two flavours:
    #
    # 1. `routes` (the simple, recommended form): the router IS the classifier.
    #    One LLM call — instructed by purpose/goal — picks one label from the dict
    #    and its target runs; every other target is pruned. N-way by construction.
    #    `repair` retries once with corrective feedback when the model's answer
    #    matches no label, before falling back to `default`.
    #
    # 2. `cases` (deterministic/advanced): ordered {when, to} predicates evaluated
    #    against state — no LLM call, free and reproducible. Cases with only
    #    labels (no `when`) behave like `routes` (legacy LLM-router form).
    routes: dict[str, str] | None = Field(
        default=None,
        description="Classification router: {label: target_node_id}. The router "
                    "itself classifies via one LLM call and picks a branch.",
    )
    repair: bool = Field(
        default=True,
        description="routes only: retry once with corrective feedback when the "
                    "model's answer matches no label (then fall back to default).",
    )
    cases: list[RouterCase] | None = Field(
        default=None,
        description="Deterministic router branches: ordered {when, to} cases.",
    )
    default: str | None = Field(
        default=None,
        description="Router fallback target node id when no route/case matches.",
    )

    # ── iteration (Phase 1e loop / 1f map) ──────────────────────────────────
    # Nested sub-graph body run by a loop/map node (a list of GraphNodes forming
    # their own DAG). Body nodes see the parent state plus iteration scope.
    body: list[GraphNode] | None = Field(
        default=None,
        description="Nested sub-graph nodes for loop/map bodies.",
    )
    body_outputs: list[str] = Field(
        default_factory=list,
        description="Body node ids whose outputs form the iteration result (default: all).",
    )
    # loop node. ONE stop condition, in one field, because a loop stops for one
    # reason and asking the author to also pick a mechanism was asking the wrong
    # question. `until` is either:
    #   a function — a callable, or the name of one in the graph's `functions:`
    #                sidecar. Gets a `LoopIteration`, returns True to stop (or
    #                `(True, "reason")`). Deterministic, free, and strictly more
    #                capable than the sandboxed expression it replaces.
    #   plain English — judged by a hidden LLM decision each iteration, which
    #                also reports a condition it cannot relate to the work at all
    #                (see `iteration._judge_loop_until`).
    # Which one is not guessed from the string: a name that the sidecar defines
    # as a callable is a function, and everything else is prose.
    max_iterations: int | None = Field(
        default=None,
        description="Hard iteration ceiling for loop nodes (always required for loops).",
    )
    until: str | Callable[..., Any] | None = Field(
        default=None,
        description="Loop stop condition: a callable, the name of one in the graph's "
                    "`functions:` file, or a plain-English condition judged by an "
                    "internal LLM decision each iteration.",
    )

    @field_serializer("until")
    def _serialize_until(self, v: Any) -> str | None:
        """A live callable dumps as its name — which is what YAML can hold.

        The name only round-trips if the graph also carries a `functions:` file
        defining it; `save_package` is where that is checked, because only there
        is there a directory to write the file into.
        """
        if v is None or isinstance(v, str):
            return v
        return getattr(v, "__name__", None) or str(v)
    accumulate: str | None = Field(
        default=None,
        description="Loop: variable name to append each iteration's body output to (a list).",
    )
    # map node
    over: str | None = Field(
        default=None,
        description="Map: expression yielding the collection to fan out over.",
    )
    item_var: str = Field(
        default="item",
        alias="as",
        description="Map/loop: name the current item is bound to in the body scope.",
    )
    concurrency: int = Field(
        default=1,
        description="Map: max body executions to run in parallel.",
    )
    # input / human-in-the-loop node (Phase 1i): the choices offered to the user.
    options: list[str] = Field(
        default_factory=list,
        description="Input node: optional multiple-choice options presented to the user.",
    )
    #: How an input node collects its value: as a chat message, or as a form.
    #:
    #: Stored rather than inferred, and the reason is a case that arrives in week
    #: one: a workflow taking exactly one string parameter is *structurally
    #: identical* to free chat, and means something different. Inferring would
    #: flip such a workflow to a chat composer and throw away the label its
    #: author wrote. An empty form is likewise indistinguishable from chat before
    #: the first parameter exists, so the dialog would have nothing to show.
    #:
    #: `mode` was taken (``NodeMode``), hence the name.
    input_mode: str = Field(
        default="text",
        description="Input node: 'text' collects one free-text message, 'dict' collects the graph's declared inputs.",
    )

    # ── output node ──────────────────────────────────────────────────────────
    #
    #: What the graph returns, as a template over what ran.
    #:
    #: `graph.outputs` could only ever *select* whole node outputs by id, which
    #: forced a `python` node whose entire job was to glue two strings together
    #: any time the answer was not exactly one node's output. `value` is an
    #: interpolated template — `"{summary} ({rows} rows)"` — over the same scope
    #: every other template field sees.
    #:
    #: Omitted, the node returns its single dependency's output unchanged, which
    #: is the common case and saves writing `"{the_only_node}"`.
    value: str | None = Field(
        default=None,
        description="Output node: template for the graph's return value. Defaults to its dependency's output.",
    )

    # function / python node: import path to the callable (module:attr or module.attr)
    callable: str | None = Field(default=None, description="Import path for function/python nodes.")
    # tool node: static kwargs merged with graph inputs + dep outputs before calling the tool
    tool_args: dict[str, Any] | None = Field(default=None, description="Static kwargs for tool nodes.")
    # Per-tool configuration: `{tool_name: {setting: value}}`.
    #
    # A **setting** is what the author decides once; `tool_args` is what gets sent
    # on a call. They are separate fields because they behave in opposite ways at
    # run time: a bound argument is *removed* from the schema the model sees
    # (see `bound_tools`), while a setting was never in that schema at all and
    # instead changes where the call lands (see `configured_tools`).
    #
    # Keyed by tool name because a node holds several. An agent that reads from
    # `/data/in` and writes to `/data/out` is an ordinary workflow, and a single
    # flat dict could not express it.
    tool_settings: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Per-tool configuration, keyed by tool name (e.g. a filesystem tool's root).",
    )
    # Stored environment values this node may use, referenced as `${NAME}` inside
    # `tool_args` and nowhere else.
    #
    # Declared rather than ambient, and confined to tool arguments, because the
    # one thing that must never happen to a database password is being
    # interpolated into a prompt: from there it is in the model's context, the
    # trace, and anything the trace is exported to. `{NAME}` cannot reach one —
    # secrets are never put in the interpolation scope — and validation rejects a
    # `${NAME}` in a prompt field, so the rule fails loudly rather than silently.
    secrets: list[str] = Field(
        default_factory=list,
        description="Names of stored secrets this node may use in tool_args as ${NAME}.",
    )

    policy: NodePolicy | None = Field(
        default=None,
        description="Optional per-node AgentConfig/policy overrides.",
    )
    # save node's output as files
    export: bool | None = Field(default=False, description="Whether to export the node's output as files.")
    export_path: str | None = Field(default=None, description="Optional custom path for exporting the node's output.")

    @field_validator("kind")
    def _kind_not_invalid(cls, v: str) -> str:
        v = v.strip()
        if v not in _VALID_NODE_KINDS:
            raise ValueError(f"node kind must be one of {sorted(_VALID_NODE_KINDS)}, got '{v}'")
        return v

    @field_validator("id")
    def _id_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("node id must not be empty")
        return v

    @field_validator("accumulate", mode="before")
    @classmethod
    def _coerce_accumulate(cls, v: object) -> object:
        # `accumulate` is the variable NAME to append each iteration's output to,
        # but LLMs routinely pass a boolean ("yes/no accumulate"). Coerce it:
        # False → None (don't accumulate); True → a sensible default var name.
        if v is False:
            return None
        if v is True:
            return "accumulated"
        return v


# Resolve the self-referential `body: list[GraphNode]` forward reference.
GraphNode.model_rebuild()


class Graph(BaseModel):
    name: str
    description: str | None = None

    # When True, the executor stops on the first failed node (LangChain's
    # RunnableSequence "short-circuit" behaviour). When False, all non-blocked
    # nodes still run and every failure is surfaced in GraphExecutionResult.
    fail_fast: bool = Field(
        default=False,
        description="Stop execution immediately on the first node failure.",
    )

    # When True, extra input keys not declared in `inputs` are rejected instead
    # of silently ignored (stricter boundary validation).
    strict_inputs: bool = Field(
        default=False,
        description="Reject undeclared input keys (default: warn and ignore).",
    )

    inputs: list[GraphInput] = Field(
        default_factory=list,
        description="Optional declared graph inputs that runtime 'inputs' must satisfy.",
    )

    functions: str | None = Field(
        default=None,
        description="Path to a Python file beside this graph, holding callables the "
                    "graph refers to by bare name (e.g. a loop's `until`). Resolved "
                    "relative to the graph file; copied with the package on export.",
    )

    nodes: list[GraphNode]
    outputs: list[str] = Field(default_factory=list)

    #: The imported `functions:` file. Set by the loader, which is the only place
    #: that knows the graph's directory; stays None for a graph built in memory,
    #: where callables are passed directly and need no lookup.
    _sidecar: Any = PrivateAttr(default=None)

    def bind_functions(self, base_dir: Any) -> None:
        """Import this graph's `functions:` file, resolved against *base_dir*."""
        if not self.functions:
            return
        from .functions import load_sidecar

        self._sidecar = load_sidecar(self.functions, base_dir)

    def function(self, name: str) -> Callable[..., Any] | None:
        """The sidecar callable *name*, or None when it is not one.

        None means "this string was prose", so every not-a-function case answers
        the same way — including a graph that declared no sidecar at all.
        """
        from .functions import resolve_name

        return resolve_name(self._sidecar, name)

    @property
    def sidecar(self) -> Any:
        """The loaded sidecar module, if any — for validators and error messages."""
        return self._sidecar

    @field_validator("nodes")
    def _as_kind_classes(cls, v: list[GraphNode]) -> list[GraphNode]:
        """Give every node the class its `kind` names — see `engine/nodes.py`.

        This is what makes `isinstance(node, RouterNode)` mean something. Without
        it the answer would depend on how the node was *built*: `RouterNode(...)`
        would pass and `GraphNode(kind="router")` — which is what YAML loading and 204
        existing call sites produce — would not. A type check that silently
        depends on the construction path is worse than no type check.

        Bodies are upgraded too, recursively, so a node inside a `map` is as
        identifiable as one at the top level.

        Imported here rather than at module scope: `nodes.py` imports `GraphNode`
        from this module, so a top-level import would be a cycle.
        """
        from .nodes import upgrade  # noqa: PLC0415 - see the docstring

        def _walk(node: GraphNode) -> GraphNode:
            out = upgrade(node)
            if out.body:
                out.body = [_walk(b) for b in out.body]
            return out

        return [_walk(n) for n in v]

    @field_validator("nodes")
    def _unique_node_ids(cls, v: list[GraphNode]) -> list[GraphNode]:
        ids = [n.id for n in v]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"duplicate node IDs in graph: {sorted(dupes)}")
        return v

    @field_validator("nodes")
    def _terminal_nodes_have_no_dependents(cls, v: list[GraphNode]) -> list[GraphNode]:
        """Nothing may depend on an ``output`` node.

        An output node *is* the end of the graph, so a step downstream of one is
        either a step that will never be reported or a second ending — and both
        are mistakes worth catching at load rather than after a run.
        """
        terminal = {n.id for n in v if n.kind in _TERMINAL_NODE_KINDS}
        if not terminal:
            return v
        for node in v:
            clash = terminal.intersection(node.depends_on)
            if clash:
                raise ValueError(
                    f"node '{node.id}' depends on {sorted(clash)}, which "
                    f"{'is an output node' if len(clash) == 1 else 'are output nodes'} — "
                    "nothing runs after the graph's output"
                )
            if node.on_error in terminal:
                raise ValueError(
                    f"node '{node.id}' routes errors to output node '{node.on_error}' — "
                    "an output node is the graph's return value, not an error handler"
                )
        return v

    @field_validator("inputs", mode="before")
    @classmethod
    def _normalize_inputs(cls, v):
        """
        Accept flexible YAML forms and normalize to a list[GraphInput]-compatible dicts.

        Supported:

        1) List of compact mappings:
            inputs:
              - topic_title: str
              - query:
                  type: string
                  required: true

        2) List of full specs:
            inputs:
              - name: topic_title
                type: str
                required: true

        3) Single mapping:
            inputs:
              topic_title: str
              query:
                type: string
                required: true
        """
        if v is None:
            return []

        # Case 3: single mapping
        if isinstance(v, dict):
            out = []
            for name, spec in v.items():
                if isinstance(spec, dict):
                    # e.g. query: { type: string, required: true }
                    merged = {"name": name, **spec}
                else:
                    # e.g. topic_title: str
                    merged = {"name": name, "type": spec}
                out.append(merged)
            return out

        # Expect list otherwise
        if not isinstance(v, list):
            raise TypeError("inputs must be a list or mapping")

        normalized = []
        for item in v:
            if isinstance(item, dict) and "name" not in item and len(item) == 1:
                # Compact mapping: { topic_title: str } OR { query: {type, required} }
                name, spec = next(iter(item.items()))
                if isinstance(spec, dict):
                    # { query: {type: string, required: true} }
                    merged = {"name": name, **spec}
                else:
                    # { topic_title: str }
                    merged = {"name": name, "type": spec}
                normalized.append(merged)
            else:
                # Already in normalized or near-normalized form
                normalized.append(item)
        return normalized

    @field_validator("inputs")
    @classmethod
    def _unique_input_names(cls, v: list[GraphInput]) -> list[GraphInput]:
        names = [f.name for f in v]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"duplicate graph input names: {sorted(dupes)}")
        return v

    def node_map(self) -> dict[str, GraphNode]:
        return {n.id: n for n in self.nodes}


class NodeExecutionResult(BaseModel):
    node_id: str
    mode: NodeMode
    raw_output: object
    structured_output: object | None = None
    tool_call_output: object | None = None
    started_at: float
    duration_ms: int
    error: str | None = None
    # True when the node was not run because an upstream dependency failed.
    skipped: bool = False
    skip_reason: str = ""
    traces: TraceResult | None = None
    # Token usage this node cost, including the engine's own hidden LLM calls
    # (router classification, loop `until` judging) and — for containers — the
    # sum of every body iteration. None for nodes that never called a model.
    usage: Usage | None = None
    #: Which model produced this node's tokens. Recorded per node because a node
    #: may name its own `provider`, and pricing a mixed graph at one rate would
    #: report the planner's cost for the worker's tokens.
    model: str | None = None
    # Names of tools the node invoked, in call order (react nodes and tool nodes).
    tool_calls: list[str] = Field(default_factory=list)
    # What this node was actually given, after template interpolation: the
    # resolved prompt for an agent node, the kwargs for a function/tool node.
    # Capped — this is for inspection, not a second copy of the payload.
    node_input: str | None = None

    @property
    def ok(self) -> bool:
        """True iff the node ran without error and was not skipped."""
        return not self.error and not self.skipped


class GraphExecutionResult(BaseModel):
    graph: Graph
    nodes: dict[str, NodeExecutionResult]
    final: dict[str, Any]
    # Nodes that failed (error is set and not skipped).
    errors: dict[str, str] = Field(default_factory=dict)
    # Nodes skipped due to upstream failures.
    skipped: list[str] = Field(default_factory=list)
    traces: TraceResult | None = None
    logs: str | None = None

    def total_usage(self) -> Usage:
        """Token usage summed across every node that called a model.

        Container nodes already fold in their body iterations, so body usage is
        counted once here, not twice.
        """
        total = Usage()
        for result in self.nodes.values():
            if result.usage is not None:
                total = total.add(result.usage)
        return total

    def total_cost(self, model: str | None = None) -> float | None:
        """List-price cost of the run, or ``None`` when the model is unpriced.

        Summed per node rather than from `total_usage()`, because nodes may name
        their own `provider` and a graph that mixes an expensive planner with a
        cheap worker would otherwise be priced entirely at one rate.
        """
        from neurosurfer.llm.pricing import estimate_cost

        total = 0.0
        priced_any = False
        for result in self.nodes.values():
            if result.usage is None:
                continue
            cost = estimate_cost(result.model or model, result.usage)
            if cost is not None:
                total += cost
                priced_any = True
        return total if priced_any else None

    @property
    def succeeded(self) -> bool:
        """True iff every non-skipped node completed without error."""
        return not self.errors

    def execution_summary(self, model: str | None = None) -> str:
        from neurosurfer.llm.pricing import format_cost

        total = len(self.nodes)
        ok = sum(1 for r in self.nodes.values() if r.ok)
        err = len(self.errors)
        skip = len(self.skipped)
        line = f"Graph '{self.graph.name}': {total} nodes — {ok} ok, {err} failed, {skip} skipped"
        cost = self.total_cost(model)
        return line if cost is None else f"{line} — {format_cost(cost)}"

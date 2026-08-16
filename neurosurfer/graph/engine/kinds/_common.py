"""Fields more than one kind shares, defined once.

A shared definition is not a convenience here — it is the reason the help text on
`when` says the same thing on a router as it does on a tool node. Eleven copies
of that sentence is eleven chances for one of them to drift into being wrong.

Kinds compose these into their own tuple rather than inheriting a base list, so a
kind that genuinely does not have a field simply does not name it — an `output`
node has no `writes` because it stores nothing, and that reads as an omission
rather than as an override.
"""

from __future__ import annotations

from .spec import FieldSpec

# ── wiring: on nearly every kind ────────────────────────────────────────────

DEPENDS_ON = FieldSpec(
    name="depends_on",
    type="node_ref",
    label="Depends on",
    help="Nodes that must finish before this one starts. Their outputs are what "
         "this node can read.",
    group="config",
)

# NOTE: `GraphNode.when` and `GraphNode.writes` exist and the executor honours
# both — `when` prunes a node as a not-taken branch, `writes` names its output as
# a run variable. Neither is offered (2026-08-02), and they came out together
# because they only make sense together.
#
# `when` takes an expression over run state: `vars.score > 7`. But `vars.score`
# exists only because some *other* node was given `writes: score`, and nothing on
# the canvas shows that it was. So the field asks for a name the studio never
# displays, produced by a field whose own purpose is only legible once you are
# writing the expression that needs it. Each is the other's prerequisite, and a
# panel cannot explain either one on its own — which is exactly how it read: two
# boxes wanting jargon, on the node you reach for first.
#
# They come back when there is a surface that makes the pair visible — a run
# variable a node can be pointed at, rather than a string typed in two places and
# matched by luck. Until then the graph branches with a `router`, which shows its
# branches on the canvas, and reads upstream output with `{node_id}`, which needs
# nothing declared.

ON_ERROR = FieldSpec(
    name="on_error",
    type="node_ref",
    label="On error, go to",
    help="Where to go if this node fails, instead of skipping everything "
         "downstream. The error text is readable as `<id>__error`.",
    group="config",
)

# ── the model call ──────────────────────────────────────────────────────────

INSTRUCTIONS = FieldSpec(
    name="instructions",
    type="text",
    label="What should this step do?",
    help="The system prompt for this step, written directly. Interpolates "
         "`{input}`, `{upstream_node}` and `{variable}` against what ran before it.",
    group="instruction",
)

PROVIDER = FieldSpec(
    name="provider",
    type="select",
    label="Provider",
    help="Which configured provider profile runs this node. Defaults to the "
         "profile the run was started with.",
    group="config",
)

# NOTE: `GraphNode.model` exists and is honoured, but is no longer offered.
# Which model a node runs is the *provider profile's* answer — a profile names a
# kind, an endpoint, a key and a model together, and they have to agree. A
# free-text override beside the profile picker let you name a model the endpoint
# does not serve, and read as though the two were independent choices. Pick a
# different profile instead; making one is two clicks in the Add panel.

MODE = FieldSpec(
    name="mode",
    type="select",
    label="Output mode",
    help="`text` returns prose. `structured` validates the answer against a "
         "schema and returns an object.",
    options=("text", "structured"),
    group="advanced",
)

OUTPUT_SCHEMA = FieldSpec(
    name="output_schema",
    type="json_schema",
    label="Output schema",
    help="The shape this step must return, as JSON Schema. Objects, arrays, "
         "enums and nested objects are all understood; `required` decides which "
         "properties must be present. An import path to a pydantic model "
         "(`my_module:ResultModel`) still works for a shape that already lives "
         "in code.",
    placeholder='{"type": "object", "properties": {…}, "required": […]}',
    required=True,
    show_when=("mode", ("structured",)),
    group="advanced",
)

# NOTE: `GraphNode.rag` exists and **nothing reads it**. `rag_agent` is accepted
# by the executor's constructor, stored, and never consulted either. So there is
# no RAG FieldSpec here: offering the toggle would promise retrieval that does
# not happen, which is worse than not offering it — a node would look configured
# and behave as though it were not.
#
# The field stays on the model (removing it is a wire-format change, and graphs
# on disk set it), and it is simply not claimed. Wiring retrieval up is a real
# piece of work; when it lands, this is where the toggle goes.

# NOTE: `GraphNode.policy` exists and the executor honours all of it — retries,
# timeout, temperature, max tokens. It is not offered because the only editor it
# ever had was a raw JSON box: the author had to know the key names, the shape
# was unvalidated, and a typo silently did nothing. Four useful settings behind a
# surface nobody could use is worse than four settings that are not offered yet.
# When they come back it should be as four named controls, not one textarea.

# ── tools and credentials ───────────────────────────────────────────────────

TOOLS_ATTACHED = FieldSpec(
    name="tools",
    type="tool_list",
    label="Tools",
    help="Tools this agent may call while it reasons.",
    group="config",
)

#: Per-tool configuration, keyed by tool name.
#:
#: On every kind that carries tools, because the question it answers — *where is
#: this tool allowed to work* — does not depend on who composes the call. A `tool`
#: node's `write_file` and a `react` node's attached `write_file` are the same
#: tool with the same need for a directory, and giving only one of them a place to
#: say so is how a field ends up configured in one kind and silently ignored in
#: another.
#:
#: Distinct from `tool_args` and not a subset of it: an argument is *sent*, and a
#: bound one is removed from the schema the model sees. A setting is never in that
#: schema at all — it changes where the call lands rather than what is in it.
TOOL_SETTINGS = FieldSpec(
    name="tool_settings",
    type="json",
    label="Tool settings",
    help="What each tool is configured with, decided once rather than per call — "
         "a filesystem tool's directory, for instance. The model never sees these "
         "and cannot choose them.",
    group="config",
)

SECRETS = FieldSpec(
    name="secrets",
    type="secret_list",
    label="Credentials",
    help="Stored credentials this node may use, written as `${NAME}` in its "
         "arguments. They never enter a prompt or a trace.",
    group="config",
)

# ── containers ──────────────────────────────────────────────────────────────

BODY = FieldSpec(
    name="body",
    type="body",
    label="Body",
    help="The nested sub-graph this node runs.",
    required=True,
    group="instruction",
)

BODY_OUTPUTS = FieldSpec(
    name="body_outputs",
    type="string_list",
    label="Body outputs",
    help="Which body nodes' outputs form the result. Empty means all of them.",
    group="advanced",
)

ITEM_VAR = FieldSpec(
    name="item_var",
    type="string",
    label="Bind each item as",
    help="The name the current item is available under inside the body.",
    placeholder="item",
    group="config",
)

# NOTE: `GraphNode.export` / `export_path` exist and the exporter honours them on
# the model path. They are not offered: writing a file is a side effect of the
# *run*, not a property of one step, and two advanced fields on every agent node
# bought a feature that a workflow's output already covers. The engine keeps
# reading them, so a graph on disk that exports still exports.

#: Wiring every non-terminal kind carries: what must finish before it, and where
#: to go if it fails. Both name another node, which is something the canvas can
#: draw — the two fields that came out of here named values it could not.
WIRING: tuple[FieldSpec, ...] = (DEPENDS_ON, ON_ERROR)

#: What every kind that calls a model carries beyond its instruction.
LLM: tuple[FieldSpec, ...] = (PROVIDER, MODE, OUTPUT_SCHEMA)

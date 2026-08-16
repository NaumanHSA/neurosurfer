"""`tool` — one registered tool, called directly. No model."""

from __future__ import annotations

from dataclasses import replace

from . import _common as c
from .spec import FieldSpec, NodeKindSpec

TOOL_ARGS = FieldSpec(
    name="tool_args",
    type="json",
    label="Arguments",
    help="The whole instruction for this node. Each argument is a literal, a "
         "reference to an upstream node or graph input, or a `${CREDENTIAL}`. "
         "Nothing composes these — there is no model call here.",
    # Not `required`, and the distinction is real: `required` means the engine
    # refuses the node without it, and a tool whose parameters all arrive from
    # graph inputs or an upstream output needs no `tool_args` at all. What
    # actually has to hold — every *required parameter of the chosen tool* is
    # supplied from somewhere — is checked against that tool's own schema in
    # `validate._check_tool_args`, which a flag on a field cannot express.
    group="instruction",
)

SPEC = NodeKindSpec(
    kind="tool",
    label="Tool",
    blurb="Calls one registered tool with the arguments you bind. No LLM call.",
    shape="tool",
    data_arrival=("bound_args",),
    fields=(
        replace(c.TOOLS_ATTACHED, required=True, label="Tool",
                help="The tool this node invokes."),
        TOOL_ARGS,
        c.TOOL_SETTINGS,
        c.SECRETS,
        *c.WIRING,  # no `export` — see function.py
    ),
    constraints=(
        "There is no LLM call, so nothing composes its arguments and nothing "
        "reads an instruction. If an argument has to be *worked out* — a query, "
        "a search phrase, a request body — the step is a `react` node with that "
        "tool attached.",
        "tool_args supplying every required parameter of that tool — from an "
        "argument here, or a graph input or upstream output of the same name. "
        "One missing calls the tool with nothing and fails at run time.",
        "A credential goes in `secrets` and is written `${NAME}` in an "
        "argument, never in a prompt.",
    ),
)

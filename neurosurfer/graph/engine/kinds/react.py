"""`react` — an LLM that calls tools in a loop."""

from __future__ import annotations

from dataclasses import replace

from . import _common as c
from .spec import FieldSpec, NodeKindSpec

#: Bound arguments, not the whole call: the engine supplies these on every call
#: the model makes and removes them from the schema it is offered. This is how a
#: react node uses a credential without the model ever seeing it.
BOUND_ARGS = FieldSpec(
    name="tool_args",
    type="json",
    label="Bound arguments",
    help="Arguments supplied on every tool call this agent makes, and hidden "
         "from the schema it is offered. How a credential reaches a tool "
         "without reaching the model.",
    group="config",
)

SPEC = NodeKindSpec(
    kind="react",
    label="ReAct Agent",
    blurb="An agent that reasons and calls tools, for work that must touch the "
          "outside world and needs a model to decide what to send.",
    shape="agent",
    calls_model=True,
    data_arrival=("prompt", "bound_args"),
    fields=(
        c.INSTRUCTIONS,
        replace(c.TOOLS_ATTACHED, required=True,
                help="Tools this agent may call. A react node with none cannot "
                     "act, and is refused."),
        BOUND_ARGS,
        c.TOOL_SETTINGS,
        c.SECRETS,
        *c.WIRING,
        # `provider` only — **not** `mode` or `output_schema`.
        #
        # `run_react_node` takes neither, and the executor passes neither: a react
        # node's structured-output settings were inert, offered by the panel,
        # written into the YAML, reviewed, and read by nothing. That is the same
        # defect Phase 1 found five of — a spec promising a field the engine
        # ignores — and it is worse here than usual, because "return an object"
        # silently returning prose is only discovered by whatever consumes it.
        #
        # Making the loop honour a schema is a real feature and a fine one; until
        # it exists the honest thing is not to offer the field.
        c.PROVIDER,
    ),
    constraints=(
        "The only kind that both reasons and acts. `base` reasons and cannot "
        "act; `tool` acts and cannot reason.",
    ),
)

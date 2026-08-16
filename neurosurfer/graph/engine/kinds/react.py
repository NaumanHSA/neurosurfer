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
    #: Unbounded here — the loop runs until it is done or the guardrails stop it.
    #: `None` rather than a large number: "as many as it takes" is the property,
    #: and a number would invite a consumer to render a limit that is not one.
    tool_rounds=None,
    fields=(
        c.INSTRUCTIONS,
        replace(c.TOOLS_ATTACHED, required=True,
                help="Tools this agent may call. A react node with none cannot "
                     "act, and is refused."),
        BOUND_ARGS,
        c.TOOL_SETTINGS,
        c.SECRETS,
        *c.WIRING,
        # `output_schema` is offered again, and now it is honoured.
        #
        # It was withdrawn because it was inert: offered by the panel, written
        # into the YAML, reviewed, and read by nothing — and "return an object"
        # silently returning prose is only discovered by whatever consumes it.
        # The note then said *"making the loop honour a schema is a real feature
        # and a fine one; until it exists the honest thing is not to offer the
        # field."* It exists now.
        #
        # The loop runs first and its answer is shaped by one structured call
        # afterwards, rather than constraining every turn — a react node decides
        # what to do next from what the last tool returned, and a schema on every
        # turn would break exactly that. The cost is one extra model call, billed
        # to the node.
        c.OUTPUT_SCHEMA,
        c.PROVIDER,
    ),
    constraints=(
        "The only kind that both reasons and acts. `base` reasons and cannot "
        "act; `tool` acts and cannot reason.",
    ),
)

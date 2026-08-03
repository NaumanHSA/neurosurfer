"""`base` — one LLM call."""

from __future__ import annotations

from . import _common as c
from .spec import NodeKindSpec

SPEC = NodeKindSpec(
    kind="base",
    label="Agent",
    blurb="One LLM call — writing, summarising, classifying, transforming text.",
    shape="agent",
    calls_model=True,
    data_arrival=("prompt", "bound_args"),
    #: `tools` was missing here, and the spec was the only place it was.
    #:
    #: `run_node_runner.run_base_node` is documented as "one LLM call, optionally
    #: with tools … the rung between `base` and `react`" — it offers the pool,
    #: allows a single round (`max_tool_rounds=1`), and feeds the result back. The
    #: studio has always let you attach one (`canAttachTool` accepts `base`). Only
    #: the declaration disagreed, which is exactly the defect class these specs
    #: exist to stop: the engine reads a field the spec does not claim, so every
    #: consumer deriving from the spec quietly skips it — validation included.
    fields=(c.INSTRUCTIONS, c.TOOLS_ATTACHED, c.TOOL_SETTINGS, c.SECRETS, *c.WIRING, *c.LLM),
    constraints=(
        "Say what the step should do in `instructions`. The older "
        "purpose/goal/expected_result trio is still read when it is absent.",
        "For structured output set an output schema — without one, a node asked "
        "for an object returns JSON as a string.",
        "Tools are optional here and the model gets one round with them. A step "
        "that must call tools repeatedly to finish its job is a `react` node.",
    ),
)

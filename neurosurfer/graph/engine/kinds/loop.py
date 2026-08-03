"""`loop` — repeats a nested body until a condition, always capped."""

from __future__ import annotations

from dataclasses import replace

from . import _common as c
from .spec import FieldSpec, NodeKindSpec

MAX_ITERATIONS = FieldSpec(
    name="max_iterations",
    type="int",
    label="Maximum iterations",
    help="A hard ceiling, always required. It is what stops a loop whose "
         "condition never becomes true.",
    required=True,
    group="instruction",
)

UNTIL = FieldSpec(
    name="until",
    type="string",
    label="Stop when (plain English)",
    help="Judged by an internal LLM decision after each iteration. The reason "
         "it gives for continuing reaches the next iteration as `{feedback}`.",
    placeholder="e.g. the review approves the draft",
    group="instruction",
)

BREAK_WHEN = FieldSpec(
    name="break_when",
    type="expression",
    label="Stop when (expression)",
    help="Evaluated after each iteration. Deterministic and free — no LLM call.",
    placeholder="e.g. vars.score >= 8",
    group="instruction",
)

ACCUMULATE = FieldSpec(
    name="accumulate",
    type="string",
    label="Collect iterations into",
    help="A variable name to append each iteration's output to, as a list.",
    group="config",
)

SPEC = NodeKindSpec(
    kind="loop",
    label="Loop",
    blurb="Runs a nested body over and over until a condition holds, up to a "
          "hard ceiling.",
    shape="cycle",
    has_body=True,
    data_arrival=("prompt", "expression"),
    fields=(
        c.BODY, MAX_ITERATIONS, UNTIL, BREAK_WHEN, ACCUMULATE,
        replace(c.ITEM_VAR, label="Bind last output as",
                help="The name the previous iteration's output is available "
                     "under inside the body."),
        c.BODY_OUTPUTS,
        *c.WIRING,
    ),
    constraints=(
        "Set `until` or `break_when`, never both. Neither means it runs to the "
        "ceiling.",
        "`until` costs one LLM call per iteration; `break_when` costs nothing.",
    ),
)

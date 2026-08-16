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
    label="Stop when",
    help="Either the name of a function in the graph's `functions:` file — it "
         "receives a LoopIteration and returns True to stop, deterministic and "
         "free — or a plain-English condition, judged by an internal LLM "
         "decision after each iteration, whose reason for continuing reaches "
         "the next iteration as `{feedback}`. Which one it is is looked up, not "
         "guessed: a name the functions file defines is the function.",
    placeholder="e.g. the review approves the draft — or: tagline_is_short",
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
        c.BODY, MAX_ITERATIONS, UNTIL, ACCUMULATE,
        replace(c.ITEM_VAR, label="Bind last output as",
                help="The name the previous iteration's output is available "
                     "under inside the body."),
        c.BODY_OUTPUTS,
        *c.WIRING,
    ),
    constraints=(
        "No `until` means it runs to the ceiling.",
        "A plain-English `until` costs one LLM call per iteration; a function "
        "costs nothing. Prefer a function whenever the condition is checkable "
        "in code.",
        "A plain-English `until` that is about a different subject than the body "
        "produces stops the loop with a warning rather than running to the "
        "ceiling — so the condition must actually describe the body's output.",
    ),
)

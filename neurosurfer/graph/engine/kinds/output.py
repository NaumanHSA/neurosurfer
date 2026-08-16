"""`output` — what the graph returns. Terminal."""

from __future__ import annotations

from . import _common as c
from .spec import FieldSpec, NodeKindSpec

VALUE = FieldSpec(
    name="value",
    type="text",
    label="Return value",
    help="A template over graph inputs, upstream outputs and variables — "
         "`\"{summary} ({rows} rows)\"`. Leave empty to pass this node's single "
         "dependency through unchanged, keeping its type.",
    group="instruction",
)

SPEC = NodeKindSpec(
    kind="output",
    label="Output",
    blurb="Declares what the workflow returns. The graph stops here.",
    shape="terminus",
    terminal=True,
    data_arrival=("passthrough", "prompt"),
    # No `writes` (it stores nothing), no `on_error` (it is terminal), no tools
    # (it runs nothing). `when` went with the others — a router already decides
    # which branch is live, and a pruned branch's output node is pruned with it,
    # so the guard was a second way to say what the graph shape already says.
    fields=(VALUE, c.DEPENDS_ON),
    constraints=(
        "Needs something to return: a dependency whose output it passes "
        "through, or a value template. With neither it returns nothing.",
        "Nothing may depend on one, and it cannot be an error target.",
        "An unresolved placeholder in the value is refused, not warned about — "
        "this text is what a caller receives.",
        "Takes precedence over the older graph-level `outputs:` list.",
    ),
)

"""`subgraph` — runs a nested body once. Composition."""

from __future__ import annotations

from . import _common as c
from .spec import NodeKindSpec

SPEC = NodeKindSpec(
    kind="subgraph",
    label="Subgraph",
    blurb="A workflow inside a workflow: runs a nested body once, and its final "
          "outputs become this node's output.",
    shape="nested",
    has_body=True,
    data_arrival=("prompt",),
    fields=(c.BODY, c.BODY_OUTPUTS, *c.WIRING),
    constraints=(
        "Body nodes may only depend on their siblings — a dependency pointing "
        "outside the body is refused when the graph loads.",
    ),
)

"""`map` — runs a nested body once per item of a collection."""

from __future__ import annotations

from . import _common as c
from .spec import FieldSpec, NodeKindSpec

OVER = FieldSpec(
    name="over",
    type="expression",
    label="Over",
    help="An expression yielding the collection to fan out over — usually an "
         "upstream node's output or a graph input.",
    required=True,
    placeholder="e.g. nodes.split_sections",
    group="instruction",
)

CONCURRENCY = FieldSpec(
    name="concurrency",
    type="int",
    label="Concurrency",
    help="How many items to process at once. 1 runs them in order.",
    group="config",
)

SPEC = NodeKindSpec(
    kind="map",
    label="Map",
    blurb="Runs a nested body once per item of a collection, and returns the "
          "ordered list of results.",
    shape="cycle",
    has_body=True,
    data_arrival=("expression",),
    fields=(
        c.BODY, OVER, c.ITEM_VAR, CONCURRENCY, c.BODY_OUTPUTS, *c.WIRING,
    ),
    constraints=(
        "The output is the ordered list of per-item results — the gather is "
        "implicit, so no node is needed to collect them.",
    ),
)

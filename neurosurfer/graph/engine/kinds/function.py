"""`function` — deterministic Python, imported and called."""

from __future__ import annotations

from . import _common as c
from .spec import FieldSpec, NodeKindSpec

CALLABLE = FieldSpec(
    name="callable",
    type="import_path",
    label="Function to call",
    help="Import path to a Python callable. It is called with the graph's "
         "inputs and this node's dependency outputs as keyword arguments, "
         "matched by name.",
    required=True,
    placeholder="my_module:transform",
    group="instruction",
)

SPEC = NodeKindSpec(
    kind="function",
    label="Function",
    blurb="Deterministic Python: imports a callable and calls it with the "
          "inputs and upstream outputs that match its signature.",
    shape="code",
    data_arrival=("kwargs",),
    # No `export`: the exporter is only consulted on the LLM path
    # (`_run_node_native`), so `export: true` on a function node is a setting
    # that silently does nothing. Offering it would be the spec asserting a
    # capability the engine does not have, which is the failure it exists to
    # prevent. Making export work everywhere is a fair change — it is just not
    # true yet, so it is not claimed here.
    fields=(CALLABLE, *c.WIRING),
    constraints=(
        "Arguments are matched to the signature **by name**, so a parameter "
        "whose name is not a graph input or an upstream node id receives "
        "nothing. Nothing warns about this today — see plan 01, phase 3.",
    ),
)

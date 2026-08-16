"""`router` — selects one branch; the rest are pruned."""

from __future__ import annotations

from . import _common as c
from .spec import FieldSpec, NodeKindSpec

ROUTES = FieldSpec(
    name="routes",
    type="node_ref_map",
    label="Routes",
    help="Label → target node. The router itself classifies with one LLM call "
         "and picks a label; every other target is pruned.",
    group="instruction",
)

CASES = FieldSpec(
    name="cases",
    type="case_list",
    label="Cases",
    help="Ordered predicates, first match wins. No LLM call — free and "
         "reproducible. The deterministic alternative to routes.",
    group="instruction",
)

REPAIR = FieldSpec(
    name="repair",
    type="bool",
    label="Retry an invalid answer",
    help="Routes only: when the model picks a label that does not exist, retry "
         "once with corrective feedback before falling back to the default.",
    group="advanced",
)

DEFAULT = FieldSpec(
    name="default",
    type="node_ref",
    label="Default",
    help="Where to go when no route or case matches.",
    group="config",
)

SPEC = NodeKindSpec(
    kind="router",
    label="Router",
    blurb="Picks one downstream branch — by LLM classification, or by "
          "deterministic predicates.",
    shape="branch",
    calls_model=True,  # only in the `routes` form; `cases` makes no call
    data_arrival=("prompt", "expression"),
    # `provider` only — not the rest of the LLM set. A router's call is a
    # fixed-format classification through `run_base_node`: it asks for one label
    # and matches it against `routes`, so there is no `output_schema` to validate
    # against, no `mode` to choose, no retrieval and no policy plumbed through.
    # Offering those would be controls writing values nothing reads. `model` went
    # with them for the reason it went everywhere else: the profile names it.
    fields=(
        c.INSTRUCTIONS, ROUTES, CASES, DEFAULT, REPAIR,
        *c.WIRING, c.PROVIDER,
    ),
    constraints=(
        "Declare `routes` or `cases`, never both — routes classify with a "
        "model, cases evaluate expressions.",
        "Every target must declare this router in its `depends_on`.",
        "The `cases` form makes no LLM call, so the provider here means "
        "nothing for it.",
    ),
)

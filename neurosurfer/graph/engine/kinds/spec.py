"""What a node kind *is*, as data rather than as eleven special cases.

``GraphNode`` is one model with ~40 optional fields shared by every kind: a
``tool`` node carries ``until``, ``over``, ``routes`` and ``options`` as ``None``
because a ``loop`` and a ``router`` needed them. That shape is convenient for the
executor — one model, one loader, no unions on the wire — and it leaves every
*other* consumer to work out for itself which of those forty fields a given kind
actually uses. They each did, separately:

* the studio hand-writes a config section per kind, and forgot five of them;
* validation is a pile of ``_check_*`` functions keyed on ``kind``;
* the Architect's manifest hand-maintains a ``key_fields`` list per kind;
* and adding a kind means editing the model, the executor, the validator, the
  Inspector, the Add panel and the card-shape table — six places, with no
  compiler error for missing any.

A :class:`NodeKindSpec` is that knowledge stated once. It is **descriptive**: the
storage and the wire format are still ``GraphNode``, so every workflow on disk
keeps loading and no migration is implied. What changes is that the consumers
above read this instead of each restating it.

The test of whether this worked is in ``tests/engine/test_node_kinds.py``: adding
a kind should mean adding one module here, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["FieldKind", "FieldSpec", "NodeKindSpec", "DataArrival"]


#: How a field's value is written, which is what tells a client how to edit it.
#:
#: Deliberately a small vocabulary. Every member here has to be renderable as a
#: control by anything consuming the spec, so a type that exists to describe one
#: field of one kind is a type that will be rendered as a raw text box anyway —
#: better to say ``json`` and mean it.
FieldKind = Literal[
    "string",       # one line of free text
    "text",         # multi-line prose (a prompt, an instruction)
    "int",
    "bool",
    "select",       # one of `options`
    "expression",   # a restricted expression over run state — see engine.expressions
    "import_path",  # `module:attr` — resolved and checked by validation
    "node_ref",     # the id of another node in the same scope
    "node_ref_map", # {label: node_id} — a router's `routes`
    "case_list",    # [{when, to}] — a router's deterministic branches
    "string_list",
    "tool_list",    # names from the tool registry
    "secret_list",  # names of stored credentials
    "json",         # an object whose shape belongs to something else (a tool, a policy)
    "json_schema",  # a JSON Schema object describing a shape — see engine.json_schema
    "body",         # nested nodes — a container's sub-graph
]

#: How run data reaches a node, which is the question every author has at the
#: moment they wire an edge and the one the canvas could never answer.
DataArrival = Literal[
    "prompt",       # interpolated into an LLM prompt: `{name}`, `inputs.`, `nodes.`, `vars.`
    "bound_args",   # bound per parameter before the call (`tool_args`)
    "kwargs",       # splatted into a Python signature by name
    "expression",   # read by an expression the author writes (`over`, `when`)
    "supplied",     # provided from outside the graph — a person, or the caller
    "passthrough",  # its dependency's output, unchanged
]


@dataclass(frozen=True)
class FieldSpec:
    """One configurable field of one node kind.

    ``name`` is the attribute on :class:`~neurosurfer.graph.engine.schema.GraphNode`.
    It is checked against the real model by a test rather than trusted, because a
    spec that names a field the engine does not have is worse than no spec: it
    renders an editor for a value nothing will ever read.
    """

    name: str
    type: FieldKind
    label: str
    help: str = ""
    #: Required *for this kind*. `depends_on` is optional on a `base` node and the
    #: whole point of an `output` one, so requiredness lives here and not on the
    #: model — which is also why the engine's own model cannot express it.
    required: bool = False
    placeholder: str | None = None
    #: For `select`.
    options: tuple[str, ...] = ()
    #: Editors are grouped in this order: what the step does, how it is wired,
    #: then the things most nodes never set.
    group: Literal["instruction", "config", "advanced"] = "config"
    #: ``(field, values)`` — this field applies only while a sibling field holds
    #: one of ``values``.
    #:
    #: It governs **both** halves at once: a field that does not apply is not
    #: offered, and is not required. Those have to move together — a schema box
    #: hidden because the node returns prose, but still marked required, is a node
    #: that cannot be made valid through the surface that hid it.
    show_when: tuple[str, tuple[str, ...]] | None = None

    def applies_to(self, node: Any) -> bool:
        """Whether this field is live for ``node``'s current configuration."""
        if self.show_when is None:
            return True
        field, values = self.show_when
        return str(getattr(node, field, None) or "") in values

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "label": self.label,
            "help": self.help,
            "required": self.required,
            "group": self.group,
        }
        if self.placeholder is not None:
            d["placeholder"] = self.placeholder
        if self.options:
            d["options"] = list(self.options)
        if self.show_when is not None:
            d["show_when"] = {"field": self.show_when[0], "values": list(self.show_when[1])}
        return d


@dataclass(frozen=True)
class NodeKindSpec:
    """One node kind, completely.

    Everything a consumer needs to render, validate, or describe a node of this
    kind without knowing anything else about it.
    """

    kind: str
    label: str
    blurb: str
    #: The silhouette this kind is drawn as. Shape carries the category so colour
    #: is not the only channel — it survives a zoomed-out canvas, a projector, and
    #: the readers who cannot separate two of the accents.
    shape: Literal["agent", "branch", "cycle", "nested", "code", "io", "terminus", "tool"]
    #: Does running this node make an LLM call? Decides whether a provider/model
    #: choice means anything here.
    calls_model: bool = False
    #: Does it carry a nested `body` sub-graph?
    has_body: bool = False
    #: Does the graph stop here? Nothing may depend on a terminal node.
    terminal: bool = False
    #: How upstream data reaches it — the three mechanisms the engine actually
    #: has, stated per kind rather than discovered by reading the executor.
    data_arrival: tuple[DataArrival, ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    #: Constraints a field list cannot express ("a plain-English `until` costs a call").
    #: Prose, for a human or a model reading the spec; the machine-checkable half
    #: is `required` on the fields themselves.
    constraints: tuple[str, ...] = ()

    def field(self, name: str) -> FieldSpec | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    @property
    def required_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields if f.required)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "blurb": self.blurb,
            "shape": self.shape,
            "calls_model": self.calls_model,
            "has_body": self.has_body,
            "terminal": self.terminal,
            "data_arrival": list(self.data_arrival),
            "fields": [f.as_dict() for f in self.fields],
            "constraints": list(self.constraints),
        }

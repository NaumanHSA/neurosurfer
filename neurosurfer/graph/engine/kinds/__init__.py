"""The node-kind registry: every kind this engine has, as a typed spec.

One module per kind, each exporting a ``SPEC``. Adding a kind means adding a
module here and naming it in ``_MODULES`` — the validator, the ``/v1/node-kinds``
API and the studio's config editors all derive from that, and none of them needs
to change. ``tests/engine/test_node_kinds.py`` is what holds that promise up: it
fails if a kind exists in the engine with no spec, or if a spec names a field
``GraphNode`` does not have.

The specs describe ``GraphNode``; they do not replace it. Storage and the wire
format are unchanged, so every workflow already on disk keeps loading.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .spec import DataArrival, FieldKind, FieldSpec, NodeKindSpec

__all__ = [
    "FieldKind",
    "FieldSpec",
    "NodeKindSpec",
    "DataArrival",
    "NODE_KIND_SPECS",
    "node_kind_spec",
    "all_kind_specs",
    "kind_specs_as_dicts",
]

#: Module names, not the specs themselves — importing them by name keeps `map`
#: and `input` from shadowing the builtins in this namespace.
_MODULES = (
    "base", "react", "function", "python", "tool",
    "router", "loop", "map", "subgraph", "input", "output",
)


def _load() -> dict[str, NodeKindSpec]:
    specs: dict[str, NodeKindSpec] = {}
    for name in _MODULES:
        module = import_module(f"{__name__}.{name}")
        spec: NodeKindSpec = module.SPEC
        specs[spec.kind] = spec
    return specs


#: Every kind, by name.
NODE_KIND_SPECS: dict[str, NodeKindSpec] = _load()


def node_kind_spec(kind: str) -> NodeKindSpec | None:
    """The spec for *kind*, or ``None`` if the engine has no such kind."""
    return NODE_KIND_SPECS.get(kind)


def all_kind_specs() -> list[NodeKindSpec]:
    """Every spec, in the order kinds are meant to be presented."""
    return [NODE_KIND_SPECS[name] for name in _MODULES]


def kind_specs_as_dicts() -> list[dict[str, Any]]:
    """Every spec as plain JSON-able data, for the API."""
    return [spec.as_dict() for spec in all_kind_specs()]

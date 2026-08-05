"""`Router(...)` beside `GraphNode(kind="router")`, and `isinstance` meaning it.

The point of these classes is dispatch and readability, so the thing worth
pinning is not that `Router` exists — it is that **`isinstance` does not depend
on how the node was built**. A type check that passes for one construction path
and fails for another is worse than no type check, because it fails silently and
only for graphs that came in through the other door.
"""

from __future__ import annotations

import pytest

from neurosurfer.graph.engine.loader import load_graph_from_dict
from neurosurfer.graph.engine.nodes import (
    Base,
    Container,
    Input,
    Loop,
    Map,
    Output,
    React,
    Router,
    Subgraph,
    for_kind,
)
from neurosurfer.graph.engine.schema import Graph, GraphNode


def test_the_class_and_the_kind_argument_build_the_same_thing():
    a = Graph(name="t", outputs=["x"], nodes=[
        Router(id="r", routes={"a": "x"}), Base(id="x", depends_on=["r"])])
    b = Graph(name="t", outputs=["x"], nodes=[
        GraphNode(id="r", kind="router", routes={"a": "x"}),
        GraphNode(id="x", kind="base", depends_on=["r"])])

    assert type(a.nodes[0]) is type(b.nodes[0]) is Router
    assert a.nodes[0].model_dump() == b.nodes[0].model_dump()


def test_a_graph_loaded_from_yaml_gets_the_classes_too():
    """The path that matters most — 204 call sites and every workflow on disk
    arrive as `kind=` strings, not as class instances."""
    g = load_graph_from_dict({
        "name": "t",
        "inputs": [{"name": "items", "type": "array"}],
        "nodes": [{"id": "fan", "kind": "map", "over": "inputs.items",
                   "body": [{"id": "one", "kind": "base", "goal": "do {item}"}]}],
        "outputs": ["fan"],
    })
    assert isinstance(g.nodes[0], Map)
    assert isinstance(g.nodes[0].body[0], Base), "bodies are upgraded too"


def test_the_wire_format_is_unchanged():
    """These classes are a second door into the same room. If `model_dump` ever
    stops producing `kind: router`, every workflow on disk breaks."""
    dumped = Router(id="r", routes={"a": "x"}).model_dump(exclude_none=True)
    assert dumped["kind"] == "router"
    assert GraphNode.model_validate(dumped).kind == "router"


@pytest.mark.parametrize(
    ("cls", "kind"),
    [(Base, "base"), (React, "react"), (Router, "router"), (Loop, "loop"),
     (Map, "map"), (Subgraph, "subgraph"), (Input, "input"), (Output, "output")],
)
def test_every_class_pins_its_kind(cls, kind):
    assert cls(id="n").kind == kind
    assert for_kind(kind) is cls


def test_container_answers_has_a_body_in_one_question():
    """`node.kind in {"loop", "map", "subgraph"}` was written wherever this was
    needed, and a set repeated is a set that falls out of date."""
    assert isinstance(Loop(id="n"), Container)
    assert isinstance(Map(id="n"), Container)
    assert isinstance(Subgraph(id="n"), Container)
    assert not isinstance(Base(id="n"), Container)
    assert not isinstance(Router(id="n"), Container)


def test_a_kind_with_no_class_still_loads():
    """Additive by construction: a kind the engine gains before this module
    hears about it must keep working, as a plain `GraphNode`."""
    assert for_kind("some_future_kind") is GraphNode


def test_every_engine_kind_is_covered_or_deliberately_not():
    """A tripwire: if the engine gains a kind, this fails until someone decides
    whether it gets a class — rather than it silently having none."""
    from neurosurfer.graph.engine.schema import _VALID_NODE_KINDS

    missing = {k for k in _VALID_NODE_KINDS if for_kind(k) is GraphNode}
    assert not missing, f"node kinds with no class: {sorted(missing)}"

"""A node kind as a class: `Router(...)` beside `GraphNode(kind="router")`.

## Why both

`GraphNode` is the wire format. Every workflow on disk, every YAML fixture and
204 call sites in this repo construct it with `kind="router"`, and there is no
version of "specialised classes" worth breaking those for.

So these are **not a replacement**. They are a second door into the same room:

    Router(id="triage", routes={...})           # says what it is
    GraphNode(id="triage", kind="router", ...)  # unchanged, still works

and, because `Graph` upgrades whatever it is given (see `schema.Graph`), both
produce a `Router` instance — so `isinstance(node, Router)` is true regardless of
which door was used, including for a graph loaded from YAML.

## What they are for

Two things, and it is worth being precise because a class hierarchy that promises
more than it delivers is worse than none:

- **Reading.** `if node.kind == "router"` scattered through an executor says
  nothing at the point where a node is *constructed*. `Router(...)` does.
- **Dispatch.** `isinstance(node, Container)` asks "does this run a nested body"
  once, instead of `node.kind in {"loop", "map", "subgraph"}` in each place that
  needs to know — a set that has been wrong before, because nothing made the
  places agree.

## What they deliberately do not do yet

**They do not narrow fields.** `Router.routes` is still `dict | None`, inherited,
not required. Making it required here would put a third source of truth beside
`GraphNode` and `engine/kinds/`, and the plan's own §0.7 diagnosis is that *six*
places already had to agree about a kind's fields with nothing forcing them to.
`engine/kinds/` is that declaration; these classes carry identity, and the specs
carry the contract. Narrowing is a follow-up that should generate from the specs
rather than restate them.
"""

from __future__ import annotations

from typing import Literal

from .schema import GraphNode

__all__ = [
    "Base", "React", "Tool", "Function", "Python",
    "Router", "Loop", "Map", "Subgraph",
    "Input", "Output",
    "Container", "for_kind",
]


# ── work ─────────────────────────────────────────────────────────────────────

class Base(GraphNode):
    """One LLM call. Optionally with tools — but only **one round** of them.

    That round limit is the whole difference from `React`: a `Base` step can
    "fetch this, then tell me about it" and cannot "fetch this, then decide what
    to fetch next". Asked to do the second it fails rather than half-finishing.
    """

    kind: Literal["base"] = "base"


class React(GraphNode):
    """An LLM that calls tools in a loop, until it has an answer.

    With no tools it cannot act, so the executor refuses to run one — a react
    node with an empty toolbelt describes work it never did.
    """

    kind: Literal["react"] = "react"


class Tool(GraphNode):
    """One registered tool, called directly. No model, so nothing composes the
    arguments: `tool_args` is the entire instruction."""

    kind: Literal["tool"] = "tool"


class Function(GraphNode):
    """A Python callable, imported by path. Deterministic; no model."""

    kind: Literal["function"] = "function"


class Python(GraphNode):
    """Alias of `Function` in the engine — both route to the same runner."""

    kind: Literal["python"] = "python"


# ── control flow ─────────────────────────────────────────────────────────────

class Router(GraphNode):
    """Picks one branch and prunes the rest.

    The router **is** the classifier: with `routes`, one LLM call chooses a
    label; with `cases`, an expression decides and no model is called at all.

    Two rules the validator enforces, both learned the hard way: every target
    must exist, and every target must list this node in its `depends_on` — a
    target that does not wait for the router is an independent node in the same
    layer, so it runs whichever branch was chosen.

    Branches are **not** nested inside the router, deliberately. They are
    ordinary nodes that other nodes may depend on, which is what makes the
    diverge-and-rejoin shape possible; nesting them would put a branch target
    out of reach of the node that joins on it.
    """

    kind: Literal["router"] = "router"


class Loop(GraphNode):
    """Runs its `body` again until `break_when`, `until`, or `max_iterations`.

    Both bounds matter: the condition is the intent, the ceiling is the
    guarantee that a model which never satisfies it still terminates.
    """

    kind: Literal["loop"] = "loop"


class Map(GraphNode):
    """Runs its `body` once per item of `over`, in parallel."""

    kind: Literal["map"] = "map"


class Subgraph(GraphNode):
    """Runs a nested graph as a single node."""

    kind: Literal["subgraph"] = "subgraph"


# ── boundaries ───────────────────────────────────────────────────────────────

class Input(GraphNode):
    """Declares what the workflow takes, and can pause to ask a person."""

    kind: Literal["input"] = "input"


class Output(GraphNode):
    """Declares what the workflow returns. Terminal — nothing may depend on it."""

    kind: Literal["output"] = "output"


# ── dispatch helpers ─────────────────────────────────────────────────────────

#: The kinds that run a nested `body`.
#:
#: A tuple for `isinstance`, so "does this node have a body" is one question
#: rather than a `{"loop", "map", "subgraph"}` literal repeated wherever it is
#: needed — which is the shape that lets one copy fall out of date silently.
Container = (Loop, Map, Subgraph)


_BY_KIND: dict[str, type[GraphNode]] = {
    c.model_fields["kind"].default: c
    for c in (Base, React, Tool, Function, Python,
              Router, Loop, Map, Subgraph, Input, Output)
}


def for_kind(kind: str) -> type[GraphNode]:
    """The class for *kind*, or `GraphNode` for one that has no class yet.

    Falling back rather than raising is what keeps this additive: a kind the
    engine gains before this module hears about it still loads, as a plain
    `GraphNode`, exactly as everything did before these classes existed.
    """
    return _BY_KIND.get(kind, GraphNode)


def upgrade(node: GraphNode) -> GraphNode:
    """Return *node* as its kind's class, or unchanged if it already is one.

    This is what makes `isinstance` trustworthy: a graph built with
    `GraphNode(kind="router")`, or loaded from YAML, still yields a `Router`.
    Without it the check would silently depend on how the node happened to be
    constructed, which is worse than not having the classes at all.
    """
    cls = for_kind(getattr(node, "kind", ""))
    if type(node) is cls or not isinstance(node, GraphNode):
        return node
    if isinstance(node, cls):
        return node
    return cls.model_validate(node.model_dump())

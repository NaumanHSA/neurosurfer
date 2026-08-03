"""Per-placeholder prompt rendering.

`str.format` is all-or-nothing: one unknown name raises and the caller is handed back
the raw template, so a single typo used to cost a node *every* value in the same field.
`render_template` fills what it can and leaves the rest as written.

The contract has two halves, and both are pinned here: on the success path it must be
indistinguishable from `str.format` (nobody's working prompt may change), and on the
failure path it must lose only the placeholder that actually failed.
"""

from __future__ import annotations

import pytest

from neurosurfer.graph import Graph, GraphNode
from neurosurfer.graph.engine.executor import GraphExecutor
from neurosurfer.graph.engine.templates import render_template

from ..fakes import ScriptedProvider

# ── it must not change a prompt that already worked ─────────────────────────────

_SCOPE = {"topic": "cats", "n": 3, "score": 1.5, "doc": {"title": "T"}, "obj": [10, 20]}

_ALREADY_WORKING = [
    "no placeholders at all",
    "one {topic}",
    "{topic} and {n} and {topic} again",
    "{score:.2f}",
    "{n:03d}",
    "{topic!r}",
    "{doc[title]}",
    "{obj[0]}",
    "escaped {{braces}} beside {topic}",
    "{{}}",
    "trailing text after {topic}.",
    "",
]


@pytest.mark.parametrize("text", _ALREADY_WORKING)
def test_identical_to_str_format_when_everything_resolves(text):
    assert render_template(text, _SCOPE) == (text.format(**_SCOPE), [])


# ── the failure path ────────────────────────────────────────────────────────────

def test_a_miss_costs_only_itself():
    rendered, unresolved = render_template("write {topic} in the style of {styel}", _SCOPE)
    assert rendered == "write cats in the style of {styel}"
    assert unresolved == ["styel"]


def test_several_misses_are_all_reported():
    rendered, unresolved = render_template("{a} {topic} {b}", _SCOPE)
    assert rendered == "{a} cats {b}"
    assert unresolved == ["a", "b"]


def test_literal_json_passes_through_untouched():
    rendered, unresolved = render_template('summarise {topic} as {"t": "x"}', _SCOPE)
    assert rendered == 'summarise cats as {"t": "x"}'
    assert unresolved == ['"t"']


def test_an_unresolved_placeholder_keeps_its_spec_and_conversion():
    """It is reproduced from source, not normalised — the text a human wrote is the
    text they see in the prompt when it fails."""
    rendered, _ = render_template("{missing:.2f} {gone!r} {absent}", _SCOPE)
    assert rendered == "{missing:.2f} {gone!r} {absent}"


def test_positional_placeholders_are_left_alone():
    rendered, unresolved = render_template("an empty object is {} or {0}", _SCOPE)
    assert rendered == "an empty object is {} or {0}"
    assert unresolved == ["", "0"]


def test_prose_after_a_colon_survives():
    rendered, _ = render_template("emit {status: ok} for {topic}", _SCOPE)
    assert rendered == "emit {status: ok} for cats"


def test_unbalanced_braces_return_the_text_untouched():
    """There is nothing to parse, so there is no partial rendering to attempt."""
    text = "summarise {topic"
    assert render_template(text, _SCOPE) == (text, [])


def test_a_none_value_still_renders():
    """A skipped branch's output is None, not missing — that is a value, not a miss."""
    assert render_template("got {x}", {"x": None}) == ("got None", [])


def test_attribute_access_on_a_missing_root_is_left_whole():
    rendered, unresolved = render_template("{nope.title}", _SCOPE)
    assert rendered == "{nope.title}"
    assert unresolved == ["nope.title"]


def test_a_failing_lookup_inside_a_present_object_is_left_whole():
    rendered, unresolved = render_template("{doc[missing]}", _SCOPE)
    assert rendered == "{doc[missing]}"
    assert unresolved == ["doc[missing]"]


def test_nested_format_spec_resolves():
    assert render_template("{score:{width}}", {"score": 1.5, "width": "8.2f"})[0] \
        == f"{1.5:8.2f}"


def test_nested_format_spec_that_cannot_resolve_leaves_the_whole_placeholder():
    rendered, unresolved = render_template("{score:{width}}", {"score": 1.5})
    assert rendered == "{score:{width}}"
    assert unresolved == ["score"]


# ── through the engine ──────────────────────────────────────────────────────────

class _RecordingProvider(ScriptedProvider):
    def __init__(self, turns):
        super().__init__(turns)
        self.systems: list[str] = []

    async def stream(self, messages, system, tools, config):
        self.systems.append(system)
        async for ev in super().stream(messages, system, tools, config):
            yield ev


def test_a_node_keeps_its_good_values_when_one_name_is_wrong(tmp_path):
    """The point of the change, end to end: a typo in the third placeholder no longer
    costs the node the first two."""
    nodes = [GraphNode(
        id="a", kind="base",
        goal="brief on {topic} for {audience} in the style of {styel}",
    )]
    graph = Graph(
        name="t",
        inputs=[{"name": "topic", "type": "string", "required": True},
                {"name": "audience", "type": "string", "required": True}],
        nodes=nodes,
        outputs=["a"],
    )
    provider = _RecordingProvider([("out", "")] * 2)
    GraphExecutor(graph=graph, provider=provider).run({"topic": "cats", "audience": "vets"})

    assert any("brief on cats for vets in the style of {styel}" in s
               for s in provider.systems)


def test_a_var_from_a_branch_that_did_not_run_costs_only_itself(tmp_path):
    """The case validation deliberately leaves as a warning: `skipped_note` is never
    written because its writer is pruned, and `topic` must survive that."""
    nodes = [
        GraphNode(id="gate", kind="base", when="inputs.topic == 'never'",
                  writes="skipped_note", goal="go"),
        GraphNode(id="main", kind="base", goal="brief on {topic}, note: {skipped_note}"),
    ]
    graph = Graph(
        name="t",
        inputs=[{"name": "topic", "type": "string", "required": True}],
        nodes=nodes,
        outputs=["main"],
    )
    provider = _RecordingProvider([("out", "")] * 3)
    GraphExecutor(graph=graph, provider=provider).run({"topic": "cats"})

    assert any("brief on cats, note: {skipped_note}" in s for s in provider.systems)


def test_router_keeps_its_graph_inputs_when_a_node_id_is_referenced(tmp_path):
    """A `routes` router sees inputs only. Before, naming a node discarded the whole
    instruction — including the steer the user actually supplied."""
    nodes = [
        GraphNode(id="research", kind="base", goal="go"),
        GraphNode(id="r", kind="router", depends_on=["research"], routes={"only": "x"},
                  purpose="steer: {depth_hint}. evidence: {research}"),
        GraphNode(id="x", kind="base", depends_on=["r"], goal="go"),
    ]
    graph = Graph(
        name="t",
        inputs=[{"name": "depth_hint", "type": "string", "required": True}],
        nodes=nodes,
        outputs=["x"],
    )

    users: list[str] = []

    class _Router(ScriptedProvider):
        async def stream(self, messages, system, tools, config):
            users.append("\n".join(str(getattr(m, "content", m)) for m in messages))
            async for ev in super().stream(messages, system, tools, config):
                yield ev

    GraphExecutor(graph=graph, provider=_Router([("go", ""), ("only", ""), ("done", "")])) \
        .run({"depth_hint": "go deep"})

    assert any("steer: go deep. evidence: {research}" in u for u in users)

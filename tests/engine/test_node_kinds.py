"""The node-kind specs describe the engine, and are checked against it.

A spec that drifts from `GraphNode` is worse than no spec: it renders an editor
for a value nothing reads, or hides a field that matters. So nothing here trusts
the specs — every claim is verified against the model, the executor's kind set,
and the loader's own rules.
"""

from __future__ import annotations

import pytest

from neurosurfer.graph.engine.kinds import (
    NODE_KIND_SPECS,
    all_kind_specs,
    kind_specs_as_dicts,
    node_kind_spec,
)
from neurosurfer.graph.engine.schema import _TERMINAL_NODE_KINDS, _VALID_NODE_KINDS, GraphNode


def test_every_engine_kind_has_a_spec():
    assert set(NODE_KIND_SPECS) == set(_VALID_NODE_KINDS)


def test_no_spec_invents_a_kind():
    for spec in all_kind_specs():
        assert spec.kind in _VALID_NODE_KINDS


@pytest.mark.parametrize("spec", all_kind_specs(), ids=lambda s: s.kind)
def test_every_field_exists_on_graphnode(spec):
    """The one that catches a rename.

    `item_var` is aliased as `as` in YAML, so the check is against the model's
    field names rather than what a workflow file writes.
    """
    model_fields = set(GraphNode.model_fields)
    for f in spec.fields:
        assert f.name in model_fields, (
            f"{spec.kind}.{f.name} is not a field of GraphNode"
        )


@pytest.mark.parametrize("spec", all_kind_specs(), ids=lambda s: s.kind)
def test_fields_are_not_declared_twice(spec):
    names = [f.name for f in spec.fields]
    assert len(names) == len(set(names)), f"{spec.kind} declares a field twice"


@pytest.mark.parametrize("spec", all_kind_specs(), ids=lambda s: s.kind)
def test_every_field_is_described(spec):
    """A field with no help text is a field the next person guesses at."""
    for f in spec.fields:
        assert f.label.strip(), f"{spec.kind}.{f.name} has no label"
        assert f.help.strip(), f"{spec.kind}.{f.name} has no help text"


@pytest.mark.parametrize("spec", all_kind_specs(), ids=lambda s: s.kind)
def test_select_fields_offer_options(spec):
    for f in spec.fields:
        if f.type == "select" and f.name != "provider":
            assert f.options, f"{spec.kind}.{f.name} is a select with no options"


def test_terminal_kinds_agree_with_the_schema():
    terminal = {s.kind for s in all_kind_specs() if s.terminal}
    assert terminal == set(_TERMINAL_NODE_KINDS)


def test_a_terminal_kind_offers_no_error_route():
    """An output node cannot be an `on_error` target, so it must not offer one."""
    for spec in all_kind_specs():
        if spec.terminal:
            assert spec.field("on_error") is None, (
                f"{spec.kind} is terminal but offers on_error"
            )
            assert spec.field("writes") is None, (
                f"{spec.kind} runs nothing but offers writes"
            )


def test_container_kinds_declare_a_body():
    """`has_body` and an actual `body` field must agree — the loader requires a
    non-empty body for exactly these kinds."""
    for spec in all_kind_specs():
        has_field = spec.field("body") is not None
        assert has_field == spec.has_body, (
            f"{spec.kind}: has_body={spec.has_body} but body field={has_field}"
        )
        if spec.has_body:
            assert spec.field("body").required


def test_kinds_the_loader_requires_a_body_from_are_containers():
    assert {s.kind for s in all_kind_specs() if s.has_body} == {"loop", "map", "subgraph"}


def test_model_choice_is_only_offered_where_a_model_runs():
    for spec in all_kind_specs():
        offers_provider = spec.field("provider") is not None
        assert offers_provider == spec.calls_model, (
            f"{spec.kind}: calls_model={spec.calls_model} but "
            f"provider offered={offers_provider}"
        )


def test_the_tool_round_budget_is_declared_by_the_kinds_that_use_tools():
    """`base` allows one round; `react` loops. That difference *is* the two kinds.

    It was a literal inside `run_base_node`, so the single most consequential
    property of a `base` node was the one thing no consumer of the specs could
    read. Asserted here against the value the executor now passes.
    """
    assert node_kind_spec("base").tool_rounds == 1
    assert node_kind_spec("react").tool_rounds is None


def test_the_executor_uses_the_budget_the_spec_declares():
    """The point of moving it: one number, not two that can drift apart."""
    import inspect

    from neurosurfer.graph.engine import node_runner

    source = inspect.getsource(node_runner.run_base_node)
    assert "NODE_KIND_SPECS[\"base\"].tool_rounds" in source, (
        "run_base_node must read the budget from the spec, not restate it"
    )


def test_kinds_that_never_call_tools_declare_no_budget():
    """`None` on a kind with no tool loop means "not applicable", and that is the
    same value `react` uses for "unbounded" — so only assert it where it is
    meaningful, which is the kinds that can hold tools at all."""
    for spec in all_kind_specs():
        if spec.field("tools") is None:
            assert spec.tool_rounds is None, (
                f"{spec.kind} declares a tool-round budget but offers no tools"
            )


def test_required_fields_match_what_the_loader_enforces():
    """Requiredness is a claim about the engine, so check it against the engine.

    These four are refused at load time (`loader.py`), which is what makes it
    honest to mark them required rather than merely important.
    """
    assert node_kind_spec("loop").field("max_iterations").required
    assert node_kind_spec("map").field("over").required
    assert node_kind_spec("react").field("tools").required
    assert node_kind_spec("function").field("callable").required


def test_python_is_specced_as_what_it_is_not_what_it_is_called():
    """`python` is an alias of `function` today. The spec says so, out loud —
    the studio's own blurb has claimed inline execution for as long as the kind
    has existed, and the executor has never had such a path."""
    py, fn = node_kind_spec("python"), node_kind_spec("function")
    assert py.field("callable").required
    assert [f.name for f in py.fields] == [f.name for f in fn.fields]
    assert any("identical to `function`" in c for c in py.constraints)


#: `GraphNode` fields no kind offers, each for a stated reason. The point of the
#: list is that "not offered" has to be a decision somebody wrote down — an
#: unlisted field falling out of every spec is the exact oversight these specs
#: exist to stop, and it is invisible without this test.
UNCLAIMED = {
    # Identity and presentation: the shell renders these, not the kind editors.
    "id": "structural",
    "kind": "structural",
    "name": "shell renders it above the fields",
    "description": "shell",
    "disabled": "a toolbar toggle, not a config field",
    # Superseded by `instructions`, still read so graphs on disk keep running.
    "purpose": "legacy, folded into instructions",
    "goal": "legacy, folded into instructions",
    "expected_result": "legacy, folded into instructions",
    # Nothing reads it. See _common.py.
    "rag": "the engine never consults it — offering it would promise retrieval "
           "that does not happen",
    # Retired from the `input` node's surface (2026-08-01). A fixed list of
    # strings was the only structure that kind could express and it is the wrong
    # one: what a workflow takes is *named, typed inputs*, which are declared as
    # graph inputs and edited on the node in `dict` mode. The field stays on the
    # model — the engine still passes it to an interactive CLI ask — and is no
    # longer something the studio asks an author to fill in.
    "options": "superseded by declared graph inputs in the input node's dict mode",
    # Retired from the agent surface (2026-08-02). All four are still read by the
    # engine — this is a narrowing of what the studio *asks for*, not a change to
    # what a graph on disk may set. See the notes in `_common.py` for each.
    "model": "the provider profile names the model; a second, free-text override "
             "beside it could disagree with the endpoint serving it",
    "policy": "honoured, but its only editor was a raw JSON box — it comes back "
              "as named controls for temperature/tokens/timeout/retries",
    "export": "writing a file is a side effect of the run, not a property of a step",
    "export_path": "goes with `export`",
    # Retired together (2026-08-02) because they only make sense together: `when`
    # reads a run variable that `writes` creates, and the canvas shows neither.
    # Both are still honoured by the engine. See the note in `_common.py`.
    "when": "an expression over run variables the studio never displays; comes "
            "back with a surface that makes them visible",
    "writes": "names a run variable for `when` to read, and nothing else offered "
              "it — retired with `when`",
}


def test_unclaimed_fields_are_a_written_decision():
    covered = {f.name for s in all_kind_specs() for f in s.fields}
    unclaimed = set(GraphNode.model_fields) - covered
    assert unclaimed == set(UNCLAIMED), (
        f"unclaimed but unexplained: {sorted(unclaimed - set(UNCLAIMED))}; "
        f"explained but now claimed: {sorted(set(UNCLAIMED) - unclaimed)}"
    )


def test_every_container_offers_body_outputs():
    """`_child_executor` builds the body graph with `outputs=body_outputs` for
    all three container kinds, so all three must offer it.

    `loop` did not, which is the mirror of the `export` defect below: one spec
    promised a field the engine ignores, the other hid a field it reads. Both
    are "the spec disagrees with the executor", and both are cheap to pin.
    """
    for spec in all_kind_specs():
        if spec.has_body:
            assert spec.field("body_outputs"), f"{spec.kind} hides body_outputs"


def test_export_is_offered_only_where_the_engine_honours_it():
    """`export` is read in exactly one place — `_run_node_native`, the LLM call
    path — so a `tool` or `function` node that sets it exports nothing.

    The first draft of these specs gave every producing kind an export field,
    which is precisely the kind of claim the spec exists to stop: a control that
    writes a value nothing reads. Offering it is a promise, and this is the test
    that the promise is kept.
    """
    offers = {s.kind for s in all_kind_specs() if s.field("export")}
    calls_model = {s.kind for s in all_kind_specs() if s.calls_model}
    assert offers <= calls_model, (
        f"{sorted(offers - calls_model)} offer export but make no model call, "
        f"and the exporter is only consulted on the model path"
    )
    assert "function" not in offers and "tool" not in offers


def test_serialises_to_plain_data():
    payload = kind_specs_as_dicts()
    assert len(payload) == len(_VALID_NODE_KINDS)
    one = next(p for p in payload if p["kind"] == "tool")
    assert one["label"] == "Tool"
    assert one["calls_model"] is False
    assert any(f["name"] == "tool_args" for f in one["fields"])
    # JSON-able all the way down: no dataclasses, no tuples left behind.
    import json

    json.dumps(payload)

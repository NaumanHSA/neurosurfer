"""The rules, one at a time — and an audit of the rule table itself.

The audit tests at the bottom are the point of the registry: *"what can go wrong
with an input node"* used to be answerable only by reading eleven hundred lines,
which is how two rules came to fire on a graph that was correct by design.
"""

from __future__ import annotations

from pathlib import Path

from neurosurfer.graph.engine.schema import Graph, GraphInput, GraphNode
from neurosurfer.graph.workflow.package import WorkflowPackage
from neurosurfer.graph.workflow.schema import WorkflowManifest
from neurosurfer.graph.workflow.validation import validate_package
from neurosurfer.graph.workflow.validation.registry import (
    graph_rules,
    node_rules,
    rules_for_kind,
)


def pkg(nodes, outputs=(), inputs=(), tmp_path: Path | None = None) -> WorkflowPackage:
    graph = Graph(name="t", nodes=list(nodes), outputs=list(outputs), inputs=list(inputs))
    return WorkflowPackage(
        manifest=WorkflowManifest(name="t"), graph=graph, path=tmp_path or Path(".")
    )


def kinds_of(report) -> set[str]:
    return {i.kind for i in report.issues}


# ── the two that used to fire on a correct graph ────────────────────────────


def test_an_input_node_is_not_asked_to_declare_itself(tmp_path):
    """Input nodes *are* the declaration; asking for a second one was noise.

    This is the exact graph the panel complained about: a human-input step whose
    key is not repeated under `graph.inputs`.
    """
    report = validate_package(pkg([
        GraphNode(id="input", kind="input"),
        GraphNode(id="agent", kind="base", instructions="Answer.", depends_on=["input"]),
        GraphNode(id="result", kind="output", depends_on=["agent"]),
    ], outputs=["result"], tmp_path=tmp_path))
    assert not any("declare" in i.message for i in report.issues), report.summary()


def test_an_output_node_is_not_an_orphan(tmp_path):
    """Nothing consuming an output node is what an output node is for."""
    report = validate_package(pkg([
        GraphNode(id="agent", kind="base", instructions="Answer."),
        GraphNode(id="result", kind="output", depends_on=["agent"]),
    ], tmp_path=tmp_path))
    orphans = [i for i in report.issues if i.kind == "structure" and i.node_id == "result"]
    assert not orphans, report.summary()


# ── agent ───────────────────────────────────────────────────────────────────


def test_an_agent_with_no_instructions_warns(tmp_path):
    report = validate_package(pkg([
        GraphNode(id="a", kind="base"),
    ], tmp_path=tmp_path))
    assert "agent.no_instructions" in kinds_of(report)
    assert report.ok, "an empty prompt is a warning, not a blocker"


def test_an_agent_with_instructions_is_quiet(tmp_path):
    report = validate_package(pkg([
        GraphNode(id="a", kind="base", instructions="Summarise the ticket."),
    ], tmp_path=tmp_path))
    assert "agent.no_instructions" not in kinds_of(report)


def test_structured_output_without_a_shape_is_an_error(tmp_path):
    report = validate_package(pkg([
        GraphNode(id="a", kind="base", instructions="Answer.", mode="structured"),
    ], tmp_path=tmp_path))
    assert "agent.structured_without_schema" in kinds_of(report)
    assert not report.ok


def test_structured_output_with_a_shape_passes(tmp_path):
    report = validate_package(pkg([
        GraphNode(id="a", kind="base", instructions="Answer.", mode="structured",
                  output_schema={"type": "object", "properties": {"t": {"type": "string"}}}),
    ], tmp_path=tmp_path))
    assert "agent.structured_without_schema" not in kinds_of(report)


def test_text_mode_never_asks_for_a_shape(tmp_path):
    report = validate_package(pkg([
        GraphNode(id="a", kind="base", instructions="Answer.", mode="text"),
    ], tmp_path=tmp_path))
    assert "agent.structured_without_schema" not in kinds_of(report)


class TestWhatAnAgentRunsOn:
    """Nothing to run on is an error; something to fall back to is a nudge."""

    NODE = [GraphNode(id="a", kind="base", instructions="Answer.")]

    def test_not_asking_says_nothing(self, tmp_path):
        # `None` means "the caller does not know", which most callers do not.
        report = validate_package(pkg(self.NODE, tmp_path=tmp_path), known_providers=None)
        assert not {"agent.no_model", "agent.no_model_available"} & kinds_of(report)

    def test_a_configured_profile_is_a_nudge(self, tmp_path):
        report = validate_package(pkg(self.NODE, tmp_path=tmp_path),
                                  known_providers={"cheap"})
        assert "agent.no_model" in kinds_of(report)
        assert report.ok, "it still runs on the default"

    def test_an_env_fallback_is_a_nudge(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        report = validate_package(pkg(self.NODE, tmp_path=tmp_path), known_providers=set())
        assert "agent.no_model" in kinds_of(report)
        assert report.ok

    def test_nothing_at_all_is_an_error(self, tmp_path, monkeypatch):
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(k, raising=False)
        report = validate_package(pkg(self.NODE, tmp_path=tmp_path), known_providers=set())
        assert "agent.no_model_available" in kinds_of(report)
        assert not report.ok

    def test_naming_a_model_settles_it(self, tmp_path):
        node = [GraphNode(id="a", kind="base", instructions="Answer.", provider="cheap")]
        report = validate_package(pkg(node, tmp_path=tmp_path), known_providers={"cheap"})
        assert not {"agent.no_model", "agent.no_model_available"} & kinds_of(report)


# ── wiring ──────────────────────────────────────────────────────────────────


def test_a_step_wired_to_nothing_warns(tmp_path):
    report = validate_package(pkg([
        GraphNode(id="a", kind="base", instructions="One."),
        GraphNode(id="b", kind="base", instructions="Two.", depends_on=["a"]),
        GraphNode(id="lonely", kind="base", instructions="Nobody calls me."),
    ], outputs=["b"], tmp_path=tmp_path))
    isolated = [i for i in report.issues if i.kind == "structure.isolated"]
    assert [i.node_id for i in isolated] == ["lonely"]


def test_a_workflow_that_returns_nothing_warns(tmp_path):
    report = validate_package(pkg([
        GraphNode(id="a", kind="base", instructions="Answer."),
    ], tmp_path=tmp_path))
    assert "workflow.no_result" in kinds_of(report)
    assert report.ok, "it runs — it just hands nothing back"


def test_an_output_node_counts_as_a_way_out(tmp_path):
    report = validate_package(pkg([
        GraphNode(id="a", kind="base", instructions="Answer."),
        GraphNode(id="r", kind="output", depends_on=["a"]),
    ], tmp_path=tmp_path))
    assert "workflow.no_result" not in kinds_of(report)


def test_an_output_with_nothing_feeding_it_warns(tmp_path):
    report = validate_package(pkg([
        GraphNode(id="a", kind="base", instructions="Answer."),
        GraphNode(id="r", kind="output"),
    ], tmp_path=tmp_path))
    assert "output.no_source" in kinds_of(report)


# ── input ───────────────────────────────────────────────────────────────────


def test_one_declared_field_suggests_free_text(tmp_path):
    report = validate_package(pkg(
        [GraphNode(id="i", kind="input", input_mode="dict")],
        inputs=[GraphInput(name="city")],
        tmp_path=tmp_path,
    ))
    assert "input.single_field" in kinds_of(report)
    assert report.ok, "a suggestion never blocks"
    assert report.infos, "and it is an info, not a warning"


def test_several_declared_fields_are_left_alone(tmp_path):
    report = validate_package(pkg(
        [GraphNode(id="i", kind="input", input_mode="dict")],
        inputs=[GraphInput(name="city"), GraphInput(name="days")],
        tmp_path=tmp_path,
    ))
    assert "input.single_field" not in kinds_of(report)


# ── messages ────────────────────────────────────────────────────────────────


#: Words that belong in `detail`, never in a sentence shown to an author.
TECHNICAL = (
    "depends_on", "output_schema", "graph.inputs", "`outputs`", "writes",
    "input_mode", "tool_args", "node_id", "provider=", "mode=",
)


def every_issue(tmp_path) -> list:
    """Provoke every new rule at least once and collect what they said."""
    from neurosurfer.graph.engine.schema import GraphInput

    graphs = [
        # empty prompt, no model, structured with no shape
        ([GraphNode(id="a", kind="base", mode="structured")], (), ()),
        # isolated step, and a workflow that returns nothing
        ([GraphNode(id="a", kind="base", instructions="One."),
          GraphNode(id="b", kind="base", instructions="Two.", depends_on=["a"]),
          GraphNode(id="lonely", kind="base", instructions="Alone.")], ("b",), ()),
        # an output with nothing feeding it
        ([GraphNode(id="a", kind="base", instructions="Answer."),
          GraphNode(id="r", kind="output")], (), ()),
        # a single declared field
        ([GraphNode(id="i", kind="input", input_mode="dict")], (), (GraphInput(name="city"),)),
        # a filesystem tool with no folder set, and a required argument nothing
        # supplies — the two states a step reaches by being dragged onto a canvas
        # and not configured, which is why they are provoked together.
        ([GraphNode(id="w", kind="tool", tools=["write_file"])], ("w",), ()),
        # a folder that is set and is not there, a setting the tool does not
        # have, and configuration left behind for a detached tool
        ([GraphNode(
            id="w2", kind="tool", tools=["write_file"],
            tool_args={"path": "a.md", "content": "x"},
            tool_settings={
                "write_file": {"root": "/no/such/folder/anywhere", "colour": "blue"},
                "read_file": {"root": "/tmp"},
            },
        )], ("w2",), ()),
    ]
    issues = []
    for nodes, outputs, inputs in graphs:
        report = validate_package(
            pkg(nodes, outputs=outputs, inputs=inputs, tmp_path=tmp_path),
            known_providers={"cheap"},
        )
        issues.extend(report.issues)
    return issues


def test_the_new_rules_all_fire(tmp_path):
    """Guards the test below: it proves nothing if nothing was provoked."""
    fired = {i.kind for i in every_issue(tmp_path)}
    assert {
        "agent.no_instructions", "agent.no_model", "agent.structured_without_schema",
        "structure.isolated", "workflow.no_result", "output.no_source",
        "input.single_field",
        "tool.setting_missing", "tool.setting_unknown_folder",
        "tool.unknown_setting", "tool.setting_for_absent_tool",
        # The pre-existing rule for "a tool node supplies nothing". It was in the
        # swept list and no test graph had ever provoked it, so its two technical
        # suggestions went unnoticed until the settings graphs above reached it.
        "tool_args",
    } <= fired, f"only got {sorted(fired)}"


#: The rules rewritten to the plain-language contract. Older messages have not
#: been swept yet — see `.dev/` — and scoping this list is what keeps the
#: contract enforced for new rules instead of asserted for all and skipped.
PLAIN = {
    "agent.no_instructions", "agent.no_model", "agent.no_model_available",
    "agent.structured_without_schema", "output.no_source", "input.single_field",
    "structure.isolated", "workflow.no_result", "structure",
    # Swept 2026-08-03: the messages that fire for the three focus kinds.
    "output_empty", "tool_args", "dag", "provider", "schema",
    "binding.source_has_no_shape", "binding.unknown_field",
    "binding.unused_argument", "agent.shape_disables_tools",
    # Tool settings (2026-08-03). Written to the contract from the start —
    # these are the messages a person meets when a step has no folder set, and
    # "root is required" is precisely the sentence they must never see.
    "tool.setting_missing", "tool.setting_unknown_folder",
    "tool.unknown_setting", "tool.setting_for_absent_tool",
}


def test_no_message_names_a_field(tmp_path):
    """The panel shows `message`; field names live in `detail`.

    This is the "no technical stuff in the UI" contract, checked against what
    the rules actually produce rather than against their source.
    """
    for issue in (i for i in every_issue(tmp_path) if i.kind in PLAIN):
        for term in TECHNICAL:
            assert term not in issue.message, (
                f"{issue.kind} shows `{term}` to a person: {issue.message!r}"
            )
            assert term not in (issue.suggestion or ""), (
                f"{issue.kind}'s suggestion shows `{term}`: {issue.suggestion!r}"
            )


def test_a_message_reads_as_a_sentence(tmp_path):
    """Capital letter, full stop, no lowercase identifier openings."""
    for issue in (i for i in every_issue(tmp_path) if i.kind in PLAIN):
        m = issue.message
        assert m[:1].isupper(), f"{issue.kind}: {m!r} does not start as a sentence"
        assert m.rstrip().endswith("."), f"{issue.kind}: {m!r} has no full stop"


# ── a base node's one round of tools ────────────────────────────────────────


def test_two_tools_on_a_one_round_step_is_warned(tmp_path):
    """`base` declares `tool_rounds=1`, so it cannot chain one tool into the next.

    A warning, not an error: two independent lookups answered in a single
    parallel round is a working step.
    """
    report = validate_package(pkg(
        [GraphNode(id="a", kind="base", instructions="Look both up.",
                   tools=["read_file", "web_search"])],
        outputs=["a"], tmp_path=tmp_path,
    ))

    assert "agent.tools_exceed_rounds" in kinds_of(report)
    assert report.ok, "a warning must not fail the workflow"


def test_one_tool_on_a_one_round_step_is_fine(tmp_path):
    report = validate_package(pkg(
        [GraphNode(id="a", kind="base", instructions="Read it.", tools=["read_file"])],
        outputs=["a"], tmp_path=tmp_path,
    ))

    assert "agent.tools_exceed_rounds" not in kinds_of(report)


def test_a_react_node_is_never_warned_about_rounds(tmp_path):
    """`react` declares `tool_rounds=None` — it loops until the guardrails stop it."""
    report = validate_package(pkg(
        [GraphNode(id="a", kind="react", instructions="Work it out.",
                   tools=["read_file", "web_search", "write_file"])],
        outputs=["a"], tmp_path=tmp_path,
    ))

    assert "agent.tools_exceed_rounds" not in kinds_of(report)


def test_a_shaped_answer_is_reported_once_not_twice(tmp_path):
    """`output_schema` disables tools outright; that is the other rule's finding.
    Reporting both would describe one node as two different mistakes."""
    report = validate_package(pkg(
        [GraphNode(id="a", kind="base", instructions="Answer.", mode="structured",
                   output_schema={"type": "object"},
                   tools=["read_file", "web_search"])],
        outputs=["a"], tmp_path=tmp_path,
    ))

    assert "agent.shape_disables_tools" in kinds_of(report)
    assert "agent.tools_exceed_rounds" not in kinds_of(report)


# ── an input nothing reads ──────────────────────────────────────────────────


def test_a_declared_input_no_step_names_is_flagged(tmp_path):
    """The rule that makes the narrowing safe. A node is no longer recited every
    graph input, so an input nothing names is a parameter that does nothing —
    the caller passes it, the run is green, and the answer ignores it."""
    report = validate_package(pkg(
        [GraphNode(id="a", kind="base", instructions="Write a summary.")],
        outputs=["a"], inputs=[GraphInput(name="article", type="string")],
        tmp_path=tmp_path,
    ))

    assert "structure" in kinds_of(report)
    assert any(i.subject == "article" for i in report.warnings)
    assert report.ok, "a warning must not fail the workflow"


def test_an_input_a_step_interpolates_is_not_flagged(tmp_path):
    report = validate_package(pkg(
        [GraphNode(id="a", kind="base", instructions="Summarise {article}.")],
        outputs=["a"], inputs=[GraphInput(name="article", type="string")],
        tmp_path=tmp_path,
    ))

    assert not [i for i in report.warnings if i.subject == "article"]


def test_an_input_a_function_node_names_as_a_parameter_is_not_flagged(tmp_path):
    """A code node is called with the inputs mapping as kwargs, so its signature
    reads them — the same argument that already exempted `tool` nodes.

    This is the capstone tutorial's shape. Judging code nodes by their templates
    reported a graph as ignoring the two inputs its functions consume every run.
    """
    report = validate_package(pkg(
        [GraphNode(id="a", kind="function",
                   callable="tests.engine.input_reading_fns:reads_one_strictly")],
        outputs=["a"], inputs=[GraphInput(name="article", type="string")],
        tmp_path=tmp_path,
    ))

    assert not [i for i in report.warnings if i.subject == "article"]


def test_an_input_no_function_node_names_is_still_flagged(tmp_path):
    """The other half: exempting the *node* rather than its parameters would
    silence the rule for any graph containing a code node at all."""
    report = validate_package(pkg(
        [GraphNode(id="a", kind="function",
                   callable="tests.engine.input_reading_fns:reads_one_strictly")],
        outputs=["a"],
        inputs=[GraphInput(name="article", type="string"),
                GraphInput(name="unused", type="string")],
        tmp_path=tmp_path,
    ))

    assert [i for i in report.warnings if i.subject == "unused"]
    assert not [i for i in report.warnings if i.subject == "article"]


def test_a_function_node_taking_kwargs_reads_everything(tmp_path):
    """`**kwargs` receives whatever it is handed, so nothing is unread and the
    rule has nothing to say — about any input, not only the ones named."""
    report = validate_package(pkg(
        [GraphNode(id="a", kind="function",
                   callable="tests.engine.input_reading_fns:reads_whatever_it_is_given")],
        outputs=["a"],
        inputs=[GraphInput(name="article", type="string"),
                GraphInput(name="anything", type="string")],
        tmp_path=tmp_path,
    ))

    assert not [i for i in report.warnings if i.kind == "structure"]


def test_a_callable_that_does_not_import_is_left_to_its_own_rule(tmp_path):
    """`callable_resolves` reports the import failure with the path and the
    exception. Reporting the same defect again as "this input is unread" would
    send the reader after the wrong thing."""
    report = validate_package(pkg(
        [GraphNode(id="a", kind="function", callable="no.such.module:fn")],
        outputs=["a"], inputs=[GraphInput(name="article", type="string")],
        tmp_path=tmp_path,
    ))

    assert "callable" in kinds_of(report)


def test_an_input_read_only_by_an_expression_is_not_flagged(tmp_path):
    """A `map`'s `over` reads its collection without a placeholder anywhere. A
    rule that only looked at templates would call a working fan-out broken."""
    report = validate_package(pkg([
        GraphNode(
            id="fan", kind="map", over="inputs.reviews", **{"as": "item"},
            body=[GraphNode(id="s", kind="base", instructions="Summarise {item}.")],
            body_outputs=["s"],
        ),
    ], outputs=["fan"], inputs=[GraphInput(name="reviews", type="array")], tmp_path=tmp_path))

    assert not [i for i in report.warnings if i.subject == "reviews"]


def test_an_input_read_only_inside_a_container_body_is_not_flagged(tmp_path):
    """The walk has to descend: the only reader is two levels down."""
    report = validate_package(pkg([
        GraphNode(
            id="fan", kind="map", over="inputs.reviews", **{"as": "item"},
            body=[GraphNode(id="s", kind="base",
                            instructions="In {house_style}, summarise {item}.")],
            body_outputs=["s"],
        ),
    ], outputs=["fan"],
        inputs=[GraphInput(name="reviews", type="array"),
                GraphInput(name="house_style", type="string")],
        tmp_path=tmp_path))

    assert not [i for i in report.warnings if i.subject == "house_style"]


def test_an_input_read_only_by_a_tool_argument_is_not_flagged(tmp_path):
    report = validate_package(pkg([
        GraphNode(id="t", kind="tool", tools=["read_file"],
                  tool_args={"path": "{doc_path}"}),
    ], outputs=["t"], inputs=[GraphInput(name="doc_path", type="string")], tmp_path=tmp_path))

    assert not [i for i in report.warnings if i.subject == "doc_path"]


# ── the router that classifies on nothing ───────────────────────────────────


def _routing_graph(router: GraphNode, tmp_path):
    return pkg(
        [router,
         GraphNode(id="a", kind="base", instructions="A.", depends_on=["r"]),
         GraphNode(id="b", kind="base", instructions="B.", depends_on=["r"])],
        outputs=["a", "b"], inputs=[GraphInput(name="ticket", type="string")],
        tmp_path=tmp_path,
    )


def test_a_routes_router_that_is_shown_nothing_is_flagged(tmp_path):
    """It runs green and takes the wrong branch every time — a `routes` router
    is not handed the graph inputs, so an instruction that never names one is
    classifying a request it cannot see."""
    report = validate_package(_routing_graph(GraphNode(
        id="r", kind="router", routes={"billing": "a", "bug": "b"},
        instructions="Decide whether this is a billing question or a bug.",
    ), tmp_path))

    assert "router.classifies_on_nothing" in kinds_of(report)
    assert report.ok, "a warning must not fail the workflow"


def test_a_routes_router_naming_an_input_is_fine(tmp_path):
    report = validate_package(_routing_graph(GraphNode(
        id="r", kind="router", routes={"billing": "a", "bug": "b"},
        instructions="Classify this ticket: {ticket}",
    ), tmp_path))

    assert "router.classifies_on_nothing" not in kinds_of(report)


def test_a_routes_router_reading_an_upstream_step_is_fine(tmp_path):
    """Dependency outputs *are* appended to the classifier prompt, so a router
    with a parent has its evidence even with no placeholder."""
    report = validate_package(pkg([
        GraphNode(id="fetch", kind="base", instructions="Fetch it."),
        GraphNode(id="r", kind="router", depends_on=["fetch"],
                  routes={"billing": "a", "bug": "b"},
                  instructions="Decide which it is."),
        GraphNode(id="a", kind="base", instructions="A.", depends_on=["r"]),
        GraphNode(id="b", kind="base", instructions="B.", depends_on=["r"]),
    ], outputs=["a", "b"], tmp_path=tmp_path))

    assert "router.classifies_on_nothing" not in kinds_of(report)


def test_a_cases_router_is_not_asked_to_interpolate(tmp_path):
    """An expression router reads state directly and makes no model call."""
    report = validate_package(_routing_graph(GraphNode(
        id="r", kind="router", default="b",
        cases=[{"when": "inputs.ticket != ''", "to": "a"}],
        instructions="Route it.",
    ), tmp_path))

    assert "router.classifies_on_nothing" not in kinds_of(report)


# ── the audit ───────────────────────────────────────────────────────────────


def test_every_rule_declares_kinds_it_can_speak_about():
    """No rule is registered against a kind that has no such field.

    This is the check that would have caught `callable_resolves` being declared
    against an `input` node — harmless in practice, because the check returns
    early, but a declaration that says something untrue.
    """
    from neurosurfer.graph.engine.kinds import node_kind_spec

    field_owner = {
        "callable_resolves": "callable",
        "output_schema_resolves": "output_schema",
        "tools_exist": "tools",
        "provider_is_configured": "provider",
    }
    for rule in node_rules():
        field = field_owner.get(rule.name)
        if not field:
            continue
        for kind in rule.kinds:
            spec = node_kind_spec(kind)
            assert spec and spec.field(field), (
                f"{rule.name} is declared for '{kind}', which has no `{field}`"
            )


def test_an_output_node_is_only_asked_about_things_it_has():
    """The audit the old shape could not answer without reading the whole file."""
    names = {r.name for r in rules_for_kind("output")}
    assert "callable_resolves" not in names
    assert "agent_has_instructions" not in names
    assert "output_has_something_to_return" in names


def test_an_input_node_is_only_asked_about_things_it_has():
    names = {r.name for r in rules_for_kind("input")}
    assert "output_schema_resolves" not in names
    assert "provider_is_configured" not in names
    assert "one_declared_field_could_be_free_text" in names


def test_every_rule_module_is_imported():
    """A rule module nobody imports registers nothing and is silently dead."""
    import pkgutil

    from neurosurfer.graph.workflow.validation import nodes

    on_disk = {m.name for m in pkgutil.iter_modules(nodes.__path__)
               if not m.name.startswith("__")}
    imported = set(nodes.__all__)
    assert on_disk == imported, f"not imported in nodes/__init__: {on_disk - imported}"


def test_the_rule_table_is_not_empty():
    assert len(node_rules()) >= 10
    assert len(graph_rules()) >= 5

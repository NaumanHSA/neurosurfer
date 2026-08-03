"""Tests for the pre-registration validation gate (Phase E1)."""

from __future__ import annotations

from pathlib import Path

from neurosurfer.graph import Graph, GraphNode, NodeMode
from neurosurfer.graph.workflow.package import WorkflowPackage
from neurosurfer.graph.workflow.schema import WorkflowManifest
from neurosurfer.graph.workflow.validate import validate_package

_SCHEMA = "neurosurfer.architect.schemas:WorkflowPlan"


def _pkg(nodes: list[GraphNode], outputs: list[str], tmp_path: Path) -> WorkflowPackage:
    graph = Graph(name="t", nodes=nodes, outputs=outputs)
    manifest = WorkflowManifest(name="t")
    return WorkflowPackage(manifest=manifest, graph=graph, path=tmp_path)


# ── happy path ──────────────────────────────────────────────────────────────────

def test_valid_package_passes(tmp_path):
    nodes = [
        GraphNode(id="a", kind="react", tools=["read_file", "run_command"]),
        GraphNode(id="b", kind="react", tools=["write_file"], depends_on=["a"]),
    ]
    report = validate_package(_pkg(nodes, ["b"], tmp_path))
    assert report.ok
    assert not report.errors and not report.gaps


def test_valid_structured_output_schema(tmp_path):
    nodes = [GraphNode(id="a", kind="base", mode=NodeMode.STRUCTURED, output_schema=_SCHEMA)]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert report.ok


# ── capability gaps ─────────────────────────────────────────────────────────────

def test_invented_tool_is_a_gap(tmp_path):
    nodes = [GraphNode(id="a", kind="react", tools=["extract_docstrings"])]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert not report.ok
    assert len(report.gaps) == 1
    assert report.gaps[0].kind == "tool_gap"
    assert report.gaps[0].node_id == "a"
    assert not report.errors


def test_gap_keeps_package_unregisterable(tmp_path):
    nodes = [GraphNode(id="a", kind="react", tools=["frobnicate_widgets"])]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert not report.ok
    assert "frobnicate_widgets" in report.summary()


# ── typos ───────────────────────────────────────────────────────────────────────

def test_tool_typo_suggests_nearest(tmp_path):
    nodes = [GraphNode(id="a", kind="react", tools=["read_fil"])]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert not report.ok
    assert len(report.errors) == 1
    assert report.errors[0].kind == "tool_typo"
    assert "read_file" in (report.errors[0].suggestion or "")


# ── DAG / edges ──────────────────────────────────────────────────────────────────

def test_unknown_depends_on_is_error(tmp_path):
    nodes = [GraphNode(id="a", kind="react", tools=["read_file"], depends_on=["ghost"])]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert not report.ok
    # The offending name lives in `detail` now: `message` is the sentence the
    # panel shows, and it must not name a field or an id the author never typed.
    assert any(e.kind == "dag" and "ghost" in (e.detail or "") for e in report.errors)


def test_unknown_output_is_error(tmp_path):
    nodes = [GraphNode(id="a", kind="react", tools=["read_file"])]
    report = validate_package(_pkg(nodes, ["nope"], tmp_path))
    assert not report.ok
    assert any(e.kind == "dag" and "nope" in (e.detail or "") for e in report.errors)


# ── schema / callable imports ────────────────────────────────────────────────────

def test_bad_output_schema_is_error(tmp_path):
    nodes = [GraphNode(id="a", kind="base", output_schema="nonexistent.module:Thing")]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert not report.ok
    assert any(e.kind == "schema" for e in report.errors)


def test_output_schema_not_basemodel_is_error(tmp_path):
    # points at a real importable object that is NOT a pydantic model
    nodes = [GraphNode(id="a", kind="base", output_schema="os:getcwd")]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert not report.ok
    assert any(e.kind == "schema" for e in report.errors)


def test_function_node_missing_callable_is_error(tmp_path):
    nodes = [GraphNode(id="a", kind="function")]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert not report.ok
    assert any(e.kind == "callable" for e in report.errors)


def test_function_node_bad_callable_is_error(tmp_path):
    nodes = [GraphNode(id="a", kind="function", callable="nonexistent.module:fn")]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert not report.ok
    assert any(e.kind == "callable" for e in report.errors)


def test_function_node_valid_callable_passes(tmp_path):
    nodes = [GraphNode(id="a", kind="function", callable="os:getcwd")]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert report.ok


# ── report rendering ─────────────────────────────────────────────────────────────

def test_report_summary_groups_errors_and_gaps(tmp_path):
    nodes = [
        GraphNode(id="a", kind="react", tools=["extract_docstrings"], depends_on=["ghost"]),
    ]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    text = report.summary()
    assert "Errors:" in text
    assert "Capability gaps" in text


# ── secrets a node may use, and the line they must not cross ───────────────────

def test_a_declared_secret_fills_a_tool_argument():
    from neurosurfer.graph.engine.schema import GraphNode
    from neurosurfer.graph.engine.secrets import expand_node_secrets
    from neurosurfer.mcp.credentials import use_credentials

    node = GraphNode(id="q", kind="tool", tools=["http"], secrets=["DB_URL"])
    with use_credentials({"DB_URL": "postgres://u:pw@host/db"}):
        assert expand_node_secrets("${DB_URL}", node) == "postgres://u:pw@host/db"


def test_an_undeclared_secret_is_left_as_written():
    """The declaration is the authorisation. Expanding a name the node never
    claimed would make `secrets:` documentation rather than a gate."""
    from neurosurfer.graph.engine.schema import GraphNode
    from neurosurfer.graph.engine.secrets import expand_node_secrets
    from neurosurfer.mcp.credentials import use_credentials

    node = GraphNode(id="q", kind="tool", tools=["http"], secrets=["OTHER"])
    with use_credentials({"DB_URL": "postgres://secret"}):
        assert expand_node_secrets("${DB_URL}", node) == "${DB_URL}"


def test_a_secret_in_a_prompt_field_is_an_error(tmp_path):
    """The constraint the whole design serves: a value in a goal is in the
    model's context, the trace, and anything the trace is exported to."""
    nodes = [GraphNode(id="a", kind="base",
                       goal="Connect using ${DB_PASSWORD} and summarise.")]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert "secret_in_prompt" in {i.kind for i in report.errors}
    assert not report.ok


def test_declaring_a_secret_on_a_node_that_cannot_use_one_warns(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal="Write something.",
                       secrets=["DB_URL"])]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert "secret_unusable" in {i.kind for i in report.warnings}


def test_a_secret_is_masked_in_what_gets_traced():
    """A tool's *output* echoes what it was given — a connection error quoting the
    URL it failed on is the usual way a password reaches a trace."""
    from neurosurfer.graph.engine.secrets import redact
    from neurosurfer.mcp.credentials import use_credentials

    with use_credentials({"DB_URL": "postgres://u:sup3rsecret@host/db"}):
        out = redact({"error": "could not connect to postgres://u:sup3rsecret@host/db"})
    assert "sup3rsecret" not in str(out)
    assert "••••••" in out["error"]


def test_a_short_value_is_not_used_to_redact_everything():
    """A secret of "1" would turn every digit in a trace into a mask, and is not
    protecting anything anyway."""
    from neurosurfer.graph.engine.secrets import redact
    from neurosurfer.mcp.credentials import use_credentials

    with use_credentials({"N": "1"}):
        assert redact("step 1 of 12") == "step 1 of 12"


# ── what a built workflow needs before it can run ──────────────────────────────

def test_requirements_name_the_node_and_the_server_that_want_them(tmp_path):
    """The registry publishes a server's config and it used to be read once during
    resolution and thrown away — so a registered workflow recorded nothing about
    what had to be set, and said so by failing on its first node."""
    from neurosurfer.config.mcp import McpServerConfig
    from neurosurfer.graph.workflow.requirements import (
        missing_requirements,
        workflow_requirements,
    )

    class _Store:
        def list(self):
            return [
                McpServerConfig(name="db", transport="stdio", command="serve",
                                enabled=True, env={"PATH_TO_DB": "${SQLITE_DB_PATH}"}),
                # Disabled servers are not connected, so they ask for nothing.
                McpServerConfig(name="off", transport="stdio", command="x",
                                enabled=False, env={"K": "${NEVER_ASKED}"}),
            ]

    nodes = [GraphNode(id="a", kind="tool", tools=["http"], secrets=["API_TOKEN"],
                       tool_args={"url": "${API_TOKEN}"})]
    pkg = _pkg(nodes, ["a"], tmp_path)

    reqs = workflow_requirements(pkg, store=_Store(), available={"API_TOKEN": "x"})
    by_name = {r.name: r for r in reqs}
    assert set(by_name) == {"API_TOKEN", "SQLITE_DB_PATH"}
    assert by_name["API_TOKEN"].source == "node:a"
    assert by_name["API_TOKEN"].satisfied is True
    assert by_name["SQLITE_DB_PATH"].source == "server:db"
    assert by_name["SQLITE_DB_PATH"].satisfied is False

    missing = missing_requirements(pkg, store=_Store(), available={"API_TOKEN": "x"})
    assert [r.name for r in missing] == ["SQLITE_DB_PATH"]


def test_requirements_reach_into_nested_bodies(tmp_path):
    """A map or loop body is where the real work usually is; a checklist that
    stopped at the top level would miss exactly the nodes that call things."""
    from neurosurfer.graph.workflow.requirements import workflow_requirements

    # The secret has to be *referenced* to be required, not merely declared —
    # `secrets:` authorises, `tool_args` uses. See `_node_secret_refs`.
    nodes = [GraphNode(id="each", kind="map", body=[
        GraphNode(id="fetch", kind="tool", tools=["http"], secrets=["INNER_KEY"],
                  tool_args={"url": "https://api.example/v1",
                             "headers": {"Authorization": "Bearer ${INNER_KEY}"}}),
    ])]

    class _Empty:
        def list(self):
            return []

    reqs = workflow_requirements(_pkg(nodes, ["each"], tmp_path), store=_Empty(),
                                 available={})
    assert [r.name for r in reqs] == ["INNER_KEY"]


# ── a tool node must supply what its tool requires ──────────────────────────────

def test_a_tool_node_without_the_required_arguments_is_refused(tmp_path):
    """The defect this gate exists for, in the shape it actually shipped.

    Seven nodes declared `tools: ['query_sql']`, no `tool_args`, and a careful
    prose goal describing the SQL to write. A tool node makes no model call, so
    the goal was read by nobody and the tool was invoked with nothing.
    """
    nodes = [GraphNode(id="fetch", kind="tool", tools=["read_file"],
                       goal="Read the report the user asked about")]
    report = validate_package(_pkg(nodes, ["fetch"], tmp_path))
    assert not report.ok
    issue = next(e for e in report.errors if e.kind == "tool_args")
    assert "path" in issue.message
    # It must still name the way out — but in the register a person reads. The
    # kind is `react`; what an author is looking at is a step with an agent in
    # it, and the suggestion says that. `detail` keeps the field name.
    assert "agent step" in (issue.suggestion or ""), "it must name the way out"
    assert "tool_args" in (issue.detail or ""), "and the field name must survive"
    assert "tool_args" not in issue.suggestion, "but not in the sentence"


def test_a_required_argument_may_come_from_the_graph_scope(tmp_path):
    """The executor passes inputs and dependency outputs alongside tool_args, so
    a parameter satisfied from there is genuinely satisfied — checking tool_args
    alone would reject working workflows."""
    from neurosurfer.graph.engine.schema import Graph, GraphInput
    from neurosurfer.graph.workflow.package import WorkflowManifest, WorkflowPackage

    graph = Graph(
        name="w",
        inputs=[GraphInput(name="path", type="string")],
        nodes=[GraphNode(id="fetch", kind="tool", tools=["read_file"])],
        outputs=["fetch"],
    )
    pkg = WorkflowPackage(
        manifest=WorkflowManifest(name="w", description="d", version="0.0.1"),
        graph=graph, path=tmp_path)
    assert not [e for e in validate_package(pkg).errors if e.kind == "tool_args"]


def _tool_args_template_pkg(tool_args: dict, tmp_path, *, secrets=()):
    """A three-node graph shaped like the build that earned these tests."""
    from neurosurfer.graph.engine.schema import Graph, GraphInput
    from neurosurfer.graph.workflow.package import WorkflowManifest, WorkflowPackage

    graph = Graph(
        name="w",
        inputs=[GraphInput(name="markdown_file_path", type="string")],
        nodes=[
            GraphNode(id="generate_markdown_table", kind="base"),
            GraphNode(id="write_to_file", kind="tool", tools=["write_file"],
                      depends_on=["generate_markdown_table"],
                      secrets=list(secrets), tool_args=tool_args),
        ],
        outputs=["write_to_file"],
    )
    return WorkflowPackage(
        manifest=WorkflowManifest(name="w", description="d", version="0.0.1"),
        graph=graph, path=tmp_path)


def test_a_prose_placeholder_in_tool_args_is_rejected(tmp_path):
    """A description where a reference belongs.

    A live build wrote `content: "{output of generate_markdown_table}"`. Nothing
    resolved it, the renderer left it as written, `write_file` wrote it, and the
    build, the run and all three nodes reported success over a 35-byte file
    containing exactly that string. `_TEMPLATE_FIELDS` never covered `tool_args`,
    so validation had never looked.
    """
    pkg = _tool_args_template_pkg(
        {"path": "{markdown_file_path}",
         "content": "{output of generate_markdown_table}"}, tmp_path)
    issue = next(e for e in validate_package(pkg).errors
                 if e.kind == "tool_args_template")
    assert "resolves to nothing" in issue.message
    # The prose contains the name it meant, which beats edit distance outright.
    assert "{generate_markdown_table}" in issue.suggestion


def test_a_resolvable_tool_args_reference_is_left_alone(tmp_path):
    pkg = _tool_args_template_pkg(
        {"path": "{markdown_file_path}", "content": "{generate_markdown_table}"},
        tmp_path)
    assert not [e for e in validate_package(pkg).errors
                if e.kind == "tool_args_template"]


def test_a_declared_secret_is_not_reported_as_unresolved(tmp_path):
    """`${NAME}` contains `{NAME}`, so the formatter sees a placeholder.

    Without the `secrets:` exemption this says "your credential did not resolve"
    in precisely the case where everything works — the defect this codebase has
    already fixed once, in the executor's warning.
    """
    pkg = _tool_args_template_pkg(
        {"path": "{markdown_file_path}", "content": "${WRITE_TOKEN}"},
        tmp_path, secrets=["WRITE_TOKEN"])
    assert not [e for e in validate_package(pkg).errors
                if e.kind == "tool_args_template"]


def test_a_nested_tool_args_placeholder_is_still_checked(tmp_path):
    """A model nests `tool_args` as readily as it writes a flat one."""
    pkg = _tool_args_template_pkg(
        {"path": "{markdown_file_path}",
         "content": {"body": ["{no_such_name}"]}}, tmp_path)
    issue = next(e for e in validate_package(pkg).errors
                 if e.kind == "tool_args_template")
    assert "content.body[0]" in issue.message


def test_an_unbound_credential_argument_says_to_use_a_secret(tmp_path):
    """`dsn` carries a password, so the fix is a stored value — not a literal."""
    nodes = [GraphNode(id="q", kind="tool", tools=["sql"])]
    report = validate_package(_pkg(nodes, ["q"], tmp_path))
    creds = [e for e in report.errors
             if e.kind == "tool_args" and e.subject == "dsn"]
    assert creds, "a credential parameter with no value must be called out"
    # The fix is a stored value, and the sentence has to say so without naming
    # the field it lands in — `secrets:` is the mechanism, "a saved credential"
    # is the thing somebody has. Both survive, in their own fields.
    assert "credential" in (creds[-1].suggestion or "")
    assert "secrets:" in (creds[-1].detail or "")


def test_a_properly_bound_tool_node_passes(tmp_path):
    nodes = [GraphNode(id="q", kind="tool", tools=["sql"],
                       secrets=["SALES_DB_URL"],
                       tool_args={"dsn": "${SALES_DB_URL}",
                                  "operation": "query", "query": "SELECT 1"})]
    report = validate_package(_pkg(nodes, ["q"], tmp_path))
    assert not [e for e in report.errors if e.kind == "tool_args"]


def test_only_referenced_secrets_are_required(tmp_path):
    """`secrets:` authorises; `tool_args` uses. A declared-but-unused name is not
    something to ask a user for — a live build invented seven of them."""
    from neurosurfer.graph.workflow.requirements import workflow_requirements

    nodes = [GraphNode(id="q", kind="tool", tools=["sql"],
                       secrets=["SALES_DB_URL", "UNUSED_DRIVER", "UNUSED_ENCRYPTION"],
                       tool_args={"dsn": "${SALES_DB_URL}", "operation": "query", "query": "SELECT 1"})]

    class _Empty:
        def list(self):
            return []

    reqs = workflow_requirements(_pkg(nodes, ["q"], tmp_path), store=_Empty(),
                                 available={})
    assert [r.name for r in reqs] == ["SALES_DB_URL"]


def test_an_undeclared_reference_is_not_requested(tmp_path):
    """The executor leaves an undeclared `${NAME}` unexpanded, so supplying it
    would not help — asking for it would be asking for nothing."""
    from neurosurfer.graph.workflow.requirements import workflow_requirements

    nodes = [GraphNode(id="q", kind="tool", tools=["sql"],
                       tool_args={"dsn": "${NEVER_DECLARED}", "operation": "query", "query": "SELECT 1"})]

    class _Empty:
        def list(self):
            return []

    reqs = workflow_requirements(_pkg(nodes, ["q"], tmp_path), store=_Empty(),
                                 available={})
    assert reqs == []


def test_a_substituted_credential_is_not_kept_in_the_node_record(tmp_path):
    """`node_input` is persisted in the run record and rendered in the studio.

    The trace *span* was redacted from the start; this field was not, and it holds
    `tool_args` **after** `${NAME}` substitution — so a node using a secret stored
    the plaintext credential. Caught by running a real query against a real
    database and grepping the result for the password.
    """
    from neurosurfer.graph.engine.executor import GraphExecutor
    from neurosurfer.graph.engine.schema import Graph
    from neurosurfer.mcp.credentials import use_credentials
    from neurosurfer.tools.base import ToolContext, ToolPool
    from neurosurfer.tools.registry import workflow_node_tools

    node = GraphNode(id="q", kind="tool", tools=["sql"],
                     secrets=["DB_URL"],
                     tool_args={"dsn": "${DB_URL}", "operation": "query", "query": "SELECT 1"})
    ex = GraphExecutor(graph=Graph(name="t", nodes=[node], outputs=["q"]))
    ex.native_tools = ToolPool(workflow_node_tools())
    ex._tool_ctx = ToolContext(cwd=tmp_path, io=None)

    secret = "postgresql+psycopg://u:sup3rsecretvalue@nowhere.invalid/db"
    with use_credentials({"DB_URL": secret}):
        result = ex._run_tool_node(node, {}, {})

    assert "sup3rsecretvalue" not in str(result.node_input)


# ── a react node that needs a credential (the shape that was impossible) ───────

def test_a_react_node_may_declare_secrets():
    """Two correct decisions used to be mutually incompatible.

    Composing SQL needs a model, so the step is `react`; reaching the database
    needs a credential, so it declares `secrets`. The validator answered "react
    has no tool_args to use them in" and a build spent fifteen rounds unable to
    satisfy both — ruling out the commonest integration shape there is.
    """
    from neurosurfer.graph.engine.bound_tools import bind_pool
    from neurosurfer.tools.base import ToolPool
    from neurosurfer.tools.registry import workflow_node_tools

    node = GraphNode(id="audit", kind="react", tools=["sql"],
                     secrets=["DB_URL"], tool_args={"dsn": "${DB_URL}"},
                     goal="Count rows by status.")
    pool = bind_pool(ToolPool([t for t in workflow_node_tools()
                               if t.name == "sql"]), {"dsn": "postgres://x"})
    bound = pool.get("sql")

    # The model is never offered the parameter it must not see.
    props = bound.schema.input_schema["properties"]
    assert "dsn" not in props, "a bound credential must not appear in the schema"
    assert "query" in props, "the model still composes the rest"
    assert "dsn" not in bound.schema.input_schema["required"]

    # And it arrives anyway, at call time.
    args = bound.parse_args({"operation": "query", "query": "SELECT 1"})
    assert args.dsn == "postgres://x"
    assert node.secrets == ["DB_URL"]


def test_a_bound_value_beats_one_the_model_supplied():
    """A passthrough-schema tool accepts anything; the binding must still win."""
    from neurosurfer.graph.engine.bound_tools import bind_pool
    from neurosurfer.tools.base import ToolPool
    from neurosurfer.tools.registry import workflow_node_tools

    pool = bind_pool(ToolPool([t for t in workflow_node_tools()
                               if t.name == "sql"]), {"dsn": "real://dsn"})
    args = pool.get("sql").parse_args({"operation": "query", "query": "SELECT 1", "dsn": "evil://dsn"})
    assert args.dsn == "real://dsn"


def test_binding_leaves_unrelated_tools_alone():
    """A node holding both sql_query and write_file must not hand write_file a
    `dsn` it has never heard of."""
    from neurosurfer.graph.engine.bound_tools import BoundTool, bind_pool
    from neurosurfer.tools.base import ToolPool
    from neurosurfer.tools.registry import workflow_node_tools

    wanted = {"sql", "write_file"}
    pool = bind_pool(
        ToolPool([t for t in workflow_node_tools() if t.name in wanted]),
        {"dsn": "postgres://x"},
    )
    assert isinstance(pool.get("sql"), BoundTool)
    assert not isinstance(pool.get("write_file"), BoundTool)


def test_declaring_a_secret_on_react_without_binding_it_warns(tmp_path):
    """Declared and never bound reaches no tool call — worth saying, since it
    looks like it should work."""
    nodes = [GraphNode(id="q", kind="react", tools=["sql"],
                       secrets=["DB_URL"], goal="Count the rows.")]
    report = validate_package(_pkg(nodes, ["q"], tmp_path))
    assert "secret_unbound" in {w.kind for w in report.warnings}
    assert "secret_unusable" not in {w.kind for w in report.warnings}


def test_a_base_node_still_cannot_use_a_secret(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal="Write something.",
                       secrets=["DB_URL"])]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert "secret_unusable" in {w.kind for w in report.warnings}


# ── the two template languages agree ───────────────────────────────────────────

def test_the_namespaced_form_of_a_variable_resolves(tmp_path):
    """Expressions say `nodes.summarise`, templates say `{summarise}`. A model
    moving between them wrote `{nodes.query_inquiry_activity_result}` fifteen
    times, was told the right name each time, and wrote it again."""
    nodes = [
        GraphNode(id="summarise", kind="base", goal="Summarise it.", writes="summary"),
        GraphNode(id="title", kind="base", depends_on=["summarise"],
                  goal="Title this: {nodes.summarise}"),
    ]
    report = validate_package(_pkg(nodes, ["title"], tmp_path))
    assert not [e for e in report.errors if e.kind == "template_var"], report.summary()


def test_a_namespaced_variable_that_does_not_exist_is_still_an_error(tmp_path):
    nodes = [
        GraphNode(id="summarise", kind="base", goal="Summarise it."),
        GraphNode(id="title", kind="base", depends_on=["summarise"],
                  goal="Title this: {nodes.no_such_node}"),
    ]
    report = validate_package(_pkg(nodes, ["title"], tmp_path))
    assert [e for e in report.errors if e.kind == "template_var"]


# ── a credential must be usable, not merely present ────────────────────────────

def _sql_pkg(tmp_path):
    nodes = [GraphNode(id="q", kind="react", tools=["sql"], secrets=["DB_URL"],
                       tool_args={"dsn": "${DB_URL}"}, goal="Count rows.")]
    return _pkg(nodes, ["q"], tmp_path)


class _NoServers:
    def list(self):
        return []


def test_a_stored_value_that_cannot_work_is_not_satisfied(tmp_path):
    """The failure this exists for, with the exact value that caused it.

    `AIAccessManagment_SQLSERVER_URL` held `127.0.0.1` — a hostname, left behind
    by an earlier build that had invented a seven-field connection model. Secrets
    are global per account, a later build coined the same name, the check asked
    only "is it set", and the user was never prompted. The run failed three nodes
    later on a malformed URL.
    """
    from neurosurfer.graph.workflow.requirements import workflow_requirements
    from neurosurfer.mcp.credentials import use_credentials

    with use_credentials({"DB_URL": "127.0.0.1"}):
        [req] = workflow_requirements(_sql_pkg(tmp_path), store=_NoServers())
    assert not req.satisfied, "a hostname is not a connection URL"
    assert "SQLAlchemy connection URL" in req.problem
    assert req.help, "and it must say how to get a good one"


def test_a_real_connection_url_is_satisfied(tmp_path):
    from neurosurfer.graph.workflow.requirements import workflow_requirements
    from neurosurfer.mcp.credentials import use_credentials

    with use_credentials({"DB_URL": "mssql+pyodbc://sa:p@h:1433/db?driver=x"}):
        [req] = workflow_requirements(_sql_pkg(tmp_path), store=_NoServers())
    assert req.satisfied and not req.problem


def test_an_unset_value_reports_no_problem_only_absence(tmp_path):
    """Missing and wrong are different states and must read differently."""
    from neurosurfer.graph.workflow.requirements import workflow_requirements
    from neurosurfer.mcp.credentials import use_credentials

    with use_credentials({"OTHER": "x"}):
        [req] = workflow_requirements(_sql_pkg(tmp_path), store=_NoServers())
    assert not req.satisfied and not req.problem
    assert req.help, "an unset value still needs telling how to obtain one"


def test_a_tool_with_no_opinion_accepts_any_non_empty_value(tmp_path):
    """The default check must not start rejecting credentials for tools that
    never declared what a good one looks like."""
    from neurosurfer.tools.registry import workflow_node_tools

    write = next(t for t in workflow_node_tools() if t.name == "write_file")
    assert write.check_secret("anything", "some-value") == ""
    assert write.check_secret("anything", "  ") == "is empty"

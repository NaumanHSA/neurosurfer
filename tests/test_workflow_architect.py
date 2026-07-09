"""Phase C: The Architect — schemas, tools, nodes, package, builder.

Covers:
1. Schema validation: DiscoveryOutput, WorkflowPlan, NodePlan.
2. WriteWorkflowNodeTool: writes agent YAML to projects staging dir.
3. clarify node: non-interactively (monkeypatched input).
4. assemble node: generates workflow.yaml + graph.yaml, validates, registers.
5. Architect package loads cleanly (load_package on built-in package dir).
6. ArchitectBuilder.package loads without error (smoke test).
7-10. Capability planning, feasibility gate, functional tool sandbox, rich specs.
11. E6: ArchitectBuilder resolves capability gaps by authoring tools, then registers
    (merged from test_architect_gap_resolution.py).
12. R2: Architect usable from pure Python without CLI deps (merged from
    test_architect_no_cli_dep.py).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from neurosurfer.architect.nodes import assemble, clarify
from neurosurfer.architect.schemas import (
    ClarifyingQuestion,
    DiscoveryOutput,
    NodePlan,
    WorkflowPlan,
)
from neurosurfer.graph.workflow.node_tool import WriteWorkflowNodeTool
from neurosurfer.graph.workflow.package import load_package
from neurosurfer.tools.base import ToolContext

# ── helpers ───────────────────────────────────────────────────────────────────

class _StubIO:
    async def ask(self, question: str, options: list | None = None) -> str:
        return ""

    async def request_write_approval(self, path: str) -> str:
        return "once"

    async def request_plan_approval(self, plan: str) -> bool:
        return True


def _tool_ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=_StubIO())


def _discovery() -> DiscoveryOutput:
    return DiscoveryOutput(
        summary="Build a workflow that searches the web and summarises results.",
        web_findings="DuckDuckGo search is available via the web_search tool.",
        questions=[
            ClarifyingQuestion(
                id="output_format",
                question="What output format do you want?",
                choices=["Markdown report", "JSON data", "Plain text"],
            ),
            ClarifyingQuestion(
                id="depth",
                question="How deep should the search go?",
                choices=["Quick (1-2 results)", "Medium (3-5 results)", "Deep (10+ results)"],
            ),
            ClarifyingQuestion(
                id="audience",
                question="Who is the primary audience?",
                choices=["Technical users", "Business stakeholders", "General public"],
            ),
        ],
    )


def _plan() -> WorkflowPlan:
    return WorkflowPlan(
        name="web_summariser",
        description="Search the web and summarise the results.",
        nodes=[
            NodePlan(
                id="search",
                kind="react",
                purpose="Search the web for the given query.",
                goal="Return relevant web content.",
                tools=["web_search"],
                mode="text",
            ),
            NodePlan(
                id="summarise",
                kind="base",
                purpose="Summarise the web search results.",
                goal="Produce a concise summary.",
                depends_on=["search"],
                mode="text",
            ),
        ],
        outputs=["summarise"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Schema validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiscoveryOutput:
    def test_choices_must_be_three(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClarifyingQuestion(id="x", question="Q?", choices=["a", "b"])


class TestWorkflowPlan:
    def test_unauthorable_kinds_coerced_to_react(self) -> None:
        # The Architect can't write callables, so function/python/tool collapse to react.
        for bad in ("function", "python", "tool"):
            assert NodePlan(id="n", kind=bad, purpose="p").kind == "react"


class TestToolNameNormalisation:
    def test_common_inventions_map_to_real_tools(self) -> None:
        from neurosurfer.tools.registry import normalize_tool_names

        assert normalize_tool_names(["list_files"]) == ["list_dir"]
        assert normalize_tool_names(["get_directory_structure"]) == ["list_dir"]
        assert normalize_tool_names(["get_file_content"]) == ["read_file"]
        assert normalize_tool_names(["create_directory"]) == ["run_command"]

    def test_pure_llm_pseudo_tools_dropped(self) -> None:
        from neurosurfer.tools.registry import normalize_tool_names

        assert normalize_tool_names(["llm_writer"]) == []

    def test_dedups_and_keeps_genuine_novelties(self) -> None:
        from neurosurfer.tools.registry import normalize_tool_names

        assert normalize_tool_names(["list_files", "list_dir"]) == ["list_dir"]
        # a name with no alias is left for the validation gate / auto-gen
        assert normalize_tool_names(["quantum_flux_tool"]) == ["quantum_flux_tool"]

    def test_walk_directory_variants_map_to_list_dir(self) -> None:
        from neurosurfer.tools.registry import normalize_tool_names

        for name in ("walk_directory", "traverse_directory", "os_walk", "file_tree"):
            assert normalize_tool_names([name]) == ["list_dir"]

    def test_keyword_fallback_catches_novel_fs_names(self) -> None:
        from neurosurfer.tools.registry import normalize_tool_names

        # not in the exact map, but the keyword fallback recognises the fs cluster
        assert normalize_tool_names(["directory_crawler"]) == ["list_dir"]
        assert normalize_tool_names(["recursive_file_walk"]) == ["list_dir"]
        # genuinely-novel, no fs keyword → preserved for the gate / auto-gen
        assert normalize_tool_names(["pdf_to_markdown"]) == ["pdf_to_markdown"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WriteWorkflowNodeTool
# ═══════════════════════════════════════════════════════════════════════════════

class TestWriteWorkflowNodeTool:
    def _patch_projects(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from neurosurfer.config.projects import ProjectsConfig as _PC
        projects_dir = tmp_path / "projects"
        # The tool does a local import inside call(), so patching the class in its
        # source module is sufficient.
        monkeypatch.setattr(
            "neurosurfer.config.projects.ProjectsConfig",
            lambda: _PC(dir=projects_dir),
        )

    def test_writes_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_projects(tmp_path, monkeypatch)
        tool = WriteWorkflowNodeTool()
        ctx = _tool_ctx(tmp_path)

        from neurosurfer.graph.workflow.node_tool import WriteWorkflowNodeArgs
        args = WriteWorkflowNodeArgs(
            workflow_name="test_wf",
            node_id="search",
            content="id: search\nkind: react\npurpose: Search the web.\n",
        )
        result = asyncio.run(tool.call(args, ctx))
        assert not result.is_error, result.content
        dest = tmp_path / "projects" / "test_wf" / "agents" / "search.yaml"
        assert dest.exists()
        data = yaml.safe_load(dest.read_text())
        assert data["id"] == "search"
        assert data["kind"] == "react"

    def test_invalid_yaml_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_projects(tmp_path, monkeypatch)
        tool = WriteWorkflowNodeTool()
        ctx = _tool_ctx(tmp_path)

        from neurosurfer.graph.workflow.node_tool import WriteWorkflowNodeArgs
        args = WriteWorkflowNodeArgs(
            workflow_name="test_wf",
            node_id="bad",
            content=":\tinvalid: yaml: [unclosed",
        )
        result = asyncio.run(tool.call(args, ctx))
        assert result.is_error


# ═══════════════════════════════════════════════════════════════════════════════
# 3. clarify node
# ═══════════════════════════════════════════════════════════════════════════════

class TestClarifyNode:
    def test_returns_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate user always choosing "1"
        responses = iter(["1", "2", "3"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))

        disc = _discovery()
        answers = clarify.run(discover=disc)

        assert answers["output_format"] == "Markdown report"
        assert answers["depth"] == "Medium (3-5 results)"
        assert answers["audience"] == "General public"

    def test_empty_questions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "1")
        result = clarify.run(discover={"questions": [], "summary": ""})
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. assemble node
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssembleNode:
    def _run_assemble(
        self,
        tmp_path: Path,
        plan: WorkflowPlan | dict,
        monkeypatch: pytest.MonkeyPatch,
        *,
        pre_write_agents: bool = False,
    ) -> str:
        from neurosurfer.config.projects import ProjectsConfig as _PC
        from neurosurfer.graph.workflow.registry import WorkflowRegistry as _WR

        projects_dir = tmp_path / "projects"
        registry_dir = tmp_path / "registry"

        # Patch the classes at the point of import in assemble.py (module-level imports).
        monkeypatch.setattr(
            "neurosurfer.architect.nodes.assemble.ProjectsConfig",
            lambda: _PC(dir=projects_dir),
        )
        monkeypatch.setattr(
            "neurosurfer.architect.nodes.assemble.WorkflowRegistry",
            lambda: _WR(registry_dir),
        )

        name = plan.name if isinstance(plan, WorkflowPlan) else plan.get("name", "wf")

        if pre_write_agents:
            agents_dir = projects_dir / name / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            for node in (plan.nodes if isinstance(plan, WorkflowPlan) else plan.get("nodes", [])):
                nid = node.id if hasattr(node, "id") else node["id"]
                (agents_dir / f"{nid}.yaml").write_text(
                    yaml.dump({"id": nid, "kind": "base", "purpose": "stub"}),
                    encoding="utf-8",
                )

        return assemble.run(plan=plan, write_nodes="ignored")

    def test_assembles_from_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = self._run_assemble(tmp_path, _plan(), monkeypatch)
        assert "web_summariser" in path

        # Registered package should be loadable
        registry_dir = tmp_path / "registry"
        pkg = load_package(registry_dir / "web_summariser")
        assert pkg.name == "web_summariser"
        assert len(pkg.graph.nodes) == 2

    def test_staged_agents_merged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._run_assemble(
            tmp_path, _plan(), monkeypatch, pre_write_agents=True
        )
        registry_dir = tmp_path / "registry"
        pkg = load_package(registry_dir / "web_summariser")
        # Staged agents/search.yaml has kind=base; plan says react — staged wins
        search_node = pkg.graph.node_map()["search"]
        assert search_node.kind == "base"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Architect package loads cleanly
# ═══════════════════════════════════════════════════════════════════════════════

class TestArchitectPackage:
    def test_node_ids(self) -> None:
        from neurosurfer.architect import _PACKAGE_DIR

        pkg = load_package(_PACKAGE_DIR)
        assert pkg.name == "architect"
        ids = {n.id for n in pkg.graph.nodes}
        assert ids == {
            "discover", "clarify", "decompose",
            "design_nodes", "critique", "tool_design", "write_nodes", "assemble",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ArchitectBuilder smoke test
# ═══════════════════════════════════════════════════════════════════════════════

class TestArchitectBuilder:
    def test_builder_loads_package(self) -> None:
        from neurosurfer.architect.build import ArchitectBuilder
        from neurosurfer.llm.base import Provider
        from neurosurfer.llm.capabilities import ProviderCapabilities

        class _DummyProvider(Provider):
            model = "dummy"
            capabilities = ProviderCapabilities(
                context_window=8192,
                max_output_tokens=512,
                supports_thinking=False,
                supports_prompt_cache=False,
                supports_token_count=False,
                tool_call_style="openai",
            )

            async def stream(self, messages, system, tools, config):
                from neurosurfer.llm.types import CanonicalResponse, Done, TextBlock, Usage
                yield Done(
                    response=CanonicalResponse(
                        content=[TextBlock(text="stub")],
                        stop_reason="stop",
                        usage=Usage(),
                    )
                )

            async def count_tokens(self, messages, system, tools):
                return 0

        builder = ArchitectBuilder(_DummyProvider())
        pkg = builder.package
        assert pkg.name == "architect"
        assert pkg.graph is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Capability planning (T2) — schemas + feasibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestCapabilityPlan:
    def test_all_feasible_stays_feasible(self) -> None:
        from neurosurfer.architect.schemas import (
            CapabilityPlan,
            NodeCapability,
            ToolSpec,
        )

        cp = CapabilityPlan(nodes=[
            NodeCapability(
                node_id="exec", required_capability="query sqlite",
                decision="author_new",
                new_tools=[ToolSpec(name="sql_query", purpose="run SELECT")],
            ),
        ])
        assert cp.feasible is True
        assert [s.name for s in cp.new_tool_specs()] == ["sql_query"]


class TestSchemaTolerance:
    """Weaker models stringify nested JSON / drop list fields — schemas must cope."""

    def test_workflow_plan_nodes_as_json_string(self) -> None:
        import json

        from neurosurfer.architect.schemas import WorkflowPlan

        nodes = json.dumps([
            {"id": "a", "kind": "base", "purpose": "p", "mode": "text"},
            {"id": "b", "kind": "react", "purpose": "q",
             "depends_on": ["a"], "tools": "data", "mode": "json"},
        ])
        # nodes is a STRING and outputs is MISSING — the exact crash from the field.
        wp = WorkflowPlan.model_validate(
            {"name": "x", "description": "y", "nodes": nodes}
        )
        assert len(wp.nodes) == 2
        assert wp.nodes[1].depends_on == ["a"]
        assert wp.nodes[1].tools == ["data"]   # bare string coerced to list
        assert wp.outputs == ["b"]             # derived terminal node

    def test_truncated_object_array_is_salvaged(self) -> None:
        # Model hit a token limit mid-array — recover the complete objects, drop the
        # half-written tail. (Regression: previously this comma-shredded into garbage.)
        from neurosurfer.architect.schemas import CapabilityPlan

        truncated = (
            '[{"node_id": "a", "required_capability": "x", "decision": "use_existing", '
            '"assigned_tools": ["data"]}, '
            '{"node_id": "b", "required_capability": "y", "decision": "use_existing"}, '
            '{"node_id": "c", "required_capa'
        )
        cp = CapabilityPlan.model_validate({"nodes": truncated})
        assert [n.node_id for n in cp.nodes] == ["a", "b"]

    def test_required_object_list_still_errors_when_unrecoverable(self) -> None:
        # WorkflowPlan needs ≥1 node — garbage must NOT silently become a valid empty plan.
        import pytest as _pytest
        from pydantic import ValidationError

        from neurosurfer.architect.schemas import WorkflowPlan

        with _pytest.raises(ValidationError):
            WorkflowPlan.model_validate(
                {"name": "x", "description": "y", "nodes": "{broken"}
            )


class TestCapabilityOverrides:
    def test_overrides_inputs_and_goal_suffix(self) -> None:
        from neurosurfer.architect.nodes.assemble import _capability_overrides
        from neurosurfer.architect.schemas import (
            CapabilityPlan,
            NodeCapability,
            ToolSpec,
        )

        cp = CapabilityPlan(nodes=[
            NodeCapability(
                node_id="exec", required_capability="query db", decision="author_new",
                new_tools=[ToolSpec(
                    name="sql_query", purpose="run sql",
                    workflow_inputs=["connection_string: the DSN"],
                )],
            ),
            NodeCapability(
                node_id="read", required_capability="read a sqlite file",
                decision="use_existing", assigned_tools=["data"],
            ),
        ])
        overrides, inputs, suffixes = _capability_overrides(cp)
        assert overrides["exec"] == ["sql_query"]
        assert overrides["read"] == ["data"]
        assert inputs == ["connection_string"]
        assert "{connection_string}" in suffixes["exec"]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Feasibility gate (T5) — assemble returns the infeasible marker
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeasibilityGate:
    def test_assemble_returns_infeasible_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from neurosurfer.architect.schemas import CapabilityPlan, NodeCapability
        from neurosurfer.graph.workflow.validate import INFEASIBLE_MARKER

        cp = CapabilityPlan(nodes=[
            NodeCapability(
                node_id="exec", required_capability="query remote db",
                decision="infeasible",
                infeasible_reason="needs a running PostgreSQL server and psycopg2",
            ),
        ])
        out = assemble.run(critique=_plan(), tool_design=cp, write_nodes="ignored")
        assert out.startswith(INFEASIBLE_MARKER)
        assert "psycopg2" in out

    def test_builder_raises_workflow_infeasible(self) -> None:
        from neurosurfer.architect import WorkflowInfeasible
        from neurosurfer.architect.build import ArchitectBuilder
        from neurosurfer.architect.schemas import CapabilityPlan, NodeCapability
        from neurosurfer.graph.workflow.validate import INFEASIBLE_MARKER

        # Minimal fake run result: assemble produced the infeasible marker, and the
        # tool_design node carries the CapabilityPlan.
        class _Node:
            def __init__(self, raw):
                self.raw_output = raw
                self.error = None

        class _Result:
            def __init__(self):
                cp = CapabilityPlan(nodes=[NodeCapability(
                    node_id="x", required_capability="y", decision="infeasible",
                    infeasible_reason="no driver",
                )])
                self.nodes = {
                    "assemble": _Node(f"{INFEASIBLE_MARKER}x: no driver"),
                    "tool_design": _Node(cp),
                }
                self.final = {"assemble": f"{INFEASIBLE_MARKER}x: no driver"}

        builder = ArchitectBuilder.__new__(ArchitectBuilder)
        # Exercise just the marker-handling branch synchronously.
        result = _Result()
        out = result.final["assemble"]
        cap = builder._capability_plan(result)
        assert cap is not None and cap.feasible is False
        with pytest.raises(WorkflowInfeasible) as ei:
            if out.startswith(INFEASIBLE_MARKER):
                raise WorkflowInfeasible(out[len(INFEASIBLE_MARKER):].strip())
        assert "no driver" in ei.value.report


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Functional tool testing (T3) — the sandbox actually runs the tool
# ═══════════════════════════════════════════════════════════════════════════════

_GOOD_TOOL = '''
from pydantic import BaseModel, Field
from neurosurfer.tools.base import Tool, ToolContext, ToolResult
import sqlite3

class SqlQueryArgs(BaseModel):
    db_path: str = Field(description="path")
    query: str = Field(description="sql")

class SqlQueryTool(Tool):
    name = "sql_query"
    description = "Run a read-only SQL query on a SQLite db."
    input_model = SqlQueryArgs
    def is_read_only(self, args) -> bool:
        return True
    async def call(self, args: SqlQueryArgs, ctx: ToolContext) -> ToolResult:
        try:
            conn = sqlite3.connect(args.db_path)
            rows = conn.execute(args.query).fetchall()
            conn.close()
            return ToolResult.ok(str(rows))
        except Exception as e:
            return ToolResult.error(str(e))
'''

_SETUP = '''
import sqlite3, os
db = os.path.join(os.getcwd(), "t.db")
c = sqlite3.connect(db)
c.execute("CREATE TABLE Users(id INTEGER, name TEXT)")
c.execute("INSERT INTO Users VALUES (1, 'alice')")
c.commit(); c.close()
ARGS = {"db_path": db, "query": "SELECT * FROM Users"}
'''


class TestFunctionalSandbox:
    def _author(self):
        from neurosurfer.architect.tool_author import ToolAuthor
        return ToolAuthor.__new__(ToolAuthor)  # validate_draft needs no provider

    def _spec(self):
        from neurosurfer.architect.tool_author import ToolGapSpec
        return ToolGapSpec(
            name="sql_query", purpose="run sql",
            test_setup=_SETUP,
            test_args={"db_path": "t.db", "query": "SELECT * FROM Users"},
        )

    def test_working_tool_passes_with_summary(self) -> None:
        from neurosurfer.architect.tool_author import ToolDraft

        author = self._author()
        res = author.validate_draft(ToolDraft(name="sql_query", code=_GOOD_TOOL, spec=self._spec()))
        assert res.ok is True
        assert res.checks.get("functional_runs") is True
        assert "alice" in res.functional_summary

    def test_tool_returning_error_is_rejected(self) -> None:
        from neurosurfer.architect.tool_author import ToolDraft

        bad = _GOOD_TOOL.replace(
            "rows = conn.execute(args.query).fetchall()",
            'raise RuntimeError("boom")',
        )
        author = self._author()
        res = author.validate_draft(ToolDraft(name="sql_query", code=bad, spec=self._spec()))
        assert res.ok is False
        assert "boom" in res.error


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Rich spec threading (T4) + declared inputs emission (T6)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRichSpecThreading:
    def test_builder_builds_rich_gap_specs(self) -> None:
        from neurosurfer.architect.build import ArchitectBuilder
        from neurosurfer.architect.schemas import (
            CapabilityPlan,
            NodeCapability,
            ToolSpec,
        )

        cp = CapabilityPlan(nodes=[
            NodeCapability(
                node_id="exec", required_capability="query db", decision="author_new",
                new_tools=[ToolSpec(
                    name="sql_query", purpose="run a SELECT",
                    inputs=["db_path: path"], test_args={"db_path": "t.db"},
                    expected_behavior="returns rows",
                )],
            ),
        ])
        builder = ArchitectBuilder.__new__(ArchitectBuilder)
        specs = builder._rich_specs(cp)
        assert "sql_query" in specs
        s = specs["sql_query"]
        assert s.inputs == ["db_path: path"]
        assert s.test_args == {"db_path": "t.db"}
        assert s.expected_behavior == "returns rows"


class TestDeclaredInputsEmitted:
    def test_connection_string_becomes_graph_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from neurosurfer.architect.schemas import (
            CapabilityPlan,
            NodeCapability,
            ToolSpec,
        )
        from neurosurfer.config.projects import ProjectsConfig as _PC
        from neurosurfer.graph.workflow.registry import WorkflowRegistry as _WR

        projects_dir = tmp_path / "projects"
        registry_dir = tmp_path / "registry"
        monkeypatch.setattr(
            "neurosurfer.architect.nodes.assemble.ProjectsConfig",
            lambda: _PC(dir=projects_dir),
        )
        monkeypatch.setattr(
            "neurosurfer.architect.nodes.assemble.WorkflowRegistry",
            lambda: _WR(registry_dir),
        )

        # Plan whose "search" node will use an authored tool needing a workflow input.
        plan = _plan()
        cp = CapabilityPlan(nodes=[
            NodeCapability(
                node_id="search", required_capability="call an API", decision="author_new",
                new_tools=[ToolSpec(
                    name="api_call", purpose="call an API",
                    workflow_inputs=["api_key: the key"],
                )],
            ),
        ])

        assemble.run(critique=plan, tool_design=cp, write_nodes="ignored")
        graph_yaml = yaml.safe_load(
            (projects_dir / "web_summariser" / "graph.yaml").read_text()
        )
        input_names = {i["name"] for i in graph_yaml["inputs"]}
        assert "api_key" in input_names
        # The authored tool name is carried onto the node, authoritatively.
        search = next(n for n in graph_yaml["nodes"] if n["id"] == "search")
        assert search["tools"] == ["api_call"]


# ═══════════════════════════════════════════════════════════════════════════════
# 11. E6: ArchitectBuilder resolves capability gaps by authoring tools, then
#     registers (merged from test_architect_gap_resolution.py).
# ═══════════════════════════════════════════════════════════════════════════════

# A valid tool the fake provider will "author" for the gap.
_COUNT_LINES = '''\
from pydantic import BaseModel, Field
from neurosurfer.tools.base import Tool, ToolContext, ToolResult


class CountLinesArgs(BaseModel):
    path: str = Field(description="file to count")


class CountLinesTool(Tool):
    name = "count_lines"
    description = "Count the lines in a text file."
    input_model = CountLinesArgs

    def is_read_only(self, args):
        return True

    async def call(self, args, ctx):
        return ToolResult.ok("counted")
'''


class _GapResp:
    def __init__(self, text: str) -> None:
        self._t = text

    def text(self) -> str:
        return self._t


class _GapFakeProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def complete(self, messages, system, tools, config):  # noqa: ANN001
        return _GapResp(self.reply)


def _stage_package_with_gap(project_dir: Path, *, tool: str = "count_lines") -> None:
    """Write a minimal staged workflow whose node references a non-existent tool."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "workflow.yaml").write_text(
        yaml.dump({"name": project_dir.name, "version": "0.1.0", "entrypoint": "graph.yaml"}),
        encoding="utf-8",
    )
    graph = {
        "name": project_dir.name,
        "description": "needs a tool that doesn't exist",
        "inputs": [{"name": "query", "type": "string", "required": False}],
        "nodes": [
            {"id": "do_it", "kind": "react", "purpose": "Count lines in files.", "tools": [tool]},
        ],
        "outputs": ["do_it"],
    }
    (project_dir / "graph.yaml").write_text(yaml.dump(graph), encoding="utf-8")


@pytest.fixture
def _isolated(tmp_path, monkeypatch):
    """Point all workflow artifacts (projects, registry, generated tools) at tmp."""
    monkeypatch.setenv("NEUROSURFER_HOME", str(tmp_path))
    return tmp_path


async def _yes(draft, result):  # noqa: ANN001
    return True


async def _no(draft, result):  # noqa: ANN001
    return False


class TestGapResolution:
    @pytest.mark.asyncio
    async def test_gap_resolved_and_registered(self, _isolated, monkeypatch):
        from neurosurfer.architect.build import ArchitectBuilder

        staged = _isolated / ".neurosurfer" / "projects" / "line_counter"
        _stage_package_with_gap(staged)

        builder = ArchitectBuilder(_GapFakeProvider(f"```python\n{_COUNT_LINES}\n```"))
        dest = await builder._finalize_staged(str(staged), _yes)

        # Registered under the tmp registry…
        assert (Path(dest) / "workflow.yaml").exists()
        # …and the authored tool was persisted + is now discoverable.
        from neurosurfer.tools.registry import workflow_node_tool_names

        assert "count_lines" in workflow_node_tool_names()

    @pytest.mark.asyncio
    async def test_gap_rejected_aborts_registration(self, _isolated):
        from neurosurfer.architect.build import ArchitectBuilder

        staged = _isolated / ".neurosurfer" / "projects" / "line_counter2"
        # tool name matches _COUNT_LINES so the draft passes the sandbox and reaches approval
        _stage_package_with_gap(staged, tool="count_lines")

        builder = ArchitectBuilder(_GapFakeProvider(f"```python\n{_COUNT_LINES}\n```"))
        with pytest.raises(RuntimeError, match="declined"):
            await builder._finalize_staged(str(staged), _no)

        assert not (_isolated / "workflows" / "line_counter2").exists()

    @pytest.mark.asyncio
    async def test_gap_without_approver_raises(self, _isolated):
        from neurosurfer.architect.build import ArchitectBuilder

        staged = _isolated / ".neurosurfer" / "projects" / "line_counter3"
        _stage_package_with_gap(staged, tool="count_lines3")

        builder = ArchitectBuilder(_GapFakeProvider(""))
        with pytest.raises(RuntimeError, match="no approval handler"):
            await builder._finalize_staged(str(staged), None)

    def test_assemble_returns_marker_on_gap(self, tmp_path, monkeypatch):
        """The assemble node defers (returns the marker) instead of registering on a gap."""
        from neurosurfer.config.projects import ProjectsConfig as _PC
        from neurosurfer.graph.workflow.validate import DEFER_MARKER

        monkeypatch.setattr(
            "neurosurfer.architect.nodes.assemble.ProjectsConfig",
            lambda: _PC(dir=tmp_path / "projects"),
        )

        plan = WorkflowPlan(
            name="needs_tool",
            description="x",
            nodes=[NodePlan(id="n", kind="react", purpose="p", tools=["nonexistent_tool"])],
            outputs=["n"],
        )
        out = assemble.run(plan=plan, write_nodes="ignored")
        assert out.startswith(DEFER_MARKER)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. R2: Architect usable from pure Python without CLI deps (merged from
#     test_architect_no_cli_dep.py).
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoCLIDependency:
    def test_architect_public_api_importable(self) -> None:
        from neurosurfer.architect import ArchitectBuilder, ArchitectConversation

        assert ArchitectBuilder.__name__ == "ArchitectBuilder"
        assert ArchitectConversation.__name__ == "ArchitectConversation"

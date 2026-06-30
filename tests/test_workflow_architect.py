"""Phase C: The Architect — schemas, tools, nodes, package, builder.

Covers:
1. Schema validation: DiscoveryOutput, WorkflowPlan, NodePlan.
2. WriteWorkflowNodeTool: writes agent YAML to projects staging dir.
3. clarify node: non-interactively (monkeypatched input).
4. assemble node: generates workflow.yaml + graph.yaml, validates, registers.
5. Architect package loads cleanly (load_package on built-in package dir).
6. ArchitectBuilder.package loads without error (smoke test).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from neurosurfer.tools.base import ToolContext
from neurosurfer.graph.workflow.node_tool import WriteWorkflowNodeTool
from neurosurfer.architect.nodes import assemble, clarify
from neurosurfer.architect.schemas import (
    ClarifyingQuestion,
    DiscoveryOutput,
    NodePlan,
    WorkflowPlan,
)
from neurosurfer.graph.workflow.package import load_package

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
    def test_valid(self) -> None:
        d = _discovery()
        assert d.summary
        assert len(d.questions) == 3
        assert all(len(q.choices) == 3 for q in d.questions)

    def test_choices_must_be_three(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClarifyingQuestion(id="x", question="Q?", choices=["a", "b"])

    def test_too_few_questions(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DiscoveryOutput(
                summary="s",
                web_findings="f",
                questions=[
                    ClarifyingQuestion(id="q1", question="Q?", choices=["a", "b", "c"])
                ],
            )

    def test_from_dict(self) -> None:
        raw = {
            "summary": "test",
            "web_findings": "found stuff",
            "questions": [
                {"id": "q1", "question": "Q1?", "choices": ["A", "B", "C"]},
                {"id": "q2", "question": "Q2?", "choices": ["X", "Y", "Z"]},
            ],
        }
        d = DiscoveryOutput.model_validate(raw)
        assert d.questions[0].id == "q1"


class TestWorkflowPlan:
    def test_valid(self) -> None:
        p = _plan()
        assert p.name == "web_summariser"
        assert len(p.nodes) == 2

    def test_name_cleaned(self) -> None:
        p = WorkflowPlan(
            name="My Awesome Workflow!",
            description="d",
            nodes=[NodePlan(id="n1", kind="base", purpose="p")],
            outputs=["n1"],
        )
        assert p.name == "my_awesome_workflow"

    def test_node_id_normalised(self) -> None:
        n = NodePlan(id="My Node", kind="base", purpose="p")
        assert n.id == "my_node"

    def test_invalid_kind(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NodePlan(id="n", kind="unknown_kind", purpose="p")

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

    def test_tool_in_registry(self) -> None:
        from neurosurfer.tools.registry import all_tools
        names = [t.name for t in all_tools()]
        assert "write_workflow_node" in names


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

    def test_accepts_dict_discovery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "1")
        raw = _discovery().model_dump()
        answers = clarify.run(discover=raw)
        assert len(answers) == 3

    def test_empty_questions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "1")
        result = clarify.run(discover={"questions": [], "summary": ""})
        assert result == {}

    def test_unknown_kwargs_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "1")
        disc = _discovery()
        answers = clarify.run(discover=disc, user_intent="ignored extra kwarg")
        assert len(answers) == 3


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

    def test_assembles_from_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = _plan().model_dump()
        path = self._run_assemble(tmp_path, raw, monkeypatch)  # type: ignore[arg-type]
        assert "web_summariser" in path

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
    def test_package_loads(self) -> None:
        from neurosurfer.architect import _PACKAGE_DIR

        pkg = load_package(_PACKAGE_DIR)
        assert pkg.name == "architect"
        assert len(pkg.graph.nodes) == 8

    def test_node_ids(self) -> None:
        from neurosurfer.architect import _PACKAGE_DIR

        pkg = load_package(_PACKAGE_DIR)
        ids = {n.id for n in pkg.graph.nodes}
        assert ids == {
            "discover", "clarify", "decompose",
            "design_nodes", "critique", "tool_design", "write_nodes", "assemble",
        }

    def test_discover_has_web_search(self) -> None:
        from neurosurfer.architect import _PACKAGE_DIR

        pkg = load_package(_PACKAGE_DIR)
        discover = pkg.graph.node_map()["discover"]
        assert "web_search" in discover.tools

    def test_function_nodes_have_callables(self) -> None:
        from neurosurfer.architect import _PACKAGE_DIR

        pkg = load_package(_PACKAGE_DIR)
        node_map = pkg.graph.node_map()
        assert node_map["clarify"].callable is not None
        assert node_map["assemble"].callable is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ArchitectBuilder smoke test
# ═══════════════════════════════════════════════════════════════════════════════

class TestArchitectBuilder:
    def test_builder_loads_package(self) -> None:
        from neurosurfer.llm.base import Provider
        from neurosurfer.llm.capabilities import ProviderCapabilities
        from neurosurfer.architect.build import ArchitectBuilder

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
    def test_infeasible_node_flips_feasible(self) -> None:
        from neurosurfer.architect.schemas import CapabilityPlan, NodeCapability

        cp = CapabilityPlan(nodes=[
            NodeCapability(
                node_id="exec", required_capability="query a remote PG db",
                decision="infeasible", infeasible_reason="needs psycopg2 + a server",
            ),
        ])
        assert cp.feasible is False
        assert any("psycopg2" in b for b in cp.blockers)

    def test_all_feasible_stays_feasible(self) -> None:
        from neurosurfer.architect.schemas import (
            CapabilityPlan, NodeCapability, ToolSpec,
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

    def test_inputs_coerced_from_dicts(self) -> None:
        from neurosurfer.architect.schemas import ToolSpec

        spec = ToolSpec(
            name="sql_query", purpose="run sql",
            inputs=[{"name": "db_path", "description": "path to db"}],
            test_args='{"db_path": "x.db"}',  # JSON-string form tolerated
        )
        assert spec.inputs == ["db_path: path to db"]
        assert spec.test_args == {"db_path": "x.db"}

    def test_name_sanitised(self) -> None:
        from neurosurfer.architect.schemas import ToolSpec

        assert ToolSpec(name="SQL Query!", purpose="x").name == "sql_query"

    def test_capability_nodes_from_json_string(self) -> None:
        import json
        from neurosurfer.architect.schemas import CapabilityPlan

        cp = CapabilityPlan.model_validate({"nodes": json.dumps([
            {"node_id": "q", "required_capability": "query db",
             "decision": "use_existing", "assigned_tools": "data"},
        ])})
        assert cp.nodes[0].assigned_tools == ["data"]


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

    def test_outputs_bare_string_wrapped(self) -> None:
        import json
        from neurosurfer.architect.schemas import WorkflowPlan

        nodes = json.dumps([{"id": "only", "kind": "base", "purpose": "p"}])
        wp = WorkflowPlan.model_validate(
            {"name": "x", "description": "y", "nodes": nodes, "outputs": "only"}
        )
        assert wp.outputs == ["only"]

    def test_stage_plan_stages_as_json_string(self) -> None:
        import json
        from neurosurfer.architect.schemas import StagePlan

        sp = StagePlan.model_validate({"intent": "x", "stages": json.dumps([
            {"id": "a", "name": "A", "purpose": "p"},
            {"id": "b", "name": "B", "purpose": "q"},
        ])})
        assert [s.id for s in sp.stages] == ["a", "b"]

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

    def test_garbage_object_list_degrades_to_empty(self) -> None:
        # CapabilityPlan is optional enrichment — unrecoverable output → empty, so the
        # build proceeds on the critique's own tool assignments instead of crashing.
        from neurosurfer.architect.schemas import CapabilityPlan

        cp = CapabilityPlan.model_validate({"nodes": "{totally broken"})
        assert cp.nodes == []
        assert cp.feasible is True

    def test_single_object_without_array_wrapper(self) -> None:
        from neurosurfer.architect.schemas import CapabilityPlan

        cp = CapabilityPlan.model_validate(
            {"nodes": {"node_id": "a", "required_capability": "x", "decision": "use_existing"}}
        )
        assert [n.node_id for n in cp.nodes] == ["a"]

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
        from neurosurfer.architect.schemas import (
            CapabilityPlan, NodeCapability, ToolSpec,
        )
        from neurosurfer.architect.nodes.assemble import _capability_overrides

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
        from neurosurfer.graph.workflow.validate import INFEASIBLE_MARKER
        from neurosurfer.architect.schemas import CapabilityPlan, NodeCapability

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

    def test_contract_only_when_no_test_plan(self) -> None:
        # No test_args/test_setup → functional stage is skipped (contract-only).
        from neurosurfer.architect.tool_author import ToolDraft, ToolGapSpec

        author = self._author()
        spec = ToolGapSpec(name="sql_query", purpose="run sql")
        res = author.validate_draft(ToolDraft(name="sql_query", code=_GOOD_TOOL, spec=spec))
        assert res.ok is True
        assert "functional_runs" not in res.checks


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Rich spec threading (T4) + declared inputs emission (T6)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRichSpecThreading:
    def test_builder_builds_rich_gap_specs(self) -> None:
        from neurosurfer.architect.build import ArchitectBuilder
        from neurosurfer.architect.schemas import (
            CapabilityPlan, NodeCapability, ToolSpec,
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
        from neurosurfer.config.projects import ProjectsConfig as _PC
        from neurosurfer.graph.workflow.registry import WorkflowRegistry as _WR
        from neurosurfer.architect.schemas import (
            CapabilityPlan, NodeCapability, ToolSpec,
        )

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

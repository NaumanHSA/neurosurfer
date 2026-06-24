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
from neurosurfer.graph.builder.nodes import assemble, clarify
from neurosurfer.graph.builder.schemas import (
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
            "neurosurfer.graph.builder.nodes.assemble.ProjectsConfig",
            lambda: _PC(dir=projects_dir),
        )
        monkeypatch.setattr(
            "neurosurfer.graph.builder.nodes.assemble.WorkflowRegistry",
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
        from neurosurfer.graph.builder import _PACKAGE_DIR

        pkg = load_package(_PACKAGE_DIR)
        assert pkg.name == "architect"
        assert len(pkg.graph.nodes) == 5

    def test_node_ids(self) -> None:
        from neurosurfer.graph.builder import _PACKAGE_DIR

        pkg = load_package(_PACKAGE_DIR)
        ids = {n.id for n in pkg.graph.nodes}
        assert ids == {"discover", "clarify", "plan", "write_nodes", "assemble"}

    def test_discover_has_web_search(self) -> None:
        from neurosurfer.graph.builder import _PACKAGE_DIR

        pkg = load_package(_PACKAGE_DIR)
        discover = pkg.graph.node_map()["discover"]
        assert "web_search" in discover.tools

    def test_function_nodes_have_callables(self) -> None:
        from neurosurfer.graph.builder import _PACKAGE_DIR

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
        from neurosurfer.graph.builder.build import ArchitectBuilder

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

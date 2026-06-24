"""E6: ArchitectBuilder resolves capability gaps by authoring tools, then registers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurosurfer.graph.builder.build import ArchitectBuilder
from neurosurfer.graph.workflow.validate import DEFER_MARKER

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


class _Resp:
    def __init__(self, text: str) -> None:
        self._t = text

    def text(self) -> str:
        return self._t


class _FakeProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def complete(self, messages, system, tools, config):  # noqa: ANN001
        return _Resp(self.reply)


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


@pytest.mark.asyncio
async def test_gap_resolved_and_registered(_isolated, monkeypatch):
    staged = _isolated / ".neurosurfer" / "projects" / "line_counter"
    _stage_package_with_gap(staged)

    builder = ArchitectBuilder(_FakeProvider(f"```python\n{_COUNT_LINES}\n```"))
    dest = await builder._finalize_staged(str(staged), _yes)

    # Registered under the tmp registry…
    assert (Path(dest) / "workflow.yaml").exists()
    # …and the authored tool was persisted + is now discoverable.
    from neurosurfer.tools.registry import workflow_node_tool_names

    assert "count_lines" in workflow_node_tool_names()


@pytest.mark.asyncio
async def test_gap_rejected_aborts_registration(_isolated):
    staged = _isolated / ".neurosurfer" / "projects" / "line_counter2"
    # tool name matches _COUNT_LINES so the draft passes the sandbox and reaches approval
    _stage_package_with_gap(staged, tool="count_lines")

    builder = ArchitectBuilder(_FakeProvider(f"```python\n{_COUNT_LINES}\n```"))
    with pytest.raises(RuntimeError, match="declined"):
        await builder._finalize_staged(str(staged), _no)

    assert not (_isolated / "workflows" / "line_counter2").exists()


@pytest.mark.asyncio
async def test_gap_without_approver_raises(_isolated):
    staged = _isolated / ".neurosurfer" / "projects" / "line_counter3"
    _stage_package_with_gap(staged, tool="count_lines3")

    builder = ArchitectBuilder(_FakeProvider(""))
    with pytest.raises(RuntimeError, match="no approval handler"):
        await builder._finalize_staged(str(staged), None)


def test_assemble_returns_marker_on_gap(tmp_path, monkeypatch):
    """The assemble node defers (returns the marker) instead of registering on a gap."""
    from neurosurfer.config.projects import ProjectsConfig as _PC
    from neurosurfer.graph.builder.nodes import assemble
    from neurosurfer.graph.builder.schemas import NodePlan, WorkflowPlan

    monkeypatch.setattr(
        "neurosurfer.graph.builder.nodes.assemble.ProjectsConfig",
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

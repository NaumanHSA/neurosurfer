"""Tests for the dynamic generated-tools registry (Phase E3)."""

from __future__ import annotations

from pathlib import Path

from neurosurfer.tools.generated import (
    GeneratedToolMeta,
    GeneratedToolsConfig,
    load_generated_tools,
    save_generated_tool,
)

_GREET_CODE = '''
from pydantic import BaseModel, Field
from neurosurfer.tools.base import Tool, ToolContext, ToolResult


class GreetArgs(BaseModel):
    who: str = Field(description="who to greet")


class GreetTool(Tool):
    name = "greet"
    description = "Greet someone by name."
    input_model = GreetArgs

    def is_read_only(self, args):
        return True

    async def call(self, args, ctx):
        return ToolResult.ok(f"Hello {args.who}")
'''

_BROKEN_CODE = "this is not valid python ::: !!!"


def _save(tmp_path: Path, code: str, name: str, **meta) -> GeneratedToolsConfig:
    cfg = GeneratedToolsConfig(dir=tmp_path)
    save_generated_tool(code, GeneratedToolMeta(name=name, **meta), cfg)
    return cfg


# ── loading ──────────────────────────────────────────────────────────────────────

def test_save_and_load_roundtrip(tmp_path):
    cfg = _save(tmp_path, _GREET_CODE, "greet")
    tools = load_generated_tools(cfg)
    assert len(tools) == 1
    assert tools[0].name == "greet"
    assert tools[0].description == "Greet someone by name."


def test_broken_tool_file_is_skipped(tmp_path):
    cfg = GeneratedToolsConfig(dir=tmp_path)
    save_generated_tool(_GREET_CODE, GeneratedToolMeta(name="greet"), cfg)
    (tmp_path / "broken.py").write_text(_BROKEN_CODE, encoding="utf-8")
    tools = load_generated_tools(cfg)
    # The good tool still loads; the broken one is skipped without raising.
    assert [t.name for t in tools] == ["greet"]


# ── registry integration ──────────────────────────────────────────────────────

def test_all_tools_includes_generated(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROSURFER_HOME", str(tmp_path))
    cfg = GeneratedToolsConfig()  # → tmp_path/tools
    save_generated_tool(_GREET_CODE, GeneratedToolMeta(name="greet"), cfg)

    from neurosurfer.tools.registry import (
        all_tools,
        format_workflow_tool_catalog,
        workflow_node_tool_names,
    )

    names = {t.name for t in all_tools()}
    assert "greet" in names
    assert "greet" in workflow_node_tool_names()
    assert "greet" in format_workflow_tool_catalog()


def test_generated_tool_passes_validation_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROSURFER_HOME", str(tmp_path))
    cfg = GeneratedToolsConfig()
    save_generated_tool(_GREET_CODE, GeneratedToolMeta(name="greet"), cfg)

    from neurosurfer.graph import Graph, GraphNode
    from neurosurfer.graph.workflow.package import WorkflowPackage
    from neurosurfer.graph.workflow.schema import WorkflowManifest
    from neurosurfer.graph.workflow.validate import validate_package

    nodes = [GraphNode(id="a", kind="react", tools=["greet"])]
    pkg = WorkflowPackage(
        manifest=WorkflowManifest(name="t"),
        graph=Graph(name="t", nodes=nodes, outputs=["a"]),
        path=tmp_path,
    )
    report = validate_package(pkg)
    assert report.ok, report.summary()

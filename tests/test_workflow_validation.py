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
    assert any(e.kind == "dag" and "ghost" in e.message for e in report.errors)


def test_unknown_output_is_error(tmp_path):
    nodes = [GraphNode(id="a", kind="react", tools=["read_file"])]
    report = validate_package(_pkg(nodes, ["nope"], tmp_path))
    assert not report.ok
    assert any(e.kind == "dag" and "nope" in e.message for e in report.errors)


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

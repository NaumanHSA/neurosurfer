"""Tracing: WorkflowRunner writes a structured JSON trace when trace_path is set."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from neurosurfer.graph.workflow.package import load_package
from neurosurfer.graph.workflow.runner import WorkflowRunner


class _DummyProvider:
    """No LLM nodes are exercised here, so the provider is never called."""

    model = "dummy"
    capabilities = type("Caps", (), {"context_window": 8192})()


def _function_only_pkg(pkg_dir: Path) -> None:
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "workflow.yaml").write_text(
        yaml.dump({"name": pkg_dir.name, "version": "0.1.0", "entrypoint": "graph.yaml"}),
        encoding="utf-8",
    )
    graph = {
        "name": pkg_dir.name,
        "nodes": [{"id": "n", "kind": "function", "callable": "os:getcwd"}],
        "outputs": ["n"],
    }
    (pkg_dir / "graph.yaml").write_text(yaml.dump(graph), encoding="utf-8")


def test_trace_written_and_valid(tmp_path):
    pkg_dir = tmp_path / "wf"
    _function_only_pkg(pkg_dir)
    pkg = load_package(pkg_dir)

    trace_path = tmp_path / "traces" / "run.json"
    runner = WorkflowRunner(_DummyProvider())
    runner.run(pkg, {}, trace_path=trace_path)

    assert trace_path.exists()
    data = json.loads(trace_path.read_text())
    assert "meta" in data and "steps" in data
    assert isinstance(data["steps"], list)
    assert data["meta"].get("workflow") == "wf"


def test_no_trace_when_path_omitted(tmp_path):
    pkg_dir = tmp_path / "wf2"
    _function_only_pkg(pkg_dir)
    pkg = load_package(pkg_dir)

    runner = WorkflowRunner(_DummyProvider())
    runner.run(pkg, {})  # no trace_path → nothing written
    assert not (tmp_path / "traces").exists()

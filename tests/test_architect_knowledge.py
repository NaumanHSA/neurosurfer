"""Phase 3 — auto-derived self-knowledge: manifest, docs index, freshness gates.

The freshness tests are the drift tripwire the plan requires: if someone adds a
node kind, tool, or expression function to the engine without the knowledge layer
picking it up (or without writing kind guidance), these tests fail in CI.
"""

from __future__ import annotations

import pytest

from neurosurfer.architect.knowledge import (
    DocsIndex,
    KnowledgeBase,
    build_manifest,
    manifest_version,
)
from neurosurfer.architect.knowledge.manifest import _KIND_GUIDANCE


@pytest.fixture(scope="module")
def manifest():
    return build_manifest()


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


# ── freshness: manifest cannot drift from the engine ───────────────────────────

def test_node_kinds_cover_engine_exactly(manifest):
    from neurosurfer.graph.engine.schema import _VALID_NODE_KINDS

    assert set(manifest["node_kinds"]) == set(_VALID_NODE_KINDS)
    # Hand-written guidance must cover every kind — no placeholder leaks.
    assert set(_KIND_GUIDANCE) == set(_VALID_NODE_KINDS)
    for kind, info in manifest["node_kinds"].items():
        assert "UNDOCUMENTED" not in info["summary"], f"kind '{kind}' lacks guidance"
        assert info["summary"]


def test_node_fields_cover_graphnode_exactly(manifest):
    from neurosurfer.graph.engine.schema import GraphNode

    assert set(manifest["node_fields"]) == set(GraphNode.model_fields)
    # The `as` alias for item_var must be advertised (agents author YAML).
    assert manifest["node_fields"]["item_var"]["alias"] == "as"


def test_expression_functions_match_evaluator(manifest):
    from neurosurfer.graph.engine.expressions import _ALLOWED_FUNCS

    assert manifest["expressions"]["functions"] == sorted(_ALLOWED_FUNCS)


def test_tools_cover_registry_exactly(manifest):
    from neurosurfer.tools.registry import all_tools, workflow_node_tool_names

    manifest_names = {t["name"] for t in manifest["tools"]}
    assert manifest_names == {t.name for t in all_tools()}
    wf_marked = {t["name"] for t in manifest["tools"] if t["workflow_usable"]}
    assert wf_marked == workflow_node_tool_names() & manifest_names


def test_graph_fields_cover_graph_model(manifest):
    from neurosurfer.graph.engine.schema import Graph

    assert set(manifest["workflow_package"]["graph_fields"]) == set(Graph.model_fields)


def test_execution_api_endpoints_derived(manifest):
    api = manifest["execution_api"]
    if not api.get("available"):
        pytest.skip(f"gateway not importable: {api.get('note')}")
    joined = " ".join(api["endpoints"])
    for fragment in (
        "GET /v1/workflows", "POST /v1/workflows/{name}/runs",
        "GET /v1/runs/{run_id}/events", "POST /v1/runs/{run_id}/resume",
        "DELETE /v1/runs/{run_id}",
    ):
        assert fragment in joined, f"missing endpoint: {fragment}"


# ── versioning ──────────────────────────────────────────────────────────────────

def test_manifest_version_stable_and_content_addressed(manifest):
    v = manifest["manifest_version"]
    assert len(v) == 12 and int(v, 16) is not None  # short hex hash
    # Same content → same version (generated_at must not affect it).
    again = build_manifest()
    assert again["manifest_version"] == v
    # Changed content → different version.
    mutated = dict(manifest)
    mutated["tools"] = manifest["tools"] + [{"name": "fake", "description": "x",
                                            "inputs": [], "workflow_usable": False}]
    assert manifest_version(mutated) != v


# ── KnowledgeBase facade ───────────────────────────────────────────────────────

def test_describe_node_kind_and_tool(kb):
    router = kb.describe_node_kind("router")
    assert "branch" in router["summary"].lower() or "selects" in router["summary"].lower()
    assert kb.describe_node_kind("nope") is None

    read_file = kb.describe_tool("read_file")
    assert read_file is not None
    assert read_file["description"]
    assert "properties" in (read_file.get("input_schema") or {})
    assert kb.describe_tool("no_such_tool") is None


def test_render_context_is_compact_and_grounded(kb):
    ctx = kb.render_context()
    # Grounded: real kinds, a real tool, the guard idiom, the version.
    assert "**router**" in ctx and "**loop**" in ctx and "**map**" in ctx
    assert "`read_file`" in ctx
    assert "contains(lower(" in ctx
    assert kb.version in ctx
    # Compact: a system-prompt block, not a dump.
    assert len(ctx) < 15_000


def test_refresh_reintrospects(kb):
    v1 = kb.version
    kb.refresh()
    assert kb.version == v1  # nothing changed → same content hash


# ── docs retrieval ──────────────────────────────────────────────────────────────

def test_docs_index_finds_relevant_sections(kb):
    hits = kb.search_docs("workflow package registry", k=5)
    assert hits, "docs search returned nothing"
    assert any("graph-workflows" in h["path"] or "workflow" in h["path"].lower()
               for h in hits)

    mcp_hits = kb.search_docs("connect MCP server tools", k=5)
    assert any("mcp" in h["path"].lower() for h in mcp_hits)


def test_docs_index_empty_dir_is_safe(tmp_path):
    idx = DocsIndex(tmp_path / "nope")
    assert idx.search("anything") == []

"""Phase 3 — auto-derived self-knowledge: manifest, docs index, freshness gates.

The freshness tests are the drift tripwire the plan requires: if someone adds a
node kind, tool, or expression function to the engine without the knowledge layer
picking it up (or without writing kind guidance), these tests fail in CI.
"""

from __future__ import annotations

import json

import pytest

from neurosurfer.architect.knowledge import (
    DocsIndex,
    KnowledgeBase,
    build_manifest,
    manifest_version,
)
from neurosurfer.architect.knowledge.manifest import (
    _BUILD_RULES,
    _KIND_GUIDANCE,
    _RECIPES,
)


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
    #
    # Raised from 15k on 2026-08-03, deliberately and temporarily. The old cap
    # was set when the manifest said less; eleven kinds' worth of honest field
    # lists is simply larger, and the last 60 characters were being bought by
    # compressing true sentences rather than by removing anything redundant.
    # Shrinking what this context costs is real work and is queued as its own
    # task — this number is a guard against a *dump*, not the optimisation.
    assert len(ctx) < 20_000


def test_refresh_reintrospects(kb):
    v1 = kb.version
    kb.refresh()
    assert kb.version == v1  # nothing changed → same content hash


# ── the craft layer: guidance that is executable, so it cannot quietly rot ──────

def _package_from_graph(graph_dict, tmp_path):
    """Wrap a recipe graph in the smallest real WorkflowPackage."""
    from neurosurfer.graph.engine.schema import Graph
    from neurosurfer.graph.workflow.package import WorkflowManifest, WorkflowPackage

    graph = Graph.model_validate(graph_dict)
    return WorkflowPackage(
        manifest=WorkflowManifest(
            name=graph.name, description="knowledge-layer recipe", version="0.0.1"
        ),
        graph=graph,
        path=tmp_path,
    )


@pytest.mark.parametrize("key", sorted(_RECIPES))
def test_recipe_passes_the_real_validator(key, tmp_path):
    """Every worked shape must survive the validator a real build runs.

    This is what separates this layer from prose: guidance describing a node the
    engine would reject fails here, at the same commit that changed the engine —
    rather than being discovered by a model that followed it.
    """
    from neurosurfer.graph.workflow.validate import validate_package

    pkg = _package_from_graph(_RECIPES[key]["graph"], tmp_path)
    report = validate_package(pkg)
    assert not report.errors, (
        f"recipe '{key}' no longer validates: "
        + "; ".join(f"{i.node_id or '-'}: {i.message}" for i in report.errors)
    )
    # Gaps matter as much as errors here. A tool that no longer exists is a
    # *gap*, not an error — reasonable for a build (the Architect can author the
    # missing tool) and unacceptable for a recipe, which teaches by naming tools
    # that are supposed to be there. Without this the shapes would keep passing
    # while pointing at tools that had been renamed away.
    assert not report.gaps, (
        f"recipe '{key}' names something that no longer exists: "
        + "; ".join(f"{i.node_id or '-'}: {i.message}" for i in report.gaps)
    )


def test_tool_recipes_supply_every_required_argument():
    """The rule the recipes exist to teach, asserted against the tools' schemas.

    A `tool` node has no model, so `tool_args` is the entire instruction. A recipe
    that omitted a required parameter would be teaching the exact defect that
    produced seven unrunnable SQL nodes.
    """
    from neurosurfer.tools.registry import all_tools

    schemas = {t.name: t.schema.input_schema for t in all_tools()}
    checked = 0
    for key, recipe in _RECIPES.items():
        for node in recipe["graph"]["nodes"]:
            if node.get("kind") != "tool":
                continue
            tool_name = node["tools"][0]
            required = set(schemas[tool_name].get("required") or [])
            supplied = set(node.get("tool_args") or {})
            assert required <= supplied, (
                f"recipe '{key}': tool node '{node['id']}' calls {tool_name} "
                f"without {sorted(required - supplied)}"
            )
            checked += 1
    assert checked, "no tool recipes found — the contrastive pair is the point"


def test_secret_recipe_keeps_the_credential_out_of_every_prompt():
    """`${NAME}` belongs in tool_args and nowhere a model can read."""
    graph = _RECIPES["tool_with_secret"]["graph"]
    node = graph["nodes"][0]
    assert node["secrets"], "the secret must be declared to be substituted"
    assert "${API_TOKEN}" in json.dumps(node["tool_args"])
    prompt_text = " ".join(
        str(node.get(f, "")) for f in ("goal", "purpose", "expected_result")
    )
    assert "${" not in prompt_text
    # And it is never a graph input — the defect this recipe is teaching against.
    assert not any(
        "token" in i["name"].lower() or "secret" in i["name"].lower()
        for i in graph.get("inputs", [])
    )


def test_always_on_context_carries_the_rules_that_were_being_dropped(kb):
    """`requires` and `notes` were manifest-only, so only a model that thought to
    call `describe_node_kind` ever saw them — and they hold the two rules most
    often broken."""
    ctx = kb.render_context()
    assert "tool_args supplying every required parameter" in ctx
    assert "${NAME}" in ctx and "secrets:" in ctx.replace("`secrets`", "secrets:")
    assert "Build rules" in ctx
    # The contrastive pair, which is the cheapest way to teach tool-vs-react.
    assert "Worked shapes" in ctx
    assert '"kind": "react"' in ctx and '"kind": "tool"' in ctx


def test_build_rules_cite_the_build_that_earned_them():
    """The discipline that keeps this list from becoming speculative bloat."""
    for rule in _BUILD_RULES:
        assert rule["rule"] and rule["why"] and rule["seen"]


def test_roadmap_is_not_indexed(kb):
    """It name-drops every feature, so BM25 ranks it for almost any build query."""
    assert kb._docs.sections, "docs index is empty — the exclusion went too far"
    assert not [s for s in kb._docs.sections if s.path.startswith("about/roadmap")]


# ── docs retrieval ──────────────────────────────────────────────────────────────

def test_docs_index_finds_relevant_sections(kb):
    hits = kb.search_docs("workflow package registry", k=5)
    assert hits, "docs search returned nothing"
    # `guides/graph-workflows.md` was one page holding a subsystem; the docs plan
    # split it into the `graph/` section, so the question this asserts — does a
    # workflow-package query reach the workflow-package page — now lands on
    # `graph/packages.md`. Matching the section rather than one filename keeps the
    # test about retrieval instead of about where a page currently sits.
    assert any(h["path"].startswith("graph/") or "workflow" in h["path"].lower()
               for h in hits)

    mcp_hits = kb.search_docs("connect MCP server tools", k=5)
    assert any("mcp" in h["path"].lower() for h in mcp_hits)


def test_docs_index_empty_dir_is_safe(tmp_path):
    idx = DocsIndex(tmp_path / "nope")
    assert idx.search("anything") == []

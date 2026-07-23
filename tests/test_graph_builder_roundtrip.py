"""Phase 1j — GraphBuilder programmatic API + YAML/JSON round-trip parity.

Proves builder-authored graphs and YAML-authored graphs produce the same
validated IR, and that a graph serializes to JSON and reloads unchanged — the
contract the UI (Phase 7) and the API (Phase 2) depend on.
"""

from __future__ import annotations

import json

import yaml

from neurosurfer.graph import Graph, GraphBuilder, GraphExecutor
from neurosurfer.graph.engine.loader import load_graph_from_dict


class _EchoProvider:
    from neurosurfer.llm.capabilities import ProviderCapabilities

    model = "echo"
    capabilities = ProviderCapabilities(
        context_window=8192, max_output_tokens=2048,
        supports_thinking=False, supports_prompt_cache=False,
        supports_token_count=False, tool_call_style="openai",
    )


FN = "tests.test_graph_builder_roundtrip"


def _double(item=None, x=None, **kwargs):
    v = item if item is not None else x
    return (v or 0) * 2


def _mark(**kwargs):
    return "ran"


# ── builder builds a valid, runnable graph ──────────────────────────────────────

def test_builder_produces_runnable_router_graph():
    g = (
        GraphBuilder("triage", description="route by size")
        .input("n", type="integer", required=True)
        .function("classify", callable=f"{FN}._mark")
        .router(
            "route",
            cases=[
                {"when": "inputs.n > 5", "to": "big"},
                {"when": "inputs.n <= 5", "to": "small"},
            ],
            default="small",
            depends_on=["classify"],
        )
        .function("big", callable=f"{FN}._mark", depends_on=["route"])
        .function("small", callable=f"{FN}._mark", depends_on=["route"])
        .outputs("big", "small")
        .build()
    )
    assert isinstance(g, Graph)
    ex = GraphExecutor(g, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"n": 10})
    assert res.nodes["route"].raw_output == "big"
    assert res.nodes["big"].skipped is False
    assert res.nodes["small"].skipped is True


def test_builder_map_with_as_alias():
    g = (
        GraphBuilder("mapper")
        .input("nums", type="array")
        .map("doubled", over="inputs.nums", as_="item",
             body=[{"id": "d", "kind": "function", "callable": f"{FN}._double"}])
        .outputs("doubled")
        .build()
    )
    # `as_` maps to the YAML alias `as` → item_var
    node = g.node_map()["doubled"]
    assert node.item_var == "item"
    ex = GraphExecutor(g, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"nums": [1, 2, 3]})
    assert res.nodes["doubled"].raw_output == [2, 4, 6]


# ── round-trip parity ───────────────────────────────────────────────────────────

def test_builder_output_equals_yaml_output():
    builder_graph = (
        GraphBuilder("parity")
        .input("n", type="integer")
        .function("a", callable=f"{FN}._mark")
        .function("b", callable=f"{FN}._mark", depends_on=["a"], when="inputs.n > 0")
        .outputs("b")
        .build()
    )
    yaml_spec = {
        "name": "parity",
        "inputs": [{"name": "n", "type": "integer", "required": True}],
        "nodes": [
            {"id": "a", "kind": "function", "callable": f"{FN}._mark"},
            {"id": "b", "kind": "function", "callable": f"{FN}._mark",
             "depends_on": ["a"], "when": "inputs.n > 0"},
        ],
        "outputs": ["b"],
    }
    yaml_graph = load_graph_from_dict(yaml_spec)
    # Same nodes, edges, guards.
    assert [n.id for n in builder_graph.nodes] == [n.id for n in yaml_graph.nodes]
    assert builder_graph.node_map()["b"].when == yaml_graph.node_map()["b"].when


def test_graph_json_round_trip():
    g = (
        GraphBuilder("rt")
        .input("nums", type="array")
        .map("doubled", over="inputs.nums",
             body=[{"id": "d", "kind": "function", "callable": f"{FN}._double"}])
        .loop("counter", max_iterations=3, break_when="index >= 2",
              body=[{"id": "s", "kind": "function", "callable": f"{FN}._mark"}])
        .outputs("doubled")
        .build()
    )
    # Serialize to JSON (what the UI/API consume) and reload.
    as_json = json.dumps(g.model_dump(mode="json"))
    reloaded = load_graph_from_dict(json.loads(as_json))
    assert reloaded.name == "rt"
    assert reloaded.node_map()["counter"].max_iterations == 3
    assert reloaded.node_map()["counter"].break_when == "index >= 2"
    assert reloaded.node_map()["doubled"].body[0].callable == f"{FN}._double"


def test_graph_yaml_round_trip():
    g = (
        GraphBuilder("yaml_rt")
        .input("n", type="integer")
        .function("a", callable=f"{FN}._mark")
        .base("b", purpose="do it", depends_on=["a"], when="inputs.n > 1")
        .outputs("b")
        .build()
    )
    text = yaml.dump(g.model_dump(mode="json"))
    reloaded = load_graph_from_dict(yaml.safe_load(text))
    assert reloaded.node_map()["b"].when == "inputs.n > 1"
    assert reloaded.node_map()["b"].purpose == "do it"

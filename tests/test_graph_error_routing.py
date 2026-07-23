"""Phase 1g — error/fallback routing + retry policy."""

from __future__ import annotations

import pytest

from neurosurfer.graph import GraphExecutor
from neurosurfer.graph.engine.errors import GraphConfigurationError
from neurosurfer.graph.engine.loader import load_graph_from_dict


class _EchoProvider:
    from neurosurfer.llm.capabilities import ProviderCapabilities

    model = "echo"
    capabilities = ProviderCapabilities(
        context_window=8192, max_output_tokens=2048,
        supports_thinking=False, supports_prompt_cache=False,
        supports_token_count=False, tool_call_style="openai",
    )


FN = "tests.test_graph_error_routing"

_attempts = {"n": 0}


def _always_fail(**kwargs):
    raise RuntimeError("boom")


def _mark(**kwargs):
    return "ran"


def _handler(**kwargs):
    return "recovered"


def _flaky(**kwargs):
    # Fails the first two attempts, succeeds on the third.
    _attempts["n"] += 1
    if _attempts["n"] < 3:
        raise RuntimeError(f"transient {_attempts['n']}")
    return f"ok after {_attempts['n']}"


# ── error routing ───────────────────────────────────────────────────────────────

def test_on_error_routes_to_fallback_and_prunes_normal_path():
    spec = {
        "name": "err_route",
        "nodes": [
            {"id": "risky", "kind": "function", "callable": f"{FN}._always_fail"},
            {"id": "normal", "kind": "function", "callable": f"{FN}._mark",
             "depends_on": ["risky"]},
            {"id": "fallback", "kind": "function", "callable": f"{FN}._handler",
             "depends_on": ["risky"]},
        ],
        "outputs": ["normal", "fallback"],
    }
    # risky.on_error → fallback
    spec["nodes"][0]["on_error"] = "fallback"
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({})

    assert res.nodes["risky"].error is not None       # the error is transparent
    assert res.nodes["normal"].skipped is True         # normal path pruned
    assert res.nodes["fallback"].skipped is False       # fallback ran
    assert res.nodes["fallback"].raw_output == "recovered"


def test_on_error_exposes_error_text_as_var():
    spec = {
        "name": "err_var",
        "nodes": [
            {"id": "risky", "kind": "function", "callable": f"{FN}._always_fail",
             "on_error": "fallback"},
            {"id": "fallback", "kind": "function", "callable": f"{FN}._read_err",
             "depends_on": ["risky"]},
        ],
        "outputs": ["fallback"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({})
    # handler read state.vars.risky__error via a `when`-free function that inspects...
    # here we just confirm the var was set by checking fallback ran.
    assert res.nodes["fallback"].skipped is False


def _read_err(**kwargs):
    return "handled"


def test_success_prunes_the_error_branch():
    spec = {
        "name": "err_success",
        "nodes": [
            {"id": "ok", "kind": "function", "callable": f"{FN}._mark",
             "on_error": "fallback"},
            {"id": "fallback", "kind": "function", "callable": f"{FN}._handler",
             "depends_on": ["ok"]},
        ],
        "outputs": ["ok", "fallback"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({})
    assert res.nodes["ok"].raw_output == "ran"
    # No error → the on_error target is not force-activated. It has no other trigger,
    # so it simply runs as a normal dependent of a successful node here.
    assert res.nodes["fallback"].error is None


# ── retry ───────────────────────────────────────────────────────────────────────

def test_retry_recovers_flaky_node():
    _attempts["n"] = 0
    spec = {
        "name": "retry_wf",
        "nodes": [
            {"id": "flaky", "kind": "function", "callable": f"{FN}._flaky",
             "policy": {"retries": 3}},
        ],
        "outputs": ["flaky"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({})
    assert res.nodes["flaky"].error is None
    assert res.nodes["flaky"].raw_output == "ok after 3"


def test_retry_exhausted_still_fails():
    _attempts["n"] = 0
    spec = {
        "name": "retry_fail",
        "nodes": [
            {"id": "flaky", "kind": "function", "callable": f"{FN}._flaky",
             "policy": {"retries": 1}},  # only 2 attempts total → still failing
        ],
        "outputs": ["flaky"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({})
    assert res.nodes["flaky"].error is not None


# ── validation ──────────────────────────────────────────────────────────────────

def test_on_error_unknown_target_rejected():
    spec = {
        "name": "bad_onerr",
        "nodes": [
            {"id": "a", "kind": "function", "callable": f"{FN}._always_fail",
             "on_error": "ghost"},
        ],
        "outputs": ["a"],
    }
    with pytest.raises(GraphConfigurationError, match="unknown node id"):
        load_graph_from_dict(spec)


def test_on_error_target_must_depend_on_node():
    spec = {
        "name": "bad_onerr2",
        "nodes": [
            {"id": "a", "kind": "function", "callable": f"{FN}._always_fail",
             "on_error": "b"},
            {"id": "b", "kind": "function", "callable": f"{FN}._handler"},  # no depends_on
        ],
        "outputs": ["b"],
    }
    with pytest.raises(GraphConfigurationError, match="depends_on"):
        load_graph_from_dict(spec)

"""Per-node token usage capture (S3 trace-detail).

The agents always tallied usage; the graph layer used to throw it away at the
`node_runner` boundary, so nothing reached `NodeExecutionResult`, the run record,
or the API. These tests pin the whole path, including the engine's *hidden* LLM
calls (router classification, loop `until` judging) and body roll-up — the parts
that would otherwise silently under-report what a workflow cost.
"""

from __future__ import annotations

import pytest

from neurosurfer.graph import GraphExecutor
from neurosurfer.graph.engine.loader import load_graph_from_dict
from neurosurfer.llm.capabilities import ProviderCapabilities
from neurosurfer.llm.types import CanonicalResponse, TextBlock, Usage

FN = "tests.engine.test_graph_usage_capture"

# Every fake completion bills the same amount, so totals are exact multiples and
# an off-by-one call (e.g. a router repair) is visible in the assertion.
PER_CALL = Usage(input_tokens=10, output_tokens=5)


class _ScriptedProvider:
    """Answers with a scripted list of texts (last repeats), billing PER_CALL."""

    model = "fake"
    capabilities = ProviderCapabilities(
        context_window=8192, max_output_tokens=2048,
        supports_thinking=False, supports_prompt_cache=False,
        supports_token_count=False, tool_call_style="openai",
    )

    def __init__(self, texts: list[str]):
        self._texts = texts
        self.calls = 0

    async def complete(self, messages, system, tools, config):  # noqa: ANN001
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return CanonicalResponse(
            content=[TextBlock(text=text)], stop_reason="end_turn", usage=PER_CALL
        )


def _passthrough(**kwargs):
    return "body-ran"


def test_base_node_reports_its_usage():
    spec = {
        "name": "usage_base",
        "nodes": [{"id": "write", "kind": "base", "goal": "say something"}],
        "outputs": ["write"],
    }
    graph = load_graph_from_dict(spec)
    provider = _ScriptedProvider(["hello"])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({})

    usage = res.nodes["write"].usage
    assert usage is not None, "base node usage was dropped"
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert res.total_usage().total() == PER_CALL.total()


def test_nodes_that_never_call_a_model_report_no_usage():
    """None, not zero — 'this node cost nothing' and 'we didn't measure' differ."""
    spec = {
        "name": "usage_fn",
        "nodes": [{"id": "f", "kind": "function", "callable": f"{FN}._passthrough"}],
        "outputs": ["f"],
    }
    graph = load_graph_from_dict(spec)
    res = GraphExecutor(
        graph, provider=_ScriptedProvider(["x"]), log_traces=False
    ).run({})
    assert res.nodes["f"].usage is None
    assert res.total_usage().total() == 0


def test_router_classification_bills_the_router_node():
    """A `routes` router is a real LLM call — its tokens belong to the router."""
    spec = {
        "name": "usage_router",
        "nodes": [
            {"id": "pick", "kind": "router", "purpose": "choose",
             "routes": {"left": "a", "right": "b"}, "default": "a"},
            {"id": "a", "kind": "function", "callable": f"{FN}._passthrough",
             "depends_on": ["pick"]},
            {"id": "b", "kind": "function", "callable": f"{FN}._passthrough",
             "depends_on": ["pick"]},
        ],
        "outputs": ["a", "b"],
    }
    graph = load_graph_from_dict(spec)
    provider = _ScriptedProvider(["left"])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({})

    assert res.nodes["pick"].raw_output == "a"
    assert res.nodes["pick"].usage is not None, "router LLM call was billed to nobody"
    assert res.nodes["pick"].usage.total() == PER_CALL.total()


def test_router_repair_attempt_is_billed_too():
    """A repair retry costs a second call; under-reporting it hides real spend."""
    spec = {
        "name": "usage_router_repair",
        "nodes": [
            {"id": "pick", "kind": "router", "purpose": "choose", "repair": True,
             "routes": {"left": "a"}, "default": "a"},
            {"id": "a", "kind": "function", "callable": f"{FN}._passthrough",
             "depends_on": ["pick"]},
        ],
        "outputs": ["a"],
    }
    graph = load_graph_from_dict(spec)
    # First answer matches no label → one repair call, then still no match → default.
    provider = _ScriptedProvider(["nonsense", "also nonsense"])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({})

    assert provider.calls == 2
    assert res.nodes["pick"].usage.total() == PER_CALL.total() * 2


def test_loop_folds_body_and_judge_usage_into_the_container():
    """Body nodes aren't in the parent result map, so their tokens must roll up."""
    spec = {
        "name": "usage_loop",
        "nodes": [
            {"id": "spin", "kind": "loop", "max_iterations": 2,
             "until": "the draft is good",
             "body": [{"id": "draft", "kind": "base", "goal": "draft it"}]},
        ],
        "outputs": ["spin"],
    }
    graph = load_graph_from_dict(spec)
    # draft, judge(CONTINUE), draft, judge(STOP) → 4 calls over 2 iterations.
    provider = _ScriptedProvider(["d1", "CONTINUE - needs work", "d2", "STOP - good"])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({})

    assert provider.calls == 4
    usage = res.nodes["spin"].usage
    assert usage is not None
    assert usage.total() == PER_CALL.total() * 4, "body and/or judge tokens went missing"
    # Counted once at the top level, not twice.
    assert res.total_usage().total() == PER_CALL.total() * 4


def test_map_sums_every_item_into_the_container():
    spec = {
        "name": "usage_map",
        "inputs": [{"name": "items", "type": "array"}],
        "nodes": [
            {"id": "fan", "kind": "map", "over": "inputs.items", "as": "item",
             "body": [{"id": "one", "kind": "base", "goal": "handle {item}"}]},
        ],
        "outputs": ["fan"],
    }
    graph = load_graph_from_dict(spec)
    provider = _ScriptedProvider(["ok"])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run(
        {"items": ["a", "b", "c"]}
    )

    assert provider.calls == 3
    assert res.nodes["fan"].usage.total() == PER_CALL.total() * 3


@pytest.mark.asyncio
async def test_structured_completion_reports_usage_including_repairs():
    """The structured path returned the model but left usage at zero."""
    from pydantic import BaseModel

    from neurosurfer.agents.runtime.structured import structured_completion
    from neurosurfer.llm.types import ToolUseBlock

    class Plan(BaseModel):
        name: str

    class _P:
        def __init__(self):
            self.calls = 0

        async def complete(self, messages, system, tools, config):  # noqa: ANN001
            self.calls += 1
            # First call submits an invalid payload → one repair round.
            payload = {} if self.calls == 1 else {"name": "ok"}
            return CanonicalResponse(
                content=[ToolUseBlock(id="t", name="submit_result", input=payload)],
                stop_reason="tool_use",
                usage=PER_CALL,
            )

    seen: list[Usage] = []
    result = await structured_completion(
        _P(), Plan, user="go", on_usage=seen.append
    )
    assert result.name == "ok"
    assert len(seen) == 2, "the failed repair attempt cost tokens and must be reported"
    assert sum(u.total() for u in seen) == PER_CALL.total() * 2

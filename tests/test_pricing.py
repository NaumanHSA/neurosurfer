"""Turning token counts into money.

`Usage` threaded through every layer of this framework and never once became a
number anyone budgets in. The tests that matter most here are about the two
honest-answer decisions: an unpriced model reports `None` rather than zero, and
a sub-cent run reports its actual size rather than `$0.00`.
"""

from __future__ import annotations

import pytest

from neurosurfer.llm.pricing import (
    PRICES,
    ModelPrice,
    estimate_cost,
    format_cost,
    price_for,
)
from neurosurfer.llm.types import Usage


class TestPriceLookup:
    def test_an_exact_id(self):
        assert price_for("claude-opus-5").input == 5.00

    def test_a_bedrock_prefixed_id(self):
        """`anthropic.claude-opus-5` is the same model at the same price."""
        assert price_for("anthropic.claude-opus-5") == price_for("claude-opus-5")

    def test_a_date_pinned_snapshot(self):
        assert price_for("claude-haiku-4-5-20251001").output == 5.00

    def test_longest_prefix_wins(self):
        """`claude-opus-4-8` must not lose to a shorter key that also matches."""
        assert price_for("claude-opus-4-8") is PRICES["claude-opus-4-8"]

    def test_case_is_ignored(self):
        assert price_for("Claude-Opus-5") is not None

    def test_an_unknown_model_is_none_not_a_guess(self):
        assert price_for("some-model-nobody-has-heard-of") is None

    def test_no_model_is_none(self):
        assert price_for(None) is None
        assert price_for("") is None


class TestEstimateCost:
    def test_input_and_output_are_priced_separately(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)

        assert estimate_cost("claude-opus-5", usage) == pytest.approx(5.00 + 25.00)

    def test_cache_reads_are_a_tenth_of_input(self):
        usage = Usage(cache_read_input_tokens=1_000_000)

        assert estimate_cost("claude-opus-5", usage) == pytest.approx(0.50)

    def test_cache_writes_carry_the_premium(self):
        usage = Usage(cache_creation_input_tokens=1_000_000)

        assert estimate_cost("claude-opus-5", usage) == pytest.approx(6.25)

    def test_an_unpriced_model_costs_none_not_zero(self):
        """`$0.00` for an unknown model quietly under-reports a bill; `None`
        forces the caller to decide."""
        assert estimate_cost("mystery-model", Usage(input_tokens=1_000_000)) is None

    def test_a_local_model_costs_zero_deliberately(self):
        """Not a claim that inference is free — a claim there is no invoice."""
        assert estimate_cost("local", Usage(input_tokens=1_000_000)) == 0.0

    def test_no_usage_is_none(self):
        assert estimate_cost("claude-opus-5", None) is None

    def test_a_realistic_small_run(self):
        usage = Usage(input_tokens=1_200, output_tokens=350)

        cost = estimate_cost("claude-haiku-4-5", usage)
        assert cost == pytest.approx((1_200 * 1.0 + 350 * 5.0) / 1_000_000)

    def test_a_custom_rate_can_be_installed(self):
        """The table is data — a caller with negotiated pricing replaces it."""
        PRICES["negotiated-model"] = ModelPrice(input=0.5, output=1.0)
        try:
            assert estimate_cost(
                "negotiated-model", Usage(input_tokens=1_000_000)
            ) == pytest.approx(0.5)
        finally:
            del PRICES["negotiated-model"]


class TestFormatCost:
    def test_sub_cent_costs_keep_their_digits(self):
        """A run that reports `$0.00` looks like a run nobody measured."""
        assert format_cost(0.0004) == "$0.0004"

    def test_ordinary_costs_are_two_decimals(self):
        assert format_cost(1.239) == "$1.24"

    def test_exact_zero_is_zero(self):
        assert format_cost(0.0) == "$0.00"

    def test_unpriced_is_not_a_number(self):
        assert format_cost(None) == "n/a"


class TestItReachesTheResults:
    def test_a_run_result_prices_itself(self):
        from neurosurfer.agents.conversation.events import RunResult

        result = RunResult(
            usage=Usage(input_tokens=1_000_000), model="claude-opus-5"
        )

        assert result.cost() == pytest.approx(5.00)

    def test_a_run_result_with_an_unpriced_model_says_none(self):
        from neurosurfer.agents.conversation.events import RunResult

        result = RunResult(usage=Usage(input_tokens=1_000), model="mystery")

        assert result.cost() is None

    def test_a_graph_prices_each_node_at_its_own_model(self):
        """A graph that mixes an expensive planner with a cheap worker must not
        be priced entirely at one rate."""
        from neurosurfer.graph.engine.schema import (
            Graph,
            GraphExecutionResult,
            GraphNode,
            NodeExecutionResult,
        )

        def node(node_id, model, tokens):
            return NodeExecutionResult(
                node_id=node_id,
                mode="text",
                raw_output="x",
                started_at=0.0,
                duration_ms=1,
                usage=Usage(input_tokens=tokens),
                model=model,
            )

        graph = Graph(name="mixed", nodes=[GraphNode(id="a", kind="base")], outputs=["a"])
        result = GraphExecutionResult(
            graph=graph,
            nodes={
                "planner": node("planner", "claude-opus-5", 1_000_000),
                "worker": node("worker", "claude-haiku-4-5", 1_000_000),
            },
            final={},
        )

        assert result.total_cost() == pytest.approx(5.00 + 1.00)
        assert "$6.00" in result.execution_summary()

    def test_a_graph_of_unpriced_nodes_reports_no_cost(self):
        from neurosurfer.graph.engine.schema import (
            Graph,
            GraphExecutionResult,
            GraphNode,
            NodeExecutionResult,
        )

        graph = Graph(name="g", nodes=[GraphNode(id="a", kind="base")], outputs=["a"])
        result = GraphExecutionResult(
            graph=graph,
            nodes={
                "a": NodeExecutionResult(
                    node_id="a", mode="text", raw_output="x", started_at=0.0,
                    duration_ms=1, usage=Usage(input_tokens=99), model="mystery",
                )
            },
            final={},
        )

        assert result.total_cost() is None
        assert "$" not in result.execution_summary()

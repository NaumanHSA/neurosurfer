"""A run handed a value no step reads should say so.

The gap this closes is the one that let a broken flagship tutorial ship.
`declared_inputs_are_read_by_something` inspects *declared* inputs; tutorial 03's
`content_pipeline` declared none, was run with `{"user_intent": …}`, and went
green while the model replied "please provide the specific topic". Nothing
anywhere was positioned to notice.

The interesting half of these tests is the silence: a warning that fires on a
working graph would be worse than the gap it closes, and three shapes of working
graph would trip a naive version of this check.
"""

from __future__ import annotations

import logging

from neurosurfer.graph.engine.loader import load_graph_from_dict
from neurosurfer.graph.engine.utils import normalize_and_validate_graph_inputs

WARN = "will not reach any node"


def _normalize(graph_dict, inputs, caplog):
    graph = load_graph_from_dict(graph_dict)
    with caplog.at_level(logging.WARNING, logger="neurosurfer.graph.engine.utils"):
        out = normalize_and_validate_graph_inputs(graph, inputs)
    return out, "\n".join(r.getMessage() for r in caplog.records)


def _base(**over):
    node = {"id": "step", "kind": "base", "goal": "Do the thing."}
    node.update(over)
    return {"name": "t", "nodes": [node], "outputs": ["step"]}


class TestItReportsTheGapThatShipped:
    def test_the_tutorial_03_shape_is_reported(self, caplog):
        """The exact graph that went green while answering nothing."""
        out, log = _normalize(_base(), {"user_intent": "explain attention"}, caplog)
        assert WARN in log and "user_intent" in log
        # Reported, not refused — the value is still passed through untouched.
        assert out == {"user_intent": "explain attention"}

    def test_the_message_names_the_fix(self, caplog):
        _, log = _normalize(_base(), {"topic": "x"}, caplog)
        assert "{topic}" in log

    def test_only_the_unread_keys_are_named(self, caplog):
        _, log = _normalize(
            _base(goal="Research {topic}."), {"topic": "x", "stray": "y"}, caplog
        )
        assert "stray" in log and "'topic'" not in log


class TestItStaysQuietOnAGraphThatWorks:
    """Each of these would trip a check that only looked at prompt placeholders."""

    def test_an_interpolated_input_is_read(self, caplog):
        _, log = _normalize(_base(goal="Research {topic}."), {"topic": "x"}, caplog)
        assert WARN not in log

    def test_a_function_node_takes_the_whole_mapping_as_kwargs(self, caplog):
        """The capstone's shape: `db_path` reaches a function node naming nothing."""
        graph = {
            "name": "t",
            "nodes": [{"id": "load", "kind": "function",
                       "callable": "os.path:basename"}],
            "outputs": ["load"],
        }
        _, log = _normalize(graph, {"db_path": "/tmp/x.db"}, caplog)
        assert WARN not in log

    def test_an_expression_counts_as_reading(self, caplog):
        """`over`/`when` name values directly rather than through `{}`."""
        graph = {
            "name": "t",
            "nodes": [{
                "id": "fan", "kind": "map", "over": "inputs.reviews",
                "item_var": "item",
                "body": [{"id": "one", "kind": "base",
                          "goal": "Summarise: {item}"}],
                "body_outputs": ["one"],
            }],
            "outputs": ["fan"],
        }
        _, log = _normalize(graph, {"reviews": ["a", "b"]}, caplog)
        assert WARN not in log

    def test_a_body_node_reading_it_counts(self, caplog):
        """A name read only inside a nested body is still read."""
        graph = {
            "name": "t",
            "nodes": [{
                "id": "fan", "kind": "map", "over": "inputs.items",
                "item_var": "item",
                "body": [{"id": "one", "kind": "base",
                          "goal": "Apply {rubric} to {item}"}],
                "body_outputs": ["one"],
            }],
            "outputs": ["fan"],
        }
        _, log = _normalize(graph, {"items": ["a"], "rubric": "be terse"}, caplog)
        assert WARN not in log

    def test_a_declared_graph_is_left_to_the_validator(self, caplog):
        """Declaring inputs takes the other path, which already warns."""
        graph = {
            "name": "t",
            "inputs": [{"name": "topic", "type": "string"}],
            "nodes": [{"id": "step", "kind": "base", "goal": "Do the thing."}],
            "outputs": ["step"],
        }
        _, log = _normalize(graph, {"topic": "x"}, caplog)
        assert WARN not in log

    def test_no_inputs_at_all_says_nothing(self, caplog):
        _, log = _normalize(_base(), {}, caplog)
        assert WARN not in log


class _Answers:
    """Minimal provider: one turn, no tools."""

    model = "fake"

    def __init__(self) -> None:
        from neurosurfer.llm.capabilities import ProviderCapabilities

        self.capabilities = ProviderCapabilities(
            context_window=8192, max_output_tokens=2048,
            supports_thinking=False, supports_prompt_cache=False,
            supports_token_count=False, tool_call_style="openai",
        )

    async def complete(self, messages, system, tools, config):  # noqa: ANN001
        from neurosurfer.llm.types import CanonicalResponse, TextBlock, Usage

        return CanonicalResponse(
            content=[TextBlock(text="done")], stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


def test_it_reaches_the_user_through_an_actual_run(caplog):
    """The check is only worth having if a real `run()` surfaces it."""
    from neurosurfer.graph.engine.executor import GraphExecutor

    graph = load_graph_from_dict(_base())
    with caplog.at_level(logging.WARNING, logger="neurosurfer.graph.engine.utils"):
        result = GraphExecutor(graph, provider=_Answers(), log_traces=False).run(
            {"user_intent": "explain attention"}
        )

    log = "\n".join(r.getMessage() for r in caplog.records)
    assert WARN in log and "user_intent" in log
    # And the run still completes — this is a diagnostic, not a gate.
    assert result.succeeded

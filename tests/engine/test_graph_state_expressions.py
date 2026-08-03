"""Phase 1a/1b — WorkflowState + safe expression evaluator.

Covers the typed state object and the restricted-AST predicate evaluator that
conditional edges, routers, and loops rely on, including adversarial expressions
that must be refused (no code execution, no Python-internals reach).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neurosurfer.graph.engine.expressions import (
    ExpressionError,
    evaluate,
    safe_bool,
)
from neurosurfer.graph.engine.state import WorkflowState

# ── WorkflowState ──────────────────────────────────────────────────────────────

def test_state_records_node_outputs_and_vars():
    st = WorkflowState(inputs={"topic": "ai"})
    st.set_node_output("research", {"summary": "hello"})
    st.set_var("count", 3)

    assert st.get_node_output("research") == {"summary": "hello"}
    assert st.get_node_output("missing") is None
    ns = st.namespace()
    assert ns["inputs"]["topic"] == "ai"
    assert ns["nodes"]["research"]["summary"] == "hello"
    assert ns["vars"]["count"] == 3
    assert ns["state"] is ns  # self-reference for `state.*` access


def test_state_child_scope_shares_nodes_and_vars():
    st = WorkflowState(inputs={}, nodes={"a": 1}, vars={"acc": 0})
    child = st.child_scope({"index": 2, "item": {"status": "ok"}})

    # Iteration values are hoisted to the top-level namespace.
    ns = child.namespace()
    assert ns["index"] == 2
    assert ns["item"]["status"] == "ok"

    # Writes to shared vars/nodes are visible to the parent.
    child.set_var("acc", 5)
    child.set_node_output("b", 99)
    assert st.vars["acc"] == 5
    assert st.nodes["b"] == 99


def test_state_snapshot_is_jsonable():
    class Out(BaseModel):
        label: str

    st = WorkflowState(inputs={"n": 1}, nodes={"x": Out(label="urgent")}, vars={"k": [1, 2]})
    snap = st.snapshot()
    assert snap["nodes"]["x"] == {"label": "urgent"}
    assert snap["vars"]["k"] == [1, 2]
    # round-trips through json
    assert st.to_json()


# ── evaluator: allowed constructs ───────────────────────────────────────────────

def _ns(**kw):
    st = WorkflowState(**kw)
    return st.namespace()


def test_eval_comparisons_and_bool_logic():
    ns = _ns(inputs={"count": 7}, nodes={"classify": {"label": "urgent"}})
    assert evaluate("inputs.count > 5", ns) is True
    assert evaluate("inputs.count > 5 and nodes.classify.label == 'urgent'", ns) is True
    assert evaluate("inputs.count < 5 or nodes.classify.label == 'urgent'", ns) is True
    assert evaluate("not (inputs.count == 7)", ns) is False


def test_eval_attribute_falls_through_to_none_when_missing():
    ns = _ns(nodes={})
    # A node that never ran → attribute access yields None, comparison is False,
    # NOT an error (fail-closed semantics for control flow).
    assert evaluate("nodes.classify.label == 'urgent'", ns) is False


def test_eval_pydantic_attribute_access():
    class Out(BaseModel):
        label: str
        score: int

    ns = _ns(nodes={"c": Out(label="ok", score=9)})
    assert evaluate("nodes.c.label == 'ok'", ns) is True
    assert evaluate("nodes.c.score >= 9", ns) is True


def test_eval_subscript_and_len():
    ns = _ns(vars={"items": [10, 20, 30]}, nodes={"d": {"k": "v"}})
    assert evaluate("len(vars.items) > 0", ns) is True
    assert evaluate("vars.items[0] == 10", ns) is True
    assert evaluate('nodes.d["k"] == "v"', ns) is True
    assert evaluate("vars.items[99] == None", ns) is True  # OOB → None


def test_eval_membership_and_string_helpers():
    ns = _ns(inputs={"path": "report.pdf"}, vars={"tags": ["a", "b"]})
    assert evaluate("'a' in vars.tags", ns) is True
    assert evaluate("'z' not in vars.tags", ns) is True
    assert evaluate("endswith(inputs.path, '.pdf')", ns) is True
    assert evaluate("lower('URGENT') == 'urgent'", ns) is True
    assert evaluate("contains(inputs.path, 'report')", ns) is True


def test_eval_arithmetic_and_ifexp():
    ns = _ns(inputs={"a": 4, "b": 2})
    assert evaluate("inputs.a * inputs.b == 8", ns) is True
    assert evaluate("(inputs.a - inputs.b) if inputs.a > inputs.b else 0", ns) == 2


def test_safe_bool_fails_closed_on_bad_expression():
    ns = _ns(inputs={})
    assert safe_bool("this is not valid !!", ns) is False
    assert safe_bool("nonexistent_name > 3", ns, default=False) is False
    assert safe_bool("nonexistent_name > 3", ns, default=True) is True


# ── evaluator: adversarial constructs must be refused ───────────────────────────

@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "().__class__.__bases__",
        "nodes.__class__",
        "open('/etc/passwd')",
        "eval('1+1')",
        "exec('x=1')",
        "[x for x in range(3)]",         # comprehensions not allowed
        "lambda: 1",                      # lambdas not allowed
        "(x := 5)",                       # walrus not allowed
        "obj.strip()",                    # arbitrary method call
        "getattr(nodes, 'x')",            # getattr not whitelisted
    ],
)
def test_eval_refuses_dangerous_expressions(expr):
    ns = _ns(nodes={"x": {"strip": "not a method"}}, inputs={"obj": "  hi  "})
    with pytest.raises(ExpressionError):
        evaluate(expr, ns)


def test_eval_blocks_dunder_attribute_specifically():
    ns = _ns(nodes={"x": {}})
    with pytest.raises(ExpressionError, match="dunder"):
        evaluate("nodes.x.__class__", ns)


def test_eval_pow_exponent_capped():
    ns = _ns(inputs={})
    with pytest.raises(ExpressionError, match="exponent"):
        evaluate("2 ** 100000", ns)


def test_eval_empty_expression_errors():
    with pytest.raises(ExpressionError):
        evaluate("   ", {})

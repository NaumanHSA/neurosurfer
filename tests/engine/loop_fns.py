"""A graph's `functions:` sidecar, as a real one looks.

Standalone on purpose. A sidecar is imported by path, not as part of a package,
so it has no parent to resolve `from ..x import y` against — the first attempt
at this used the test module itself and failed with "attempted relative import
with no known parent package". Real sidecars are helper files next to a
`graph.yaml`, which is exactly this shape.

Each function takes a `LoopIteration` and returns `True` to stop, or
`(stop, reason)` to also set the next iteration's `{feedback}`.
"""

from __future__ import annotations


def stop_at_three(it):
    """Stop once the body's counter reaches 3.

    Reads `it.result` — the body run itself — which is what the sandboxed
    expression this replaces could only reach through the parent state.
    """
    return it.result.nodes["step"].raw_output["value"] >= 3


def never_stops(it):
    """Always False, so only `max_iterations` can end the loop."""
    return False


def stop_with_reason(it):
    """`(stop, reason)`: the reason becomes the next iteration's `{feedback}`."""
    return it.iteration >= 2, f"saw {it.iteration} iteration(s)"


def explodes(it):
    raise ValueError("this condition is broken")


def reads_the_whole_iteration(it):
    """Everything the contract promises, touched once so a rename cannot pass."""
    assert it.index == it.iteration - 1
    assert it.max_iterations >= it.iteration
    assert isinstance(it.history, list) and len(it.history) == it.iteration
    assert isinstance(it.vars, dict)
    assert it.is_last == (it.iteration >= it.max_iterations)
    assert it.result.nodes                     # the body run, in full
    assert it.output is not None or True       # may legitimately be empty
    return it.is_last


NOT_A_FUNCTION = "a module-level string is not a callable"

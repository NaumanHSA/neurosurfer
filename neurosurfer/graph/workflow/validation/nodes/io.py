"""Rules for the two ends of a workflow — `input` and `output`.

## Two warnings that used to fire here and were wrong

**"input node collects 'x', which the graph does not declare as an input."**
That was written before input nodes became the *only* place a workflow declares
what it takes. Once they are the declaration, asking the author to also list the
same name under `graph.inputs` is asking them to say it twice — and the warning
appeared on a graph that was correct by design.

**"orphan node — nothing consumes its output."** An output node is where the
graph stops; the engine refuses anything that depends on one. Nothing consuming
it is not a smell, it is the definition. The orphan rule now skips terminal
kinds — see `graph.nothing_consumes_this_node`.
"""

from __future__ import annotations

from ..models import Severity, ValidationIssue
from ..registry import node_rule


@node_rule(kinds=("input",), severity=Severity.INFO)
def one_declared_field_could_be_free_text(node, ctx, report) -> None:
    """A form with a single box is a question with extra steps.

    Declared fields earn their keep when there are several of them and each
    means something different. With exactly one, the person answering gets a
    labelled form where a sentence would have done, and the workflow gets a
    value it has to unwrap.
    """
    if str(getattr(node, "input_mode", "") or "") != "dict":
        return
    declared = list(ctx.graph.inputs or [])
    if len(declared) != 1:
        return
    report.add(ValidationIssue(
        severity=Severity.INFO,
        kind="input.single_field",
        node_id=node.id,
        message=(
            "This step asks for one named value. A plain message is usually "
            "easier to answer; named values are worth it when there are several."
        ),
        suggestion="Switch it to free text, unless the name matters to a later step.",
    ))


@node_rule(kinds=("output",), severity=Severity.WARNING)
def output_has_something_to_return(node, ctx, report) -> None:
    """An output node with nothing feeding it returns nothing.

    Distinct from the hard error the loader already raises for an output with
    neither a dependency nor a value: this is the case where it *has* a value
    template but nothing upstream, which runs and returns a constant. Usually
    that means a wire was never drawn.
    """
    if node.depends_on:
        return
    if getattr(node, "value", None):
        return  # a literal answer is a choice, not an oversight
    report.add(ValidationIssue(
        severity=Severity.WARNING,
        kind="output.no_source",
        node_id=node.id,
        message=(
            "Nothing is connected to this step, so the workflow finishes without "
            "returning anything."
        ),
        suggestion="Connect the step whose result should be the answer.",
    ))

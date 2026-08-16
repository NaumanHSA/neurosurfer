"""The bar a build is measured against must be one it can clear.

`derive_acceptance` asks a model to write the criteria the *same model* is then
judged against, and the judge fails closed. A more capable model writes a stricter
bar, so capability can raise the bar faster than it raises the ability to clear
it — which is how `gpt-5.1` came to write "no information not present in the
source" for a summariser, fail it repeatedly, and declare the build infeasible.

The criterion was reasonable as an aspiration and impossible as a *test*: a judge
reading one output cannot certify an absence, and no wording of a prompt promises
one. `drop_unfalsifiable` removes that class and nothing else — which is the half
of these tests that matters, because over-matching would delete real criteria and
leave verification vacuous.
"""

from __future__ import annotations

from neurosurfer.architect.agent.verify import (
    AcceptanceCriterion,
    drop_unfalsifiable,
)


def _kept(*descriptions: str) -> list[str]:
    crit = [AcceptanceCriterion(id=f"c{i}", description=d)
            for i, d in enumerate(descriptions)]
    return [c.description for c in drop_unfalsifiable(crit)]


# ── the class that has to go ────────────────────────────────────────────────────

def test_the_criterion_that_ended_a_build_is_dropped():
    """Verbatim from the `gpt-5.1` transcript."""
    assert _kept(
        "The summary introduces no new information not present in the source text"
    ) == []


def test_every_shape_of_unprovable_absence_is_dropped():
    unfalsifiable = [
        "The output contains no hallucinated details",
        "The summary is verifiably grounded in the article",
        "The workflow guarantees the title reflects the summary",
        "The summary never invents facts",
        "The output is free of fabrication",
        "The summary is strictly extractive",
        "Contains no invented statistics",
        "Includes only information that appears in the source document",
        "No claims that are not supported by the input",
    ]
    assert _kept(*unfalsifiable) == [], "one of these survived"


# ── the half that matters more: what must survive ───────────────────────────────

def test_ordinary_criteria_are_untouched():
    """These are all observable from a single output by a person reading it."""
    good = [
        "The summary is exactly three sentences",
        "The title is no more than 10 words",
        "The output is valid JSON with a 'summary' field",
        "The report contains a table of statuses and counts",
        "The summary mentions at least two distinct complaint themes",
        "The file is written to the path given in the input",
    ]
    assert _kept(*good) == good


def test_a_negative_that_is_checkable_survives():
    """Not every negative is unfalsifiable. "Does not include X" is decidable by
    looking; "does not include anything not in the source" is not.

    This is the line the pattern set has to walk, and the reason it matches on
    unprovable *absences* rather than on negative words.
    """
    checkable = [
        "The summary does not include the raw table",
        "The title does not exceed one line",
        "The output has no markdown code fences",
    ]
    assert _kept(*checkable) == checkable


def test_a_good_criterion_survives_beside_a_bad_one():
    """The common real case: one absolute criterion among several sound ones. The
    build should lose the bar it cannot clear and keep the ones it can."""
    kept = _kept(
        "The summary is exactly three sentences",
        "The summary contains no information not present in the article",
        "The title is catchy and under 10 words",
    )
    assert kept == [
        "The summary is exactly three sentences",
        "The title is catchy and under 10 words",
    ]


def test_dropping_everything_leaves_the_caller_a_fallback():
    """`derive_acceptance` substitutes a single criterion from the intent when
    nothing survives, so verification is never vacuous — assert the empty result
    that triggers it, since a silent pass would be worse than a strict bar."""
    assert _kept("No hallucinations", "Nothing invented") == []


# ── the bar does not move once the repair loop starts ───────────────────────────


async def test_the_acceptance_plan_is_derived_once_and_frozen(tmp_path):
    """A repair loop measured against a moving bar cannot converge.

    `gpt-5.1`'s transcript reports "increasingly strict instructions" — the model
    responding to a failing criterion by tightening what it demanded of itself.
    The acceptance plan is cached on the session for exactly this reason: derive
    once, then repair against a fixed target. (The one exception is a broken test
    rig, which re-derives at most once per build — `fixture_retry_used`.)
    """
    from neurosurfer.architect.agent import BuildSession
    from neurosurfer.architect.agent.verify import AcceptancePlan
    from neurosurfer.architect.knowledge import KnowledgeBase
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    s = BuildSession(
        intent="summarise an article",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=KnowledgeBase(),
    )
    assert s.acceptance_plan is None
    first = AcceptancePlan(criteria=[AcceptanceCriterion(id="c", description="three sentences")])
    s.acceptance_plan = first

    # Whatever the design does next, the plan the loop is judged against is this
    # object — `test_workflow` only derives when `acceptance_plan is None`.
    s.nodes = [{"id": "a", "kind": "base", "instructions": "changed"}]
    assert s.acceptance_plan is first
    assert not s.fixture_retry_used, "the one re-derivation is reserved for a broken rig"

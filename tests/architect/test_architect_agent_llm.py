"""Real-LLM integration: the ReAct Architect agent driven by a real model.

The decisive Phase 4 check — a real model must drive the toolbelt end-to-end:
set_workflow → add_node(s) → validate → register — and must declare_blocked on an
impossible request. Defaults to LM Studio (qwen/qwen3.5-9b) and auto-skips when it
is down; override the model/endpoint via NEUROSURFER_TEST_* (see
``tests/_llm_test_provider.py``) to run against a hosted model.

Run explicitly:
    conda run -n LLMs python -m pytest tests/test_architect_agent_llm.py -v
    # or against a hosted model:
    NEUROSURFER_TEST_BASE_URL=openai NEUROSURFER_TEST_MODEL=gpt-4o-mini \\
        conda run -n LLMs python -m pytest tests/test_architect_agent_llm.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .._llm_test_provider import make_provider, provider_ready, skip_reason

pytestmark = pytest.mark.skipif(not provider_ready(), reason=skip_reason())


@pytest.fixture(scope="module")
def provider():
    return make_provider()


def _agent(provider, tmp_path: Path):
    from neurosurfer.architect import ArchitectAgent
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    return ArchitectAgent(
        provider,
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        staging_root=tmp_path / "staging",
        notify=lambda m: print(f"  [architect] {m}"),
        max_turns=30,
    )


async def test_agent_builds_simple_workflow_with_real_llm(provider, tmp_path):
    from neurosurfer.graph.workflow.package import load_package
    from neurosurfer.graph.workflow.validate import validate_package

    agent = _agent(provider, tmp_path)
    path = await agent.build(
        "Build a workflow that takes a short article text as input, summarises it "
        "in 3 sentences, and then writes a catchy title for the summary."
    )
    pkg = load_package(Path(path))
    assert validate_package(pkg).ok
    # A real design: at least two LLM nodes wired in sequence.
    assert len(pkg.graph.nodes) >= 2
    kinds = {n.kind for n in pkg.graph.nodes}
    assert "base" in kinds or "react" in kinds
    # Something depends on something (it's a pipeline, not a bag of nodes).
    assert any(n.depends_on for n in pkg.graph.nodes)


async def test_closed_loop_engine_with_real_llm(provider, tmp_path):
    """Phase 5: the verification engine works against a real model — derive real
    acceptance criteria from the intent, RUN the workflow, and judge the outputs.
    Driven directly (not via the agent's tool choice, which is model-dependent) so
    the assertion is robust; the agent's autonomous use is covered by the scripted
    test_architect_verify suite."""
    from neurosurfer.architect.agent import (
        BuildSession,
        derive_acceptance,
        verify_workflow,
    )
    from neurosurfer.architect.knowledge import KnowledgeBase
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    intent = ("Take an article text and produce a JSON object with a 3-sentence "
              "summary and a catchy title.")
    session = BuildSession(
        intent=intent, staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=KnowledgeBase(), provider=provider,
    )
    session.name = "summ"
    session.inputs = [{"name": "article", "type": "string", "required": True}]
    session.nodes = [
        {"id": "summarize", "kind": "base",
         "purpose": "Summarise the article in exactly 3 sentences.",
         "goal": "Summarise this article in exactly 3 sentences: {article}"},
        {"id": "title", "kind": "base", "depends_on": ["summarize"],
         "purpose": "Write a catchy title for the summary.",
         "goal": "Write one catchy title for the 3-sentence summary."},
    ]
    session.outputs = ["title"]
    session.stage()

    # Real model derives criteria + concrete test inputs from the intent (with
    # required-input backfill so a run can always start).
    plan = await derive_acceptance(
        provider, intent, session.to_yaml(), declared_inputs=session.inputs)
    assert plan.criteria and plan.test_inputs.get("article")

    # Real run of the staged workflow + real judge over the outputs.
    report = await verify_workflow(
        provider, intent=intent,
        package_dir=session.staging_root / session.name, plan=plan,
    )
    assert report.run_ok  # the two base nodes actually executed on the model
    assert len(report.verdicts) == len(plan.criteria)
    # Each criterion got a real boolean verdict from the judge.
    assert all(isinstance(v["passed"], bool) for v in report.verdicts)


_BRANCHING_INTENT = (
    "Build a workflow that takes a customer support ticket as input, decides "
    "whether it is urgent or routine, and drafts an escalation notice for "
    "urgent tickets but a polite standard reply for routine ones."
)


def _branching(pkg):
    """(has_router, has_guards) for a built package."""
    return (
        "router" in [n.kind for n in pkg.graph.nodes],
        sum(1 for n in pkg.graph.nodes if n.when) >= 2,
    )


def _branching_agent(provider, tmp_path, **over):
    from neurosurfer.architect import ArchitectAgent
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    kwargs = dict(
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        staging_root=tmp_path / "staging",
        notify=lambda m: print(f"  [architect] {m}"),
        max_turns=30,
    )
    kwargs.update(over)
    return ArchitectAgent(provider, **kwargs)


async def test_agent_designs_a_branch_with_real_llm(provider, tmp_path):
    """**Structure**: given an intent that warrants branching, does the agent
    *design* one — a router, or when-guards on alternative paths?

    `verify="off"` on purpose. With verification required this assertion sits
    behind the repair loop, because the Architect refuses to register a workflow
    that fails its own verification — so a design that was right on the first
    plan was reported as "the model cannot design a branch". The transcripts say
    otherwise: `gpt-5-mini` produced the router in its *first* plan and then
    ground through 17 graph runs across 6 verification rounds without converging.

    Two questions, two tests. This one is about the design; the next is about
    whether the built graph satisfies its own judge. A red suite should say which.
    """
    from neurosurfer.graph.workflow.package import load_package
    from neurosurfer.graph.workflow.validate import validate_package

    agent = _branching_agent(provider, tmp_path, verify="off")
    pkg = load_package(Path(await agent.build(_BRANCHING_INTENT)))

    assert validate_package(pkg).ok
    has_router, has_guards = _branching(pkg)
    assert has_router or has_guards, (
        f"expected branching (router or ≥2 when-guards); "
        f"got kinds={[n.kind for n in pkg.graph.nodes]}, "
        f"guards={[n.when for n in pkg.graph.nodes]}"
    )


@pytest.mark.slow
async def test_agent_verifies_the_branch_it_designed_with_real_llm(provider, tmp_path):
    """**Behaviour**: does the branching workflow survive the Architect's own
    verification — built, run, and judged?

    Marked slow because it is: this is the repair loop, and it is the expensive
    half of the pair. On `gpt-4o-mini` it gave up at 12 graph runs and on
    `gpt-5-mini` at 17, so budget ~20 minutes and the API spend when it is run
    deliberately. A failure here means the repair loop did not converge, which is
    a different finding from the design being wrong — see the test above.
    """
    from neurosurfer.graph.workflow.package import load_package
    from neurosurfer.graph.workflow.validate import validate_package

    agent = _branching_agent(provider, tmp_path)   # verify="required", the default
    pkg = load_package(Path(await agent.build(_BRANCHING_INTENT)))

    assert validate_package(pkg).ok
    assert any(_branching(pkg)), "a verified build should still be the branching one"


async def test_agent_declares_blocked_with_real_llm(provider, tmp_path):
    from neurosurfer.architect import WorkflowInfeasible

    agent = _agent(provider, tmp_path)
    with pytest.raises(WorkflowInfeasible):
        await agent.build(
            "Build a workflow that logs into my company's production Oracle database "
            "(you don't have the credentials, hostname, or any driver installed and I "
            "will not provide them) and silently deletes old records every night."
        )

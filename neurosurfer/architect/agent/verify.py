"""Closed-loop verification of staged workflows (Phase 5).

The engine that lets the Architect prove its own work before registering:

1. :func:`derive_acceptance` — one LLM call turns the user's intent + the staged
   graph's declared inputs into an :class:`AcceptancePlan`: 2–6 explicit success
   criteria, concrete test inputs, and — when the workflow reads a file or a
   directory — the :class:`Fixture` that *creates* it.
2. :func:`verify_workflow` — actually RUNS the staged package on those inputs
   (in a worker thread; the runner is synchronous), then:
   - a failed run yields a deterministic diagnosis from the node errors (no judge
     call — criteria can't pass on a crashed run);
   - a clean run is scored by an LLM judge, per criterion, fail-closed (a
     criterion the judge doesn't rule on counts as failed), with a diagnosis +
     design suggestions for anything failing.

Fixtures (V3 Phase 5a) exist because a workflow that reads a file cannot be
tested against a *sentence*. The run happens in a throwaway sandbox directory
that the fixture script populates first, and a source path with nothing behind
it is a hard failure that names the fixture it needs — not a placeholder string
handed to `read_file` so it can fail as "no such file".

The report is rendered for the agent's `test_workflow` tool; the agent applies
fixes with its normal graph-editing tools — design revision, not field patching.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from neurosurfer.llm.types import GenerationConfig, Message

# Tolerant JSON extraction moved to `jsonio` in V3 Phase 3 — the planner asks a
# model for structured output too, and needs exactly the same tolerance.
from .jsonio import parse_json as _parse_json

logger = logging.getLogger(__name__)

__all__ = [
    "AcceptanceCriterion",
    "drop_unfalsifiable",
    "AcceptancePlan",
    "Fixture",
    "VerificationReport",
    "derive_acceptance",
    "verify_workflow",
]

_MAX_OUTPUT_CHARS = 1500   # per-node output shown to the judge
_MAX_CRITERIA = 6
_FIXTURE_TIMEOUT_S = 30    # a setup script that hangs is a failed verification


# ── path-shaped inputs ──────────────────────────────────────────────────────────
#
# Lexical, like the Phase 1 capability rules, and for the same reason: the signal
# is in what the graph author called things, and nothing else declares it. The
# stake here is lower than Phase 1's — a wrong guess costs one fixture, and the
# failure message says exactly what to do about it.

_PATH_WORDS = ("path", "file", "dir", "folder", "filename")
_SINK_WORDS = ("output", "out_", "_out", "dest", "target", "save", "write", "report")


def _path_role(spec: dict[str, Any]) -> str | None:
    """``"source"`` (must exist to be read), ``"sink"`` (written by the run), or None.

    A sink is forgiven — the workflow creates it. A source is the case a
    placeholder string can never satisfy.
    """
    name = str(spec.get("name") or "").lower()
    declared = str(spec.get("type") or "string").lower()
    if declared not in {"string", "path"}:
        return None
    if declared == "path" or any(w in name for w in _PATH_WORDS):
        return "sink" if any(w in name for w in _SINK_WORDS) else "source"
    return None


class AcceptanceCriterion(BaseModel):
    id: str = Field(description="Short snake_case id, e.g. 'summary_is_3_sentences'.")
    description: str = Field(description="One testable statement about the workflow's output.")


class Fixture(BaseModel):
    """The world a workflow needs before it can be tested in it.

    The shape is `ToolAuthor.test_setup`, which has worked since V2 Phase 4:
    self-contained stdlib Python. It runs in a subprocess whose cwd is the
    throwaway sandbox, so `creates` are relative paths and the workflow — run
    with the same cwd — resolves them without knowing the sandbox exists.
    """

    setup: str = Field(
        default="",
        description="Self-contained stdlib Python that creates the test files.",
    )
    creates: list[str] = Field(
        default_factory=list,
        description="Relative paths the setup script is expected to create.",
    )

    def __bool__(self) -> bool:
        return bool(self.setup.strip())


class AcceptancePlan(BaseModel):
    criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    test_inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Concrete values for every required declared graph input.",
    )
    # Phase 5a. None when the workflow needs nothing on disk — the common case,
    # and one that must stay free.
    fixtures: Fixture | None = Field(
        default=None,
        description="Files to create before the run, for workflows that read them.",
    )
    # Branch coverage (Phase 6): extra input sets, each designed to exercise a
    # different path (router case / guard direction / loop exit). Judged runs use
    # `test_inputs`; extra cases must merely run cleanly and light up their branch.
    extra_cases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{label, test_inputs}] — one per distinct branch/path.",
    )

    def render(self) -> str:
        lines = ["Acceptance criteria:"]
        lines += [f"  - [{c.id}] {c.description}" for c in self.criteria]
        lines.append(f"Test inputs: {json.dumps(self.test_inputs, ensure_ascii=False)[:800]}")
        if self.fixtures:
            lines.append(
                "Fixtures created before the run: "
                + (", ".join(self.fixtures.creates) or "(unnamed)")
            )
        for case in self.extra_cases:
            lines.append(
                f"Extra case [{case.get('label', '?')}]: "
                f"{json.dumps(case.get('test_inputs', {}), ensure_ascii=False)[:300]}"
            )
        return "\n".join(lines)


@dataclass
class VerificationReport:
    passed: bool
    run_ok: bool
    verdicts: list[dict[str, Any]] = field(default_factory=list)  # {id, passed, reason}
    node_summaries: list[dict[str, Any]] = field(default_factory=list)
    diagnosis: str = ""
    suggestions: str = ""
    # Branch coverage: per-extra-case outcomes + nodes never executed in ANY run.
    case_results: list[dict[str, Any]] = field(default_factory=list)  # {label, ok, error}
    coverage_gaps: list[str] = field(default_factory=list)
    # What the run actually had to work with. A PASSED whose fixture turns out to
    # have been one empty file is a verdict about nothing, and the only way to
    # notice is for the report to say what was on disk.
    fixtures_created: list[str] = field(default_factory=list)
    # Inputs that needed a file and had none. Names the repair rather than
    # describing the failure: the caller re-derives the acceptance plan demanding
    # a fixture for exactly these, instead of the model being told to do
    # something its toolbelt cannot do.
    missing_fixture_for: list[str] = field(default_factory=list)
    # The fixture script itself did not work (raised, timed out, or wrote nothing
    # usable). Like `missing_fixture_for`, this is the harness's problem and not
    # the graph's — see `fixture_problem`.
    fixture_setup_failed: bool = False
    # {input: filename} where the harness repointed an input at the fixture's
    # file. Disclosed because the acceptance plan still shows what the model
    # wrote, and a report claiming an input the run did not use is a small lie.
    paired_inputs: dict[str, str] = field(default_factory=dict)
    # Phase 5b — nodes whose tool this machine cannot provide, so they ran against
    # a stub. A PASSED that quietly tested half the graph is worse than a FAILED,
    # so these go in the headline, not a footnote.
    not_exercised: list[str] = field(default_factory=list)
    stubbed_tools: list[str] = field(default_factory=list)

    @property
    def partial(self) -> bool:
        """True when some step could not be exercised for real."""
        return bool(self.not_exercised)

    @property
    def fixture_problem(self) -> bool:
        """The test rig failed, not the workflow.

        Nothing in the builder's toolbelt can fix one of these, so handing it over
        invites a graph edit that cannot help. Observed twice on gpt-4o-mini: told
        the test file was missing, it started adding `setup_fixtures` /
        `cleanup_fixtures` *function nodes to the workflow*, then looped on the
        validation errors those produced. The caller re-derives the acceptance
        plan instead.
        """
        return bool(self.missing_fixture_for) or self.fixture_setup_failed
    # Full graph executions this verification paid for (main run + branch cases).
    # Verification is the expensive half of a build; counting the runs is what makes
    # that a number rather than an impression.
    graph_runs: int = 0

    def render(self) -> str:
        verdict = "PASSED" if self.passed else "FAILED"
        if self.partial:
            verdict += (
                f" — PARTIAL, {len(self.not_exercised)} step"
                f"{'s' if len(self.not_exercised) != 1 else ''} NOT exercised"
            )
        lines = [f"VERIFICATION {verdict} "
                 f"(run {'ok' if self.run_ok else 'errored'}, "
                 f"{self.graph_runs} graph run{'s' if self.graph_runs != 1 else ''})"]
        if self.not_exercised:
            lines.append(
                "  NOT exercised — "
                + ", ".join(self.not_exercised)
                + " ran against a stub because this machine cannot provide "
                + ", ".join(sorted(self.stubbed_tools))
                + " (not installed, not connected, or missing credentials)."
            )
            lines.append(
                "  Whatever those steps would really return was NOT tested. "
                "Supply what they need and test again to cover them."
            )
        if self.fixtures_created:
            lines.append("Ran against fixture files: " + ", ".join(self.fixtures_created))
        for name, filename in self.paired_inputs.items():
            lines.append(
                f"Input `{name}` was repointed at the fixture's file: {filename}"
            )
        for v in self.verdicts:
            mark = "✓" if v.get("passed") else "✗"
            lines.append(f"  {mark} [{v.get('id')}] {v.get('reason', '')}")
        if self.node_summaries:
            lines.append("Node results:")
            for n in self.node_summaries:
                status = n.get("status")
                extra = n.get("error") or (str(n.get("output"))[:200] if n.get("output") is not None else "")
                lines.append(f"  - {n.get('id')} [{status}] {extra}")
        for c in self.case_results:
            mark = "✓" if c.get("ok") else "✗"
            lines.append(f"  {mark} branch case [{c.get('label')}] "
                         f"{c.get('error') or 'ran cleanly'}")
        if self.coverage_gaps:
            lines.append(
                "COVERAGE WARNING — these nodes never executed in any test case "
                "(dead branch, wrong guard, or missing test case): "
                + ", ".join(self.coverage_gaps)
            )
        if self.diagnosis:
            lines.append(f"Diagnosis: {self.diagnosis}")
        if self.suggestions:
            lines.append(f"Suggested design changes: {self.suggestions}")
        return "\n".join(lines)




# ── acceptance derivation ───────────────────────────────────────────────────────

_DERIVE_SYSTEM = """\
You derive an acceptance test plan for a workflow that is about to be verified.
Output STRICT JSON only, no prose, no code fences:
  {"criteria": [{"id": "<snake_case>", "description": "<testable statement>"}],
   "test_inputs": {<one concrete realistic value for EVERY required input>},
   "fixtures": {"setup": "<python>", "creates": ["<relative path>"]},
   "extra_cases": [{"label": "<branch name>", "test_inputs": {...}}]}
Rules:
- 2 to 6 criteria. Each must be checkable by reading the workflow's outputs alone.
- Criteria must reflect what the USER asked for — not generic quality platitudes.
- Judge the OUTCOME, never the SHAPE. Say what a person would look for — "the file
  contains a table of statuses and counts" — not which keys an object has, what a
  field is named, or how a value is typed. You are writing the bar the build is
  measured against, and a bar invented here is one the build can never clear:
  asked only for a markdown file, a plan once demanded "an object with keys
  `markdown` and `counts`, each containing exactly `result_status` and
  `inquiry_count`" — field names nobody had specified — and failed a workflow that
  had done the job.
- If a clause in the intent is hedged or incidental ("and/or", "for visibility",
  "optionally"), it is NOT a criterion. Criteria are the things the user would be
  disappointed to find missing.
- NEVER write a criterion that demands a GUARANTEE or the ABSENCE of something
  unstated — "no information not present in the source", "no hallucinated
  details", "verifiably grounded", "never invents". You are judging one output by
  reading it, and reading an output cannot certify an absence; an LLM step cannot
  be *made* to satisfy such a bar by any wording, so the build repairs forever and
  then gives up on a workflow that was fine. Write the observable form instead:
  "every figure in the summary also appears in the article" is checkable, "the
  summary invents nothing" is not. (Criteria of this shape are dropped before the
  run, so writing one only costs you a criterion.)
- test_inputs must contain realistic sample data (e.g. an actual short article
  text, not a placeholder like 'test' or 'lorem ipsum').
- fixtures: REQUIRED whenever an input is a file or directory path. The workflow
  will really try to open it, so it has to really be there.
    * `setup` is self-contained Python using only the standard library. It runs
      in an empty scratch directory, so write RELATIVE paths. Prefer plain
      filenames in that directory (`notes.txt`, not `input/notes.txt`); if you do
      use a subdirectory, list the path in `creates` so it gets created for you.
    * `creates` lists ONLY the files the setup script itself writes — never the
      workflow's own node outputs or the results it produces. The matching
      test_inputs value must be exactly one of these paths.
    * The file's CONTENT must be realistic and substantial enough to satisfy the
      criteria — a workflow asked to analyse a document deeply cannot be judged
      on 'lorem ipsum'. Write several real paragraphs.
    * Omit `fixtures` entirely when no input is a path.
- extra_cases: ONLY if the workflow branches (router nodes, `when:` guards): add
  one case per distinct branch, with inputs crafted to trigger that branch (e.g.
  an obviously-urgent ticket vs. an obviously-trivial one). Max 3. Omit (empty
  list) for linear workflows.
"""


#: Criteria that ask for a **guarantee about what is absent**, which no judge
#: reading one output can establish and no redesign can deliver.
#:
#: Deliberately narrow. The failure being prevented is one specific shape — an
#: LLM step required to be provably free of invention — and over-matching here
#: would delete good criteria: "exactly three sentences", "no more than 200
#: words" and "does not include the raw table" are all observable from a single
#: output and all survive. Each pattern below needs an unfalsifiable *negative*,
#: not merely a negative.
_UNFALSIFIABLE = (
    r"\bno\s+(?:new|additional|extra|invented|fabricated|unsupported|external)\b",
    r"\bno\s+(?:information|content|details?|facts?|claims?)\s+(?:that\s+)?"
    r"(?:is\s+|are\s+)?(?:not|beyond|outside)\b",
    r"\bhallucinat",
    r"\bnot\s+present\s+in\s+the\s+(?:source|input|original|article|document)\b",
    r"\bfree\s+of\s+(?:any\s+)?(?:invention|fabrication|hallucination)",
    # Inflected on purpose: "never invents" is the form a model actually writes,
    # and `\binvent\b` does not match it.
    r"\b(?:never|does\s+not\s+ever)\s+\w*\s*(?:invent|fabricat|introduc|add)\w*\b",
    # The bare assertion, with no verb to hang a pattern on: "nothing invented",
    # "nothing fabricated".
    r"\bnothing\s+(?:invented|fabricated|made\s+up|hallucinated|added)\b",
    r"\bverifiabl[ey]\b",
    r"\bguarantee[sd]?\b",
    r"\bstrictly\s+extractive\b",
    r"\bonly\s+(?:information|content|facts?)\s+(?:that\s+)?(?:appears?|is)\s+in\b",
)


def drop_unfalsifiable(
    criteria: list[AcceptanceCriterion],
) -> list[AcceptanceCriterion]:
    """Remove criteria that demand a guarantee rather than an observable result.

    **The bar is written by the same model that is measured against it**, and a
    more capable model writes a stricter bar. `gpt-5.1` derived "no information
    not present in the source" for a summariser, could not prove it of an LLM
    step, repaired the prompt until it ran out of ideas, and declared a two-node
    workflow infeasible. The criterion was not unreasonable as an *aspiration*;
    it was impossible as a *test*, because a judge reading one output cannot
    certify an absence and no wording of a prompt can promise one.

    The prompt already says to judge the outcome and not invent a bar. This is
    the enforcing half, because prose is what a model drops when it is
    struggling — the same reason the capability ladder runs in code.

    Everything left is kept: if *every* criterion is unfalsifiable the caller
    falls back to a single criterion straight from the intent, which is a bar
    that can at least be met.
    """
    keep: list[AcceptanceCriterion] = []
    for c in criteria:
        text = (c.description or "").lower()
        if any(re.search(p, text) for p in _UNFALSIFIABLE):
            logger.info(
                "dropped an unfalsifiable acceptance criterion (%s): %s",
                c.id, c.description,
            )
            continue
        keep.append(c)
    return keep


async def derive_acceptance(
    provider: Any,
    intent: str,
    graph_yaml: str,
    *,
    declared_inputs: list[dict[str, Any]] | None = None,
    require_fixture_for: list[str] | None = None,
) -> AcceptancePlan:
    """One LLM call → criteria + concrete test inputs for the staged workflow.

    *declared_inputs* (the graph's ``inputs``) is used to backfill placeholder
    values for any required input the model didn't supply, so verification can
    always attempt a run rather than failing on a missing input.

    *require_fixture_for* names inputs a previous attempt left with no file
    behind them. Repairing that here is deliberate: the builder's toolbelt edits
    graphs, not acceptance plans, so telling *it* to add a fixture would name a
    move it cannot make.
    """
    demand = ""
    if require_fixture_for:
        named = ", ".join(f"`{n}`" for n in require_fixture_for)
        demand = (
            f"\n\nThe previous attempt could not be verified: {named} names a file "
            f"the workflow opens, and nothing created it. `fixtures` is MANDATORY "
            f"this time. Write a `setup` script that creates a real file for every "
            f"one of {named}, put its relative path in `creates`, and use exactly "
            f"that path as the {named} value in `test_inputs`. The content must be "
            f"substantial enough to judge the criteria against."
        )
    prompt = (
        f"User's request for the workflow:\n{intent}\n\n"
        f"The staged workflow (its declared `inputs` need test values):\n{graph_yaml}\n\n"
        f"Produce the JSON acceptance plan.{demand}"
    )
    response = await provider.complete(
        messages=[Message.user_text(prompt)],
        system=_DERIVE_SYSTEM,
        tools=[],
        config=GenerationConfig(stream=False),
    )
    data = _parse_json(response.text(), want="criteria") or {}
    criteria = []
    for c in (data.get("criteria") or [])[:_MAX_CRITERIA]:
        if isinstance(c, dict) and c.get("description"):
            criteria.append(AcceptanceCriterion(
                id=str(c.get("id") or f"criterion_{len(criteria) + 1}"),
                description=str(c["description"]),
            ))
    criteria = drop_unfalsifiable(criteria)
    if not criteria:
        # Fail-safe: a single criterion straight from the intent.
        criteria = [AcceptanceCriterion(
            id="fulfils_intent",
            description=f"The output fulfils the request: {intent[:300]}",
        )]
    test_inputs = dict(data.get("test_inputs")) if isinstance(data.get("test_inputs"), dict) else {}
    _backfill_inputs(test_inputs, declared_inputs)

    raw_fx = data.get("fixtures")
    fixtures: Fixture | None = None
    if isinstance(raw_fx, dict) and str(raw_fx.get("setup") or "").strip():
        creates = raw_fx.get("creates")
        fixtures = Fixture(
            setup=str(raw_fx["setup"]),
            creates=[str(p) for p in creates if str(p).strip()]
            if isinstance(creates, list) else [],
        )

    extra_cases: list[dict[str, Any]] = []
    raw_cases = data.get("extra_cases")
    if isinstance(raw_cases, list):
        for i, case in enumerate(raw_cases[:3]):
            if not isinstance(case, dict):
                continue
            case_inputs = case.get("test_inputs")
            if not isinstance(case_inputs, dict):
                continue
            case_inputs = dict(case_inputs)
            _backfill_inputs(case_inputs, declared_inputs)
            extra_cases.append({
                "label": str(case.get("label") or f"case_{i + 1}"),
                "test_inputs": case_inputs,
            })
    return AcceptancePlan(criteria=criteria, test_inputs=test_inputs,
                          fixtures=fixtures, extra_cases=extra_cases)


def _backfill_inputs(
    test_inputs: dict[str, Any], declared: list[dict[str, Any]] | None
) -> None:
    """Give every required declared input *some* value so a run can start.

    Every input except a source path. A sentence in a `file_path` is the exact
    lie this phase exists to stop: it turns "verification had nothing to read"
    into "the workflow failed to find a file", which reads like a bug in the
    graph. Leaving it absent lets :func:`verify_workflow` say what really went
    wrong, and what to do about it.
    """
    if not declared:
        return
    placeholders = {
        "string": "Sample input text for testing the workflow end to end.",
        "integer": 3, "float": 1.5, "boolean": True,
        "array": ["a", "b"], "object": {"key": "value"},
    }
    for spec in declared:
        name = spec.get("name")
        if not name or name in test_inputs or not spec.get("required", True):
            continue
        if _path_role(spec) == "source":
            continue
        test_inputs[name] = placeholders.get(str(spec.get("type", "string")), "sample")


# ── judging ─────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are a strict workflow-output judge. Given the user's intent, acceptance
criteria, and the workflow's actual outputs, rule on EVERY criterion.
Output STRICT JSON only, no prose, no code fences:
  {"verdicts": [{"id": "<criterion id>", "passed": true|false, "reason": "<short>"}],
   "diagnosis": "<if anything failed: which node/design aspect is at fault>",
   "suggestions": "<if anything failed: the concrete graph change to make>"}
Judge only from the evidence shown. Be strict: partially met = failed.
Output marked [NOT EXERCISED] came from a stub, not from the real tool: it is
NOT evidence. A criterion that depends on such a step has not been met.
"""


async def _judge(
    provider: Any,
    intent: str,
    plan: AcceptancePlan,
    final_outputs: dict[str, Any],
    node_summaries: list[dict[str, Any]],
    not_exercised: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    criteria_text = "\n".join(f"- [{c.id}] {c.description}" for c in plan.criteria)
    outputs_text = json.dumps(final_outputs, ensure_ascii=False, default=str)[:4000]
    nodes_text = json.dumps(node_summaries, ensure_ascii=False, default=str)[:3000]
    stub_note = ""
    if not_exercised:
        stub_note = (
            "\nThese nodes did NOT run for real — no tool for them on this "
            f"machine, so their output is a stub: {', '.join(not_exercised)}.\n"
            "Fail any criterion that depends on them.\n"
        )
    prompt = (
        f"User intent:\n{intent}\n\n"
        f"Acceptance criteria:\n{criteria_text}\n\n"
        f"Test inputs used: {json.dumps(plan.test_inputs, ensure_ascii=False)[:800]}\n\n"
        f"Workflow FINAL outputs:\n{outputs_text}\n\n"
        f"Per-node results:\n{nodes_text}\n{stub_note}\n"
        "Rule on every criterion."
    )
    response = await provider.complete(
        messages=[Message.user_text(prompt)],
        system=_JUDGE_SYSTEM,
        tools=[],
        config=GenerationConfig(stream=False),
    )
    data = _parse_json(response.text(), want="verdicts") or {}
    raw = data.get("verdicts") if isinstance(data.get("verdicts"), list) else []
    by_id = {str(v.get("id")): v for v in raw if isinstance(v, dict)}
    verdicts: list[dict[str, Any]] = []
    for c in plan.criteria:
        v = by_id.get(c.id)
        if v is None:
            # Fail-closed: an unruled criterion is a failed criterion.
            verdicts.append({"id": c.id, "passed": False,
                             "reason": "judge did not rule on this criterion"})
        else:
            verdicts.append({"id": c.id, "passed": bool(v.get("passed")),
                             "reason": str(v.get("reason", ""))})
    return verdicts, str(data.get("diagnosis", "")), str(data.get("suggestions", ""))


# ── the sandbox the run happens in ──────────────────────────────────────────────

def _run_fixture(fixture: Fixture, sandbox: Any) -> tuple[str, list[str]]:
    """Execute the setup script in *sandbox*. Returns (hard failure reason, unmet).

    A subprocess, not `exec`: this is model-written Python, and the gateway it
    would otherwise run inside is long-lived and serving other builds. The script
    only has to create files, so it loses nothing by being isolated.

    Only a script that *fails to run* is a hard failure. A `creates` entry with
    nothing behind it is merely reported: models list node outputs there as
    readily as filenames (`creates: [doc.txt, analysis_results, report]`, seen on
    gpt-4o-mini), and failing the build over a mislabelled declaration would
    reject a fixture that did its job. What has to be true is checked against the
    inputs the workflow will actually open — see :func:`_missing_sources`.
    """
    import subprocess
    import sys

    # Make room for what the fixture said it would create. Models write
    # `open('input/notes.txt', 'w')` readily, and in a fresh sandbox that raises
    # FileNotFoundError on the missing directory — observed on gpt-4o-mini, where
    # it cost the whole build. The paths are declared, so creating their parents is
    # not a guess.
    for declared in fixture.creates:
        try:
            (sandbox / declared).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # an unusable path will fail in the script, with a better message

    try:
        proc = subprocess.run(
            [sys.executable, "-c", fixture.setup],
            cwd=str(sandbox), capture_output=True, text=True,
            timeout=_FIXTURE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"the fixture setup script did not finish within {_FIXTURE_TIMEOUT_S}s", []
    except Exception as e:  # noqa: BLE001 - a fixture that cannot start is a failure
        return f"the fixture setup script could not be run: {e}", []
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:600]
        return f"the fixture setup script failed:\n{err}", []

    return "", [p for p in fixture.creates if not (sandbox / p).exists()]


def _missing_sources(
    test_inputs: dict[str, Any],
    declared: list[dict[str, Any]] | None,
    sandbox: Any,
) -> list[str]:
    """Required source paths with nothing behind them once fixtures have run."""
    missing = []
    for spec in declared or []:
        name = spec.get("name")
        if not name or not spec.get("required", True):
            continue
        if _path_role(spec) != "source":
            continue
        value = test_inputs.get(name)
        if not isinstance(value, str) or not value.strip() or not (sandbox / value).exists():
            missing.append(str(name))
    return missing


def _pair_lone_fixture_file(
    test_inputs: dict[str, Any],
    declared: list[dict[str, Any]] | None,
    sandbox: Any,
) -> str | None:
    """Point a single unsatisfied source input at a single fixture file.

    Models write the fixture and then forget to reference it — on gpt-4o-mini,
    a setup script creating `input_file.txt` alongside `test_inputs` with no
    `file_path` at all. One file, one input wanting one: the pairing is not a
    guess, and making it in code beats spending a model turn to be told the
    obvious. Anything less clear-cut (two files, two inputs) is left to fail.
    """
    missing = _missing_sources(test_inputs, declared, sandbox)
    if len(missing) != 1:
        return None
    files = [p for p in sorted(sandbox.iterdir()) if p.is_file()]
    if len(files) != 1:
        return None
    test_inputs[missing[0]] = files[0].name
    return missing[0]


# ── execution + verification ────────────────────────────────────────────────────

def _summarise_nodes(result: Any) -> list[dict[str, Any]]:
    out = []
    for nid, nr in result.nodes.items():
        entry: dict[str, Any] = {
            "id": nid,
            "status": ("error" if (nr.error and not nr.skipped)
                       else ("skipped" if nr.skipped else "ok")),
        }
        if nr.error:
            entry["error"] = str(nr.error)[:400]
        elif nr.raw_output is not None:
            entry["output"] = str(nr.raw_output)[:_MAX_OUTPUT_CHARS]
        out.append(entry)
    return out


async def verify_workflow(
    provider: Any,
    *,
    intent: str,
    package_dir: Any,
    plan: AcceptancePlan,
    runner: Any = None,
    declared_inputs: list[dict[str, Any]] | None = None,
    on_node_event: Any = None,
) -> VerificationReport:
    """Run the staged package on the plan's inputs, then judge (clean runs only).

    The run happens in a throwaway sandbox directory, populated by
    ``plan.fixtures`` first. *declared_inputs* (the graph's ``inputs``) lets the
    source-path check know which values have to be real files.

    *runner* is for tests. An injected runner keeps its own ``cwd``, so it will
    not see the sandbox the fixtures were written into.
    """
    import asyncio
    import tempfile
    from pathlib import Path

    from neurosurfer.graph.workflow.package import load_package
    from neurosurfer.graph.workflow.runner import WorkflowRunner

    pkg = load_package(Path(package_dir))
    all_node_ids = [n.id for n in pkg.graph.nodes]
    executed: set[str] = set()

    runs = 0

    with tempfile.TemporaryDirectory(prefix="ns-verify-") as tmp:
        sandbox = Path(tmp)

        unmet: list[str] = []
        if plan.fixtures:
            reason, unmet = await asyncio.to_thread(_run_fixture, plan.fixtures, sandbox)
            if reason:
                return VerificationReport(
                    passed=False, run_ok=False, graph_runs=0,
                    fixtures_created=[], fixture_setup_failed=True,
                    diagnosis=f"Could not set up the test fixtures — {reason}",
                    suggestions=(
                        "The fixture setup script needs fixing: it must be "
                        "self-contained stdlib Python and write relative paths. "
                        "This is the test rig, NOT the workflow — do not change "
                        "the graph, and do not add nodes to create test files."
                    ),
                )

        test_inputs = dict(plan.test_inputs)
        paired_inputs: dict[str, str] = {}
        paired = _pair_lone_fixture_file(test_inputs, declared_inputs, sandbox)
        if paired:
            paired_inputs[paired] = str(test_inputs[paired])
            plan = plan.model_copy(update={"test_inputs": test_inputs})

        missing = _missing_sources(test_inputs, declared_inputs, sandbox)
        if missing:
            named = ", ".join(f"`{m}`" for m in missing)
            return VerificationReport(
                passed=False, run_ok=False, graph_runs=0,
                fixtures_created=sorted(p.name for p in sandbox.iterdir()),
                missing_fixture_for=missing,
                diagnosis=(
                    f"The workflow reads {named}, and no test fixture created "
                    "anything there. It cannot be verified against a file that "
                    "does not exist."
                ),
                suggestions=(
                    f"A fixture that writes realistic content to {named} is "
                    "needed — the test harness creates it, so do NOT add a node "
                    "to the workflow to write it. If the workflow should not be "
                    "reading a file at all, change the design instead."
                ),
            )

        fixtures_created = sorted(p.name for p in sandbox.iterdir())
        if unmet:
            fixtures_created += [f"(declared but not created: {', '.join(unmet)})"]
        # The workflow's tools resolve relative paths against this cwd, so the
        # fixture's `article.txt` and the graph's `article.txt` are one file.
        # `stub_missing_tools` is Phase 5b: a step this machine cannot provide is
        # stubbed so the rest of the graph is still tested, and declared loudly.
        wf_runner = runner or WorkflowRunner(
            provider, cwd=sandbox, stub_missing_tools=True
        )

        return await _verify_in_sandbox(
            provider, intent=intent, plan=plan, pkg=pkg, wf_runner=wf_runner,
            on_node_event=on_node_event,
            all_node_ids=all_node_ids, executed=executed, runs=runs,
            fixtures_created=fixtures_created, paired_inputs=paired_inputs,
        )


def _stubbed_nodes(pkg: Any, stubbed: set[str]) -> list[str]:
    """Nodes that named a stubbed tool, bodies included."""
    if not stubbed:
        return []

    def walk(nodes: Any) -> list[Any]:
        out = []
        for n in nodes:
            out.append(n)
            out.extend(walk(getattr(n, "body", None) or []))
        return out

    return [n.id for n in walk(pkg.graph.nodes)
            if any(t in stubbed for t in (n.tools or []))]


async def _verify_in_sandbox(
    provider: Any,
    *,
    intent: str,
    plan: AcceptancePlan,
    pkg: Any,
    wf_runner: Any,
    all_node_ids: list[str],
    executed: set[str],
    runs: int,
    fixtures_created: list[str],
    paired_inputs: dict[str, str],
    on_node_event: Any = None,
) -> VerificationReport:
    """The run/judge half, once the world the workflow needs actually exists."""
    import asyncio

    def _run(inputs: dict[str, Any]):
        # The same callback a registered run uses, so the canvas lights up
        # while a test runs. Verification executes the real graph through the
        # real runner and simply never asked for the events — so a build spent
        # its longest phase showing nothing but 'running verification'.
        return wf_runner.run(pkg, dict(inputs), on_node_event=on_node_event)

    try:
        runs += 1
        result = await asyncio.to_thread(_run, plan.test_inputs)
    except Exception as e:  # noqa: BLE001 - a crashing run is a failed verification
        return VerificationReport(
            passed=False, run_ok=False, graph_runs=runs,
            fixtures_created=fixtures_created, paired_inputs=paired_inputs,
            diagnosis=f"The workflow could not run at all: {e}",
            suggestions="Fix the workflow inputs/structure so a test run can start.",
        )

    node_summaries = _summarise_nodes(result)
    executed |= {nid for nid, nr in result.nodes.items()
                 if not nr.skipped and nr.error is None}

    # Phase 5b: which steps ran against a stub rather than the real thing.
    stubbed = set(getattr(wf_runner, "stubbed_tools", set()) or set())
    not_exercised = _stubbed_nodes(pkg, stubbed)

    if result.errors:
        # Deterministic diagnosis — no judge call on a crashed run.
        failed = "; ".join(f"{nid}: {err[:200]}" for nid, err in result.errors.items())
        return VerificationReport(
            passed=False, run_ok=False, graph_runs=runs,
            fixtures_created=fixtures_created, paired_inputs=paired_inputs,
            not_exercised=not_exercised, stubbed_tools=sorted(stubbed),
            node_summaries=node_summaries,
            diagnosis=f"Run failed at node(s): {failed}",
            suggestions=(
                "Fix the failing node(s): check tool assignments, required inputs, "
                "and depends_on wiring; then test again."
            ),
        )

    # Branch coverage: run each extra case; it must run cleanly, and together the
    # cases should light up every branch. Extra cases are not judged (cost) — the
    # main case carries the criteria. Each extra case re-runs the WHOLE graph, so
    # to keep cost down we (a) stop as soon as every node has been exercised and
    # (b) skip cases whose inputs duplicate an already-run set.
    case_results: list[dict[str, Any]] = []
    cases_ok = True
    seen_inputs: list[dict[str, Any]] = [dict(plan.test_inputs)]
    for case in plan.extra_cases:
        if all(nid in executed for nid in all_node_ids):
            break  # full node coverage reached — further re-runs are pure cost
        case_inputs = case.get("test_inputs", {})
        if any(case_inputs == prev for prev in seen_inputs):
            continue  # identical inputs already exercised
        seen_inputs.append(dict(case_inputs))
        label = case.get("label", "?")
        try:
            runs += 1
            case_run = await asyncio.to_thread(_run, case_inputs)
            errs = case_run.errors or {}
            executed |= {nid for nid, nr in case_run.nodes.items()
                         if not nr.skipped and nr.error is None}
            if errs:
                cases_ok = False
                case_results.append({
                    "label": label, "ok": False,
                    "error": "; ".join(f"{k}: {v[:150]}" for k, v in errs.items()),
                })
            else:
                case_results.append({"label": label, "ok": True, "error": None})
        except Exception as e:  # noqa: BLE001
            cases_ok = False
            case_results.append({"label": label, "ok": False, "error": str(e)[:300]})

    coverage_gaps = [nid for nid in all_node_ids if nid not in executed]

    final_outputs = {k: v for k, v in (result.final or {}).items()}
    verdicts, diagnosis, suggestions = await _judge(
        provider, intent, plan, final_outputs, node_summaries, not_exercised
    )
    passed = all(v["passed"] for v in verdicts) and cases_ok
    if not cases_ok and not diagnosis:
        diagnosis = "One or more branch test cases failed to run cleanly."
        suggestions = "Fix the failing branch (see the branch case errors above)."
    return VerificationReport(
        passed=passed, run_ok=True, graph_runs=runs,
        fixtures_created=fixtures_created, paired_inputs=paired_inputs,
        not_exercised=not_exercised, stubbed_tools=sorted(stubbed),
        verdicts=verdicts, node_summaries=node_summaries,
        diagnosis="" if passed else diagnosis,
        suggestions="" if passed else suggestions,
        case_results=case_results,
        coverage_gaps=coverage_gaps,
    )

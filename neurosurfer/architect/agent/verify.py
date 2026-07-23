"""Closed-loop verification of staged workflows (Phase 5).

The engine that lets the Architect prove its own work before registering:

1. :func:`derive_acceptance` — one LLM call turns the user's intent + the staged
   graph's declared inputs into an :class:`AcceptancePlan`: 2–6 explicit success
   criteria and concrete test inputs.
2. :func:`verify_workflow` — actually RUNS the staged package on those inputs
   (in a worker thread; the runner is synchronous), then:
   - a failed run yields a deterministic diagnosis from the node errors (no judge
     call — criteria can't pass on a crashed run);
   - a clean run is scored by an LLM judge, per criterion, fail-closed (a
     criterion the judge doesn't rule on counts as failed), with a diagnosis +
     design suggestions for anything failing.

The report is rendered for the agent's `test_workflow` tool; the agent applies
fixes with its normal graph-editing tools — design revision, not field patching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from neurosurfer.llm.types import GenerationConfig, Message

__all__ = [
    "AcceptanceCriterion",
    "AcceptancePlan",
    "VerificationReport",
    "derive_acceptance",
    "verify_workflow",
]

_MAX_OUTPUT_CHARS = 1500   # per-node output shown to the judge
_MAX_CRITERIA = 6


class AcceptanceCriterion(BaseModel):
    id: str = Field(description="Short snake_case id, e.g. 'summary_is_3_sentences'.")
    description: str = Field(description="One testable statement about the workflow's output.")


class AcceptancePlan(BaseModel):
    criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    test_inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Concrete values for every required declared graph input.",
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

    def render(self) -> str:
        lines = [f"VERIFICATION {'PASSED' if self.passed else 'FAILED'} "
                 f"(run {'ok' if self.run_ok else 'errored'})"]
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


# ── tolerant JSON extraction ────────────────────────────────────────────────────

def _balanced_objects(s: str) -> list[str]:
    """Every top-level brace-balanced ``{...}`` substring (string-literal aware)."""
    objs: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                objs.append(s[start:i + 1])
                start = -1
    return objs


def _parse_json(text: str, *, want: str | None = None) -> Any:
    """Tolerant object extraction. Small models wrap JSON in a "Thinking…" preamble
    that itself contains ``{...}`` example fragments, so we scan ALL balanced
    objects and return the LAST one that parses (optionally requiring a key), which
    is the actual answer after any reasoning."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    best: Any = None
    for chunk in _balanced_objects(text):
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if want is None or want in parsed:
            best = parsed  # keep the last match
    return best


# ── acceptance derivation ───────────────────────────────────────────────────────

_DERIVE_SYSTEM = """\
You derive an acceptance test plan for a workflow that is about to be verified.
Output STRICT JSON only, no prose, no code fences:
  {"criteria": [{"id": "<snake_case>", "description": "<testable statement>"}],
   "test_inputs": {<one concrete realistic value for EVERY required input>},
   "extra_cases": [{"label": "<branch name>", "test_inputs": {...}}]}
Rules:
- 2 to 6 criteria. Each must be checkable by reading the workflow's outputs alone.
- Criteria must reflect what the USER asked for — not generic quality platitudes.
- test_inputs must contain realistic sample data (e.g. an actual short article
  text, not a placeholder like 'test' or 'lorem ipsum').
- extra_cases: ONLY if the workflow branches (router nodes, `when:` guards): add
  one case per distinct branch, with inputs crafted to trigger that branch (e.g.
  an obviously-urgent ticket vs. an obviously-trivial one). Max 3. Omit (empty
  list) for linear workflows.
"""


async def derive_acceptance(
    provider: Any,
    intent: str,
    graph_yaml: str,
    *,
    declared_inputs: list[dict[str, Any]] | None = None,
) -> AcceptancePlan:
    """One LLM call → criteria + concrete test inputs for the staged workflow.

    *declared_inputs* (the graph's ``inputs``) is used to backfill placeholder
    values for any required input the model didn't supply, so verification can
    always attempt a run rather than failing on a missing input.
    """
    prompt = (
        f"User's request for the workflow:\n{intent}\n\n"
        f"The staged workflow (its declared `inputs` need test values):\n{graph_yaml}\n\n"
        "Produce the JSON acceptance plan."
    )
    response = await provider.complete(
        messages=[Message.user_text(prompt)],
        system=_DERIVE_SYSTEM,
        tools=[],
        config=GenerationConfig(max_tokens=1200, temperature=0.2, stream=False),
    )
    data = _parse_json(response.text(), want="criteria") or {}
    criteria = []
    for c in (data.get("criteria") or [])[:_MAX_CRITERIA]:
        if isinstance(c, dict) and c.get("description"):
            criteria.append(AcceptanceCriterion(
                id=str(c.get("id") or f"criterion_{len(criteria) + 1}"),
                description=str(c["description"]),
            ))
    if not criteria:
        # Fail-safe: a single criterion straight from the intent.
        criteria = [AcceptanceCriterion(
            id="fulfils_intent",
            description=f"The output fulfils the request: {intent[:300]}",
        )]
    test_inputs = dict(data.get("test_inputs")) if isinstance(data.get("test_inputs"), dict) else {}
    _backfill_inputs(test_inputs, declared_inputs)

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
                          extra_cases=extra_cases)


def _backfill_inputs(
    test_inputs: dict[str, Any], declared: list[dict[str, Any]] | None
) -> None:
    """Ensure every required declared input has *some* value so a run can start."""
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
"""


async def _judge(
    provider: Any,
    intent: str,
    plan: AcceptancePlan,
    final_outputs: dict[str, Any],
    node_summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str]:
    criteria_text = "\n".join(f"- [{c.id}] {c.description}" for c in plan.criteria)
    outputs_text = json.dumps(final_outputs, ensure_ascii=False, default=str)[:4000]
    nodes_text = json.dumps(node_summaries, ensure_ascii=False, default=str)[:3000]
    prompt = (
        f"User intent:\n{intent}\n\n"
        f"Acceptance criteria:\n{criteria_text}\n\n"
        f"Test inputs used: {json.dumps(plan.test_inputs, ensure_ascii=False)[:800]}\n\n"
        f"Workflow FINAL outputs:\n{outputs_text}\n\n"
        f"Per-node results:\n{nodes_text}\n\n"
        "Rule on every criterion."
    )
    response = await provider.complete(
        messages=[Message.user_text(prompt)],
        system=_JUDGE_SYSTEM,
        tools=[],
        config=GenerationConfig(max_tokens=1200, temperature=0.0, stream=False),
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
) -> VerificationReport:
    """Run the staged package on the plan's inputs, then judge (clean runs only)."""
    import asyncio
    from pathlib import Path

    from neurosurfer.graph.workflow.package import load_package
    from neurosurfer.graph.workflow.runner import WorkflowRunner

    pkg = load_package(Path(package_dir))
    wf_runner = runner or WorkflowRunner(provider)
    all_node_ids = [n.id for n in pkg.graph.nodes]
    executed: set[str] = set()

    def _run(inputs: dict[str, Any]):
        return wf_runner.run(pkg, dict(inputs))

    try:
        result = await asyncio.to_thread(_run, plan.test_inputs)
    except Exception as e:  # noqa: BLE001 - a crashing run is a failed verification
        return VerificationReport(
            passed=False, run_ok=False,
            diagnosis=f"The workflow could not run at all: {e}",
            suggestions="Fix the workflow inputs/structure so a test run can start.",
        )

    node_summaries = _summarise_nodes(result)
    executed |= {nid for nid, nr in result.nodes.items()
                 if not nr.skipped and nr.error is None}

    if result.errors:
        # Deterministic diagnosis — no judge call on a crashed run.
        failed = "; ".join(f"{nid}: {err[:200]}" for nid, err in result.errors.items())
        return VerificationReport(
            passed=False, run_ok=False,
            node_summaries=node_summaries,
            diagnosis=f"Run failed at node(s): {failed}",
            suggestions=(
                "Fix the failing node(s): check tool assignments, required inputs, "
                "and depends_on wiring; then test again."
            ),
        )

    # Branch coverage: run each extra case; it must run cleanly, and together the
    # cases should light up every branch. Extra cases are not judged (cost) — the
    # main case carries the criteria.
    case_results: list[dict[str, Any]] = []
    cases_ok = True
    for case in plan.extra_cases:
        label = case.get("label", "?")
        try:
            case_run = await asyncio.to_thread(_run, case.get("test_inputs", {}))
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
        provider, intent, plan, final_outputs, node_summaries
    )
    passed = all(v["passed"] for v in verdicts) and cases_ok
    if not cases_ok and not diagnosis:
        diagnosis = "One or more branch test cases failed to run cleanly."
        suggestions = "Fix the failing branch (see the branch case errors above)."
    return VerificationReport(
        passed=passed, run_ok=True,
        verdicts=verdicts, node_summaries=node_summaries,
        diagnosis="" if passed else diagnosis,
        suggestions="" if passed else suggestions,
        case_results=case_results,
        coverage_gaps=coverage_gaps,
    )

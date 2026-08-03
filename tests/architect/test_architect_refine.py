"""V3 Phase 5d — runtime repair for a workflow that registered and then broke.

The arm for the failure nobody could have tested at build time: a renamed file, an
API that changed shape, a credential that expired. Two properties matter more than
the healing itself —

1. It **proposes**. The legacy E7 wrote overrides straight into a registered
   package and re-ran, editing production with nobody's approval.
2. It knows what config cannot fix. A missing credential is not a prompt bug, and
   rewriting the node would bury a missing secret under a plausible edit.

`test_workflow_refine.py` covers the original E7 healing loop; this file covers
what 5d added to it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from neurosurfer.architect.refine import WorkflowRefiner, _external_failure
from neurosurfer.graph.workflow.package import load_package

from ..fakes import ScriptedProvider

FN = "tests.architect.test_architect_refine"


def _boom(**kwargs):
    raise RuntimeError("no such file: notes.txt")


def _fine(**kwargs):
    return "worked"


def _pkg(tmp_path: Path, nodes: list[dict], name: str = "wf") -> Path:
    pkg_dir = tmp_path / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "workflow.yaml").write_text(yaml.dump(
        {"name": name, "version": "1.0.0", "entrypoint": "graph.yaml"}))
    (pkg_dir / "graph.yaml").write_text(yaml.dump({
        "name": name, "nodes": nodes, "outputs": [nodes[-1]["id"]],
    }))
    return pkg_dir


def _prose_pkg(tmp_path: Path, name: str = "wf") -> Path:
    """A `base` node — the kind for which a prose patch is a real repair.

    Deliberately not a `function` node: since the inert-field check, patching
    `goal` on one of those is dropped as a no-op, which is correct and makes it a
    bad fixture for testing that patches land.
    """
    return _pkg(tmp_path, [{"id": "read", "kind": "base",
                            "goal": "read notes.txt and summarise it"}], name)


class _Registry:
    def __init__(self, pkg_dir: Path):
        self._pkg_dir = pkg_dir

    def get(self, _name):
        return load_package(self._pkg_dir)


class _FailingRunner:
    """Reports *error* on the named node; succeeds from the *succeed_on* run.

    A stub rather than the real runner because these tests are about the doctor
    and the proposal, not about executing a graph — and a `base` node would need a
    provider turn of its own to fail.
    """

    def __init__(self, node_id="read", error="no such file: notes.txt",
                 succeed_on: int | None = None):
        self.node_id = node_id
        self.error = error
        self.succeed_on = succeed_on
        self.runs = 0

    def run(self, pkg, inputs):
        self.runs += 1
        ok = self.succeed_on is not None and self.runs >= self.succeed_on
        err = None if ok else self.error

        class _NR:
            def __init__(self, e):
                self.error = e
                self.skipped = False
                self.raw_output = None if e else "fine"

        return type("R", (), {
            "nodes": {self.node_id: _NR(err)},
            "errors": {} if ok else {self.node_id: err},
            "final": {},
        })()


def _doctor(diagnosis="the goal names the wrong file", patch=None) -> str:
    return json.dumps({"diagnosis": diagnosis,
                       "patch": patch if patch is not None else {"goal": "read report.txt"}})


# ── it proposes; it does not touch the package ──────────────────────────────────

async def test_diagnose_reports_without_writing_anything(tmp_path):
    pkg_dir = _prose_pkg(tmp_path)
    provider = ScriptedProvider(turns=[(_doctor(), [])])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    proposal = await refiner.diagnose("wf", {})

    assert proposal.ran and not proposal.ok
    assert [d.node_id for d in proposal.failed_nodes] == ["read"]
    d = proposal.failed_nodes[0]
    assert "no such file" in d.error
    assert d.diagnosis == "the goal names the wrong file"
    assert d.patch == {"goal": "read report.txt"}
    assert proposal.actionable

    # Nothing was written: the override layer does not exist.
    assert not (pkg_dir / "agents").exists()


async def test_a_clean_run_has_nothing_to_repair(tmp_path):
    pkg_dir = _pkg(tmp_path, [{"id": "ok", "kind": "function", "callable": f"{FN}._fine"}])
    provider = ScriptedProvider(turns=[])   # the doctor must not be consulted
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir))

    proposal = await refiner.diagnose("wf", {})
    assert proposal.ok and not proposal.failed_nodes
    assert provider.calls == 0
    assert "ran cleanly" in proposal.render()


async def test_apply_is_a_separate_explicit_step(tmp_path):
    pkg_dir = _prose_pkg(tmp_path)
    provider = ScriptedProvider(turns=[(_doctor(), [])])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    proposal = await refiner.diagnose("wf", {})
    applied = refiner.apply(proposal)

    assert applied == ["read"]
    override = yaml.safe_load((pkg_dir / "agents" / "read.yaml").read_text())
    assert override == {"goal": "read report.txt", "id": "read"}


# ── what config cannot fix ──────────────────────────────────────────────────────

@pytest.mark.parametrize("error", [
    "HTTP 401 Unauthorized",
    "missing API key",
    "Connection refused",
    "the request timed out",
    "429 rate limit exceeded",
    "[NOT EXERCISED] `gmail_read` was not called",
])
def test_access_and_connectivity_errors_are_not_design_problems(error):
    node = type("N", (), {"id": "n", "tools": [], "kind": "base"})()
    assert _external_failure(node, error)


@pytest.mark.parametrize("error", [
    "no such file: notes.txt",
    "KeyError: 'summary'",
    "output did not match the schema",
])
def test_ordinary_failures_stay_open_to_a_patch(error):
    node = type("N", (), {"id": "n", "tools": [], "kind": "base"})()
    assert _external_failure(node, error) == ""


async def test_an_external_failure_gets_advice_and_no_model_call(tmp_path):
    """Asking for a patch here invites one that buries a missing secret."""
    pkg_dir = _pkg(tmp_path, [{"id": "call", "kind": "function", "callable": f"{FN}._unauthorized"}])
    provider = ScriptedProvider(turns=[])   # must not be consulted
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir))

    proposal = await refiner.diagnose("wf", {})

    assert provider.calls == 0
    d = proposal.failed_nodes[0]
    assert d.external_reason and not d.patch
    assert not proposal.actionable
    assert proposal.blocked_on_external == [d]
    assert "no graph change will fix them" in proposal.render()


def _unauthorized(**kwargs):
    raise RuntimeError("HTTP 401 Unauthorized — check your API key")


async def test_healing_stops_rather_than_retrying_an_external_failure(tmp_path):
    """Re-running fails identically, and there is no patch to try."""
    pkg_dir = _pkg(tmp_path, [{"id": "call", "kind": "function",
                               "callable": f"{FN}._unauthorized"}])
    provider = ScriptedProvider(turns=[])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir), max_rounds=3)

    result = await refiner.refine("wf", {})
    assert not result.ok
    assert result.rounds == 1           # did not burn all three
    assert result.patched_nodes == []
    assert "no graph change will fix them" in result.message


# ── a patch may not invent a tool ───────────────────────────────────────────────

async def test_a_tool_that_does_not_exist_is_dropped_from_the_patch(tmp_path):
    """'Use only tools that exist' is an instruction, so it is also checked."""
    pkg_dir = _prose_pkg(tmp_path)
    provider = ScriptedProvider(turns=[
        (_doctor(patch={"tools": ["read_file", "magically_fix_everything"]}), []),
    ])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    d = (await refiner.diagnose("wf", {})).failed_nodes[0]
    assert d.patch["tools"] == ["read_file"]
    assert d.rejected_tools == ["magically_fix_everything"]
    assert "no such tool" in d.render()


async def test_a_patch_of_only_invented_tools_leaves_no_tools_key(tmp_path):
    pkg_dir = _prose_pkg(tmp_path)
    provider = ScriptedProvider(turns=[(_doctor(patch={"tools": ["nope_not_real"]}), [])])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    d = (await refiner.diagnose("wf", {})).failed_nodes[0]
    assert "tools" not in d.patch
    assert d.rejected_tools == ["nope_not_real"]
    assert not d.actionable


async def test_a_patch_to_a_field_the_kind_never_reads_is_dropped(tmp_path):
    """Straight from a live run: `goal` proposed on a `kind: tool` node.

    A tool node invokes its tool with `tool_args` and makes no LLM call, so
    `goal` is never read. Applying it would write a no-op override into a
    registered package and report a repair that changes nothing.
    """
    pkg_dir = _pkg(tmp_path, [
        {"id": "read", "kind": "tool", "tools": ["read_file"],
         "tool_args": {"path": "nope.txt"}},
    ])
    provider = ScriptedProvider(turns=[
        (_doctor(patch={"goal": "Load the contents of a valid file."}), []),
    ])

    class _Runner:
        def run(self, pkg, inputs):
            class _NR:
                error = "File not found: nope.txt"
                skipped = False
            return type("R", (), {"nodes": {"read": _NR()}, "errors": {"read": "x"},
                                  "final": {}})()

    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir), runner=_Runner())
    d = (await refiner.diagnose("wf", {})).failed_nodes[0]

    assert d.inert_fields == ["goal"]
    assert d.patch == {}
    assert not d.actionable
    assert d.answered                      # the diagnosis still stands
    assert "never reads these" in d.render()


def test_inert_fields_per_kind():
    from neurosurfer.architect.refine import inert_patch_fields

    patch = {"goal": "x", "tools": ["read_file"], "depends_on": ["a"]}
    # A tool node reads its tools; it does not read prose.
    assert inert_patch_fields("tool", patch) == ["goal"]
    # A function node calls a callable — neither prose nor tools reach it.
    assert inert_patch_fields("function", patch) == ["goal", "tools"]
    # base/react read everything on the patchable list.
    assert inert_patch_fields("base", patch) == []
    assert inert_patch_fields("react", patch) == []


async def test_unpatchable_fields_are_filtered_out(tmp_path):
    pkg_dir = _prose_pkg(tmp_path)
    provider = ScriptedProvider(turns=[
        (_doctor(patch={"goal": "fixed", "kind": "react", "callable": "os.system"}), []),
    ])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    d = (await refiner.diagnose("wf", {})).failed_nodes[0]
    assert d.patch == {"goal": "fixed"}


async def test_a_patch_writing_instructions_clears_the_fields_it_supersedes(tmp_path):
    """A node states its job once.

    `instructions` supersedes purpose/goal/expected_result and the engine prefers
    it outright, so a patch that adds `instructions` to a node that still carries
    a `goal` would leave YAML showing two versions of the job and a run silently
    using one.
    """
    pkg_dir = _prose_pkg(tmp_path)
    provider = ScriptedProvider(turns=[
        (_doctor(patch={"instructions": "Read notes.txt and summarise it."}), []),
    ])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    d = (await refiner.diagnose("wf", {})).failed_nodes[0]
    assert d.patch["instructions"] == "Read notes.txt and summarise it."
    for field in ("purpose", "goal", "expected_result"):
        assert d.patch[field] is None, field


async def test_a_prose_patch_to_a_one_field_node_clears_instructions(tmp_path):
    """The same rule in the other direction.

    Patching `goal` on a node that already has `instructions` would be a repair
    the engine ignores — it prefers `instructions` — reported as a success.
    """
    pkg_dir = _pkg(tmp_path, [{"id": "read", "kind": "base",
                               "instructions": "read notes.txt and summarise it"}])
    provider = ScriptedProvider(turns=[(_doctor(patch={"goal": "fixed"}), [])])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    d = (await refiner.diagnose("wf", {})).failed_nodes[0]
    assert d.patch["goal"] == "fixed"
    assert d.patch["instructions"] is None


async def test_an_unreadable_doctor_answer_is_reported_not_guessed(tmp_path):
    pkg_dir = _prose_pkg(tmp_path)
    provider = ScriptedProvider(turns=[("I think perhaps the node is sad.", [])])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    d = (await refiner.diagnose("wf", {})).failed_nodes[0]
    assert not d.patch
    assert "could not be read" in d.diagnosis


# ── "the design is fine, the input wasn't" is an answer, not a shrug ────────────

async def test_a_diagnosis_with_no_patch_is_a_verdict_not_a_failure(tmp_path):
    """Straight from the first live run of this phase.

    Asked about `File not found: quarterly_report.txt`, gpt-4o-mini correctly
    returned a diagnosis and an empty patch — the caller passed a path that does
    not exist, and no edit to the node would change that. Rendering that as "no
    patch could be proposed" made a correct diagnosis read as the refiner failing.
    """
    pkg_dir = _prose_pkg(tmp_path)
    provider = ScriptedProvider(turns=[
        (_doctor(diagnosis="The specified file 'notes.txt' is missing.", patch={}), []),
    ])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    proposal = await refiner.diagnose("wf", {})
    d = proposal.failed_nodes[0]

    assert not d.actionable          # nothing to apply
    assert d.answered                # but it was answered
    assert "no change proposed" in d.render()
    assert "the workflow's design is not the fault" in d.render()
    body = proposal.render()
    assert "the workflow is fine; what it was given is not" in body
    assert "No diagnosis could be produced" not in body


async def test_an_unanswered_failure_says_so_distinctly(tmp_path):
    pkg_dir = _prose_pkg(tmp_path)
    provider = ScriptedProvider(turns=[("nonsense", [])])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    proposal = await refiner.diagnose("wf", {})
    # The fail-safe diagnosis counts as answered; what must not happen is the two
    # cases rendering identically.
    assert "No diagnosis could be read" in proposal.failed_nodes[0].render() or \
        "could not be read" in proposal.failed_nodes[0].diagnosis


# ── the healing loop still heals ────────────────────────────────────────────────

async def test_refine_patches_revalidates_and_reruns(tmp_path):
    """The legacy loop, intact: patch the node, re-validate, run again, succeed."""
    pkg_dir = _pkg(tmp_path, [
        {"id": "step", "kind": "base", "goal": "do the thing"},
    ])
    # A base node with no provider errors on run 1; the patch makes the graph
    # valid-but-still-failing, so this asserts the mechanics, not a real cure.
    provider = ScriptedProvider(turns=[
        (_doctor(patch={"goal": "do the thing correctly"}), []),
        ("", []), ("", []),
    ])

    class _Runner:
        """Fails once, then succeeds — stands in for the patch having worked."""

        def __init__(self):
            self.runs = 0

        def run(self, pkg, inputs):
            self.runs += 1
            failed = self.runs == 1

            class _NR:
                def __init__(self, err):
                    self.error = err
                    self.skipped = False
                    self.raw_output = None if err else "ok"

            return type("R", (), {
                "nodes": {"step": _NR("KeyError: 'thing'" if failed else None)},
                "errors": {"step": "KeyError"} if failed else {},
                "final": {},
            })()

    runner = _Runner()
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir), runner=runner)

    result = await refiner.refine("wf", {})
    assert result.ok, result.message
    assert result.patched_nodes == ["step"]
    assert runner.runs == 2
    # The patch really landed in the override layer.
    assert yaml.safe_load(
        (pkg_dir / "agents" / "step.yaml").read_text())["goal"] == "do the thing correctly"


async def test_refine_apply_false_diagnoses_without_writing(tmp_path):
    pkg_dir = _prose_pkg(tmp_path)
    provider = ScriptedProvider(turns=[(_doctor(), [])])
    refiner = WorkflowRefiner(provider, registry=_Registry(pkg_dir),
                              runner=_FailingRunner())

    result = await refiner.refine("wf", {}, apply=False)
    assert not result.ok and result.patched_nodes == []
    assert not (pkg_dir / "agents").exists()
    assert result.proposal is not None and result.proposal.actionable


async def test_a_workflow_that_cannot_start_is_a_finding_not_a_crash(tmp_path):
    pkg_dir = _pkg(tmp_path, [{"id": "n", "kind": "function", "callable": f"{FN}._fine"}])

    class _Exploding:
        def run(self, pkg, inputs):
            raise ValueError("this workflow references tool(s) that are not registered")

    refiner = WorkflowRefiner(
        ScriptedProvider(turns=[]), registry=_Registry(pkg_dir), runner=_Exploding())

    proposal = await refiner.diagnose("wf", {})
    assert not proposal.ran and not proposal.ok
    assert proposal.failed_nodes[0].node_id == "(workflow)"
    assert "could not start" in proposal.failed_nodes[0].diagnosis

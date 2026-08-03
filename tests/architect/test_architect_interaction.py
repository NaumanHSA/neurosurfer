"""S5 depth: a build that stops and asks, and one that can be called off.

The interesting property is a rendezvous across threads — the build parks, an
HTTP request answers, the build resumes — so these tests drive both sides for
real rather than mocking the gate. A fake agent stands in for the LLM.

The defaults matter as much as the happy path. An unattended build must not
install generated code because nobody was watching, and a build parked on a
question must still die promptly when cancelled.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from neurosurfer.app.server.architect_builds.interaction import (
    Cancelled,
    InteractionGate,
)
from neurosurfer.app.server.architect_builds.manager import ArchitectManager
from neurosurfer.app.server.gateway import NeurosurferServer
from neurosurfer.graph.workflow.registry import WorkflowRegistry


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.01):
    """Poll until *predicate* is truthy. Returns its value, or None on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


# ── the gate itself ─────────────────────────────────────────────────────────

def test_a_waiting_thread_resumes_with_the_answer():
    gate = InteractionGate()
    result: list = []

    def build_thread():
        result.append(gate.ask(kind="question", prompt="which?", on_timeout="default"))

    t = threading.Thread(target=build_thread)
    t.start()
    pending = _wait_for(lambda: gate.pending)
    assert pending is not None, "the thread should have parked"

    assert gate.respond(pending.id, "markdown") is True
    t.join(timeout=5)
    assert result == ["markdown"]
    assert gate.pending is None


def test_answering_the_wrong_interaction_is_refused():
    """A stale tab must not feed its answer into whatever is being asked now."""
    gate = InteractionGate()
    threading.Thread(
        target=lambda: gate.ask(kind="question", prompt="q", on_timeout="d"), daemon=True
    ).start()
    pending = _wait_for(lambda: gate.pending)

    assert gate.respond("some-other-id", "wrong") is False
    assert gate.pending is pending, "still waiting for its own answer"


def test_nobody_answering_applies_the_stated_default():
    gate = InteractionGate()
    value = gate.ask(kind="question", prompt="q", on_timeout="no preference", timeout_s=0.05)
    assert value == "no preference"


def test_cancelling_wakes_a_parked_thread():
    """Otherwise a build waiting on an abandoned question would hang until its
    timeout rather than dying when asked to."""
    gate = InteractionGate()
    error: list = []

    def build_thread():
        try:
            gate.ask(kind="question", prompt="q", on_timeout="d", timeout_s=30)
        except Cancelled as e:
            error.append(e)

    t = threading.Thread(target=build_thread)
    t.start()
    _wait_for(lambda: gate.pending)

    gate.cancel()
    t.join(timeout=5)
    assert not t.is_alive(), "cancel must not leave the thread parked"
    assert len(error) == 1


def test_cancellation_survives_a_broad_except_clause():
    """Regression: `Cancelled` used to subclass RuntimeError, so the agent's tool
    runner — which catches Exception to turn tool failures into results the model
    can react to — swallowed it. The build then carried on and finished *blocked*
    instead of *cancelled*. Only a BaseException gets through untouched."""
    gate = InteractionGate()
    gate.cancel()

    with pytest.raises(Cancelled):
        try:
            gate.raise_if_cancelled()
        except Exception:  # noqa: BLE001 - exactly what the tool runner does
            pytest.fail("a cancel must not be catchable as an ordinary error")


# ── a build that asks ───────────────────────────────────────────────────────

class _AskingSession:
    name = "asked_wf"

    def graph_dict(self) -> dict:
        return {"name": self.name, "nodes": [], "inputs": [], "outputs": []}


class _AskingAgent:
    """Stands in for the LLM: reports the intent it was handed, then finishes."""

    def __init__(self, notify, registry, sink: dict) -> None:
        self._notify = notify
        self._registry = registry
        self._sink = sink
        self.session = _AskingSession()

    async def build(self, intent: str, answers=None, refines=None) -> str:
        self._sink["intent"] = intent
        self._sink["answers"] = answers
        self._sink["refines"] = refines
        self._notify("building")
        return "/tmp/asked_wf"


@pytest.fixture
def api(tmp_path):
    registry = WorkflowRegistry(workflows_dir=tmp_path / "registry")
    sink: dict = {}
    manager = ArchitectManager(
        _DummyProvider(),
        registry=registry,
        staging_root=tmp_path / "staging",
        agent_factory=lambda notify, _v: _AskingAgent(notify, registry, sink),
    )
    server = NeurosurferServer(app_name="test", api_keys=None)
    server.architect_manager = manager
    return TestClient(server.create_app()), manager, sink


class _DummyProvider:
    model = "dummy"
    capabilities = type("Caps", (), {"context_window": 8192, "max_output_tokens": 512})()


def test_a_build_can_be_cancelled_while_running(api):
    client, manager, _ = api
    build = client.post("/v1/architect/builds", json={"intent": "something"}).json()

    # The fake finishes immediately, so cancel whichever side wins the race —
    # what must hold is that cancel never reports success on a finished build.
    res = client.post(f"/v1/architect/builds/{build['id']}/cancel")
    assert res.status_code in (200, 409)
    final = _wait_for(
        lambda: client.get(f"/v1/architect/builds/{build['id']}").json()["status"] != "running"
    )
    assert final is not None
    status = client.get(f"/v1/architect/builds/{build['id']}").json()["status"]
    assert status in {"cancelled", "succeeded"}


def test_cancelling_a_finished_build_is_refused(api):
    client, _, _ = api
    build = client.post("/v1/architect/builds", json={"intent": "x"}).json()
    _wait_for(
        lambda: client.get(f"/v1/architect/builds/{build['id']}").json()["status"] != "running"
    )
    res = client.post(f"/v1/architect/builds/{build['id']}/cancel")
    assert res.status_code == 409


def test_cancelling_an_unknown_build_is_404(api):
    client, _, _ = api
    assert client.post("/v1/architect/builds/nope/cancel").status_code == 404


def test_responding_needs_an_interaction_that_is_waiting(api):
    client, _, _ = api
    build = client.post("/v1/architect/builds", json={"intent": "x"}).json()
    res = client.post(
        f"/v1/architect/builds/{build['id']}/respond",
        json={"interaction_id": "whatever", "value": "yes"},
    )
    assert res.status_code == 409, "nothing is being asked"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"value": "yes"}, id="no-interaction-id"),
        pytest.param({"interaction_id": "abc"}, id="no-value"),
    ],
)
def test_malformed_responses_are_rejected(api, body):
    client, _, _ = api
    build = client.post("/v1/architect/builds", json={"intent": "x"}).json()
    res = client.post(f"/v1/architect/builds/{build['id']}/respond", json=body)
    assert res.status_code == 422


# ── the whole rendezvous, through the HTTP surface ──────────────────────────

class _ParkingAgent:
    """An agent that stops mid-build and waits for a person, like tool approval."""

    def __init__(self, notify, gate, sink: dict) -> None:
        self._notify = notify
        self._gate = gate
        self._sink = sink
        self.session = _AskingSession()

    async def build(self, intent: str, answers=None) -> str:
        import asyncio

        self._notify("authoring a tool")
        self._sink["answer"] = await asyncio.to_thread(
            self._gate.ask,
            kind="tool_approval",
            prompt="Register the authored tool 'fetch_page'?",
            choices=["approve", "reject"],
            detail={"name": "fetch_page", "source": "def run(): ..."},
            on_timeout="reject",
            timeout_s=10,
        )
        self._notify("done")
        return "/tmp/parked_wf"


@pytest.fixture
def parking_api(tmp_path):
    registry = WorkflowRegistry(workflows_dir=tmp_path / "registry")
    sink: dict = {}
    manager = ArchitectManager(
        _DummyProvider(),
        registry=registry,
        staging_root=tmp_path / "staging",
        agent_factory=lambda notify, _v, gate: _ParkingAgent(notify, gate, sink),
    )
    server = NeurosurferServer(app_name="test", api_keys=None)
    server.architect_manager = manager
    return TestClient(server.create_app()), sink


def test_a_build_parks_is_answered_over_http_and_resumes(parking_api):
    """The whole point of S5 depth: the build stops, a person decides, it goes on."""
    client, sink = parking_api
    build = client.post(
        "/v1/architect/builds", json={"intent": "scrape a page", "approve_tools": True}
    ).json()
    url = f"/v1/architect/builds/{build['id']}"

    pending = _wait_for(lambda: client.get(url).json().get("pending"))
    assert pending is not None, "the build should be waiting on a person"
    assert pending["kind"] == "tool_approval"
    assert pending["detail"]["source"], "a reviewer needs to see the code"

    res = client.post(
        f"{url}/respond", json={"interaction_id": pending["id"], "value": "approve"}
    )
    assert res.status_code == 200

    assert _wait_for(lambda: sink.get("answer")) == "approve"
    assert _wait_for(lambda: client.get(url).json()["status"] == "succeeded")
    assert client.get(url).json()["pending"] is None, "must clear once answered"


def test_the_pending_interaction_reaches_the_event_log(parking_api):
    """A studio that connects mid-build reads the log, not just the record."""
    client, _ = parking_api
    build = client.post(
        "/v1/architect/builds", json={"intent": "x", "approve_tools": True}
    ).json()
    url = f"/v1/architect/builds/{build['id']}"
    pending = _wait_for(lambda: client.get(url).json().get("pending"))

    events = client.get(f"{url}?events=true").json()["events"]
    assert any(e["type"] == "pending" for e in events)

    client.post(f"{url}/respond", json={"interaction_id": pending["id"], "value": "reject"})
    _wait_for(lambda: client.get(url).json()["status"] != "running")

    events = client.get(f"{url}?events=true").json()["events"]
    resolved = [e for e in events if e["type"] == "resolved"]
    assert resolved and resolved[0]["value"] == "reject"
    assert resolved[0]["answered"] is True


def test_cancelling_a_build_parked_on_a_person_ends_it(parking_api):
    """A build waiting on an abandoned browser tab must still be killable."""
    client, _ = parking_api
    build = client.post(
        "/v1/architect/builds", json={"intent": "x", "approve_tools": True}
    ).json()
    url = f"/v1/architect/builds/{build['id']}"
    _wait_for(lambda: client.get(url).json().get("pending"))

    assert client.post(f"{url}/cancel").status_code == 200
    assert _wait_for(lambda: client.get(url).json()["status"] == "cancelled")
    assert client.get(url).json()["pending"] is None


# ── refine ──────────────────────────────────────────────────────────────────

def test_refining_a_missing_workflow_is_404_not_a_late_failure(api):
    """It would otherwise surface deep in the agent as 'cannot load package'."""
    client, _, _ = api
    res = client.post(
        "/v1/architect/builds", json={"intent": "add a step", "refines": "ghost"}
    )
    assert res.status_code == 404


def test_a_refine_build_tells_the_agent_what_it_is_changing(api, tmp_path):
    client, manager, sink = api
    # Register something to refine.
    import yaml

    from neurosurfer.graph.workflow.package import load_package

    pkg_dir = tmp_path / "src" / "existing_wf"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "workflow.yaml").write_text(
        yaml.safe_dump({"name": "existing_wf", "version": "0.1.0", "entrypoint": "graph.yaml"})
    )
    (pkg_dir / "graph.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "existing_wf",
                "nodes": [{"id": "a", "kind": "base", "goal": "do"}],
                "inputs": [],
                "outputs": ["a"],
            }
        )
    )
    manager.registry.save(load_package(pkg_dir))

    res = client.post(
        "/v1/architect/builds",
        json={"intent": "also send an email", "refines": "existing_wf"},
    )
    assert res.status_code == 202
    assert res.json()["refines"] == "existing_wf"

    _wait_for(lambda: "intent" in sink)
    # Handed the target explicitly — the agent seeds its session from it, rather
    # than being asked in prose to go and fetch the workflow itself.
    assert sink["refines"] == "existing_wf"
    assert "also send an email" in sink["intent"], "…and what to change"


def test_refining_seeds_the_session_with_the_real_graph(tmp_path):
    """The defect this closes: `refines` only reworded the intent. The agent had
    no way to load an existing workflow, so it began on an empty canvas and
    rebuilt from scratch — replacing the workflow rather than changing it."""
    import yaml

    from neurosurfer.architect.agent.session import BuildSession
    from neurosurfer.graph.workflow.package import load_package

    registry = WorkflowRegistry(workflows_dir=tmp_path / "registry")
    pkg_dir = tmp_path / "src" / "seeded_wf"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "workflow.yaml").write_text(
        yaml.safe_dump({"name": "seeded_wf", "version": "0.1.0", "entrypoint": "graph.yaml"})
    )
    (pkg_dir / "graph.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "seeded_wf",
                "description": "does a thing",
                "inputs": [{"name": "text", "required": True}],
                "nodes": [
                    {"id": "a", "kind": "base", "goal": "first"},
                    {"id": "b", "kind": "base", "goal": "second", "depends_on": ["a"]},
                ],
                "outputs": ["b"],
            }
        )
    )
    registry.save(load_package(pkg_dir))

    session = BuildSession(
        intent="add a branch",
        staging_root=tmp_path / "staging",
        registry=registry,
        knowledge=None,
    )
    session.load_from(registry.get("seeded_wf"))

    assert session.name == "seeded_wf"
    assert session.description == "does a thing"
    assert session.node_ids() == ["a", "b"], "the existing design must survive"
    assert session.outputs == ["b"]
    assert [i["name"] for i in session.inputs] == ["text"]


def test_a_seeded_session_starts_unverified(tmp_path):
    """It was verified as a different graph, and is about to change again."""
    import yaml

    from neurosurfer.architect.agent.session import BuildSession
    from neurosurfer.graph.workflow.package import load_package

    registry = WorkflowRegistry(workflows_dir=tmp_path / "registry")
    pkg_dir = tmp_path / "src" / "wf"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "workflow.yaml").write_text(
        yaml.safe_dump({"name": "wf", "version": "0.1.0", "entrypoint": "graph.yaml"})
    )
    (pkg_dir / "graph.yaml").write_text(
        yaml.safe_dump(
            {"name": "wf", "nodes": [{"id": "a", "kind": "base", "goal": "g"}], "outputs": ["a"]}
        )
    )
    registry.save(load_package(pkg_dir))

    session = BuildSession(
        intent="x", staging_root=tmp_path / "s", registry=registry, knowledge=None
    )
    session.record_verification(
        passed=True, rendered="stale report", report=None, test_inputs=None
    )
    assert session.last_verification is not None

    session.load_from(registry.get("wf"))
    assert session.last_verification is None


# ── credentials the workflow needs before it can be run (V4 Phase 10) ──────────

def test_supplied_secrets_are_stored_and_never_reach_the_build_record():
    """The security property, asserted on the record the browser actually sees.

    A `secrets_request` answer is a dict of credential values. The build record is
    streamed over SSE, replayed on reconnect, and kept for the life of the build —
    so the values must be dropped and only the names kept. This is the test that
    fails if someone later "simplifies" `_resolved` back to recording the answer.
    """
    from neurosurfer.app.server.architect_builds.manager import ArchitectManager
    from neurosurfer.app.server.architect_builds.store import BuildRecord

    rec = BuildRecord(intent="anything")

    class _I:
        id = "i1"
        kind = "secrets_request"

    ArchitectManager._resolved(rec, _I(), {"SALES_DB_URL": "postgres://u:hunter2@h/db"}, True)

    blob = json.dumps([e.to_dict() if hasattr(e, "to_dict") else e for e in rec.events],
                      default=str)
    assert "hunter2" not in blob, "a credential reached the build record"
    assert "postgres://" not in blob
    assert "SALES_DB_URL" in blob, "the name is kept — it is what makes the log legible"


def test_other_interactions_still_record_their_answer():
    """The redaction is scoped to the one kind that carries secrets."""
    from neurosurfer.app.server.architect_builds.manager import ArchitectManager
    from neurosurfer.app.server.architect_builds.store import BuildRecord

    rec = BuildRecord(intent="anything")

    class _I:
        id = "i2"
        kind = "question"

    ArchitectManager._resolved(rec, _I(), "weekly", True)
    blob = json.dumps([e.to_dict() if hasattr(e, "to_dict") else e for e in rec.events],
                      default=str)
    assert "weekly" in blob


@pytest.mark.asyncio
async def test_verification_asks_for_a_missing_value_and_carries_on():
    """The build parks instead of running a workflow that cannot connect."""
    from neurosurfer.architect.agent.session import BuildSession

    asked: list[list[dict]] = []

    async def _request(missing):
        asked.append(missing)
        return True

    s = BuildSession.__new__(BuildSession)
    s.request_secrets = _request
    assert await BuildSession.await_secrets(s, [{"name": "DB_URL", "source": "node:load"}])
    assert asked == [[{"name": "DB_URL", "source": "node:load"}]]


@pytest.mark.asyncio
async def test_headless_verification_does_not_park():
    """With nobody to ask, the run goes ahead and fails honestly."""
    from neurosurfer.architect.agent.session import BuildSession

    s = BuildSession.__new__(BuildSession)
    s.request_secrets = None
    assert not await BuildSession.await_secrets(s, [{"name": "DB_URL"}])


def test_a_secret_saved_mid_build_is_visible_to_the_next_check():
    """The double-ask, which cost a real user seven fields twice.

    A build binds its account's credentials once, on the request thread, because
    the worker starts with an empty context. Values supplied *during* the build
    went to the account store and not into that binding — so the requirement
    check that ran seconds later still reported every one of them missing.
    """
    from neurosurfer.mcp.credentials import (
        lookup,
        remember_credential,
        use_credentials,
    )

    with use_credentials({"ALREADY": "x"}):
        assert lookup("SUPPLIED_MID_BUILD") == ("", "")
        remember_credential("SUPPLIED_MID_BUILD", "value")
        assert lookup("SUPPLIED_MID_BUILD") == ("value", "account")
        assert lookup("ALREADY")[0] == "x", "the existing binding survives"

    # And the binding is still scoped — it does not leak past the build.
    assert lookup("SUPPLIED_MID_BUILD") == ("", "")


def test_remembering_an_empty_value_is_not_remembering_it():
    """A field left blank must stay missing, or the next check reports it set."""
    from neurosurfer.mcp.credentials import lookup, remember_credential, use_credentials

    with use_credentials({"A": "1"}):
        remember_credential("BLANK", "   ")
        assert lookup("BLANK") == ("", "")


# ── the panel is a chat, so the agent is given the chat (V4 Phase 14) ──────────

def test_earlier_turns_reach_the_clarifier():
    """The gap this closes, in the shape it appeared.

    A build blocked asking for a connection string. The user pasted one. The
    Architect replied "what would you like to automate with it?" — because
    `refines` only links to a *registered* workflow, and a blocked build registers
    nothing, so the follow-up arrived as a cold start.
    """
    from neurosurfer.architect.conversation import _prior_turns

    turns = _prior_turns([
        {"role": "user", "text": "Audit the access database"},
        {"role": "assistant", "text": "I could not build this: no connection"},
    ])
    assert len(turns) == 2
    rendered = " ".join(str(t.content) for t in turns)
    assert "Audit the access database" in rendered
    assert "no connection" in rendered


def test_history_is_bounded_and_skips_empties():
    """Browser-supplied text becoming model context has to be capped."""
    from neurosurfer.architect.conversation import _HISTORY_TURNS, _prior_turns

    many = [{"role": "user", "text": f"turn {i}"} for i in range(40)]
    assert len(_prior_turns(many)) == _HISTORY_TURNS
    assert _prior_turns([{"role": "user", "text": "   "}]) == []
    assert _prior_turns(None) == []


def test_a_long_turn_is_truncated():
    from neurosurfer.architect.conversation import _HISTORY_CHARS, _prior_turns

    [turn] = _prior_turns([{"role": "user", "text": "x" * 5000}])
    assert len(str(turn.content)) <= _HISTORY_CHARS + 200


def test_the_route_rejects_a_malformed_history():
    """It is request data, not something to trust into a prompt."""
    from neurosurfer.app.server.api.routes_architect import _history

    assert _history("not a list") is None
    assert _history([]) is None
    assert _history(["bare string", 42, None]) is None
    assert _history([{"role": "user", "text": "ok"}]) == [
        {"role": "user", "text": "ok"}
    ]
    # An unknown role becomes assistant rather than being passed through.
    assert _history([{"role": "system", "text": "x"}])[0]["role"] == "assistant"


# ── the plan's open questions get asked (V4) ───────────────────────────────────

@pytest.mark.asyncio
async def test_open_questions_are_put_to_the_user_and_reach_the_builder():
    """They were rendered at the foot of the plan card and never asked — so a
    decision only the user could make was shown to them as a footnote and then
    guessed at."""
    from neurosurfer.architect.agent.agent import ArchitectAgent

    asked: list[str] = []

    async def _ask(question, _choices=None):
        asked.append(question)
        return "use the Inquiries table"

    class _Session:
        ask_question = staticmethod(_ask)
        plan_answers = ""

    class _Plan:
        open_questions = ["Which table holds the inquiries?"]

    agent = ArchitectAgent.__new__(ArchitectAgent)
    agent._notify = lambda _m: None
    session = _Session()
    await ArchitectAgent._ask_open_questions(agent, session, _Plan())

    assert asked == ["Which table holds the inquiries?"]
    assert "use the Inquiries table" in session.plan_answers

    # And it travels to the builder as a decision, not a suggestion. Only with a
    # plan: answers to a plan's questions have nowhere to belong without one.
    class _Rendered:
        steps = ["a step"]          # `_render_prompt` gates on a plan having steps
        external_steps: list = []

        @staticmethod
        def render():
            return "PLAN: x"

    prompt = ArchitectAgent._render_prompt(
        "build it", None, _Rendered(), plan_answers=session.plan_answers
    )
    assert "use the Inquiries table" in prompt
    assert "decisions, not suggestions" in prompt


@pytest.mark.asyncio
async def test_silence_on_an_open_question_does_not_stall_the_build():
    """The gate's timeout answer means "you decide" — which is what the build
    would have done anyway."""
    from neurosurfer.architect.agent.agent import ArchitectAgent

    async def _ask(_q, _c=None):
        return "no preference"

    class _Session:
        ask_question = staticmethod(_ask)
        plan_answers = ""

    class _Plan:
        open_questions = ["anything?"]

    agent = ArchitectAgent.__new__(ArchitectAgent)
    agent._notify = lambda _m: None
    session = _Session()
    await ArchitectAgent._ask_open_questions(agent, session, _Plan())
    assert session.plan_answers == ""


@pytest.mark.asyncio
async def test_no_channel_means_no_questions_rather_than_a_crash():
    from neurosurfer.architect.agent.agent import ArchitectAgent

    class _Session:
        ask_question = None
        plan_answers = ""

    class _Plan:
        open_questions = ["anything?"]

    agent = ArchitectAgent.__new__(ArchitectAgent)
    agent._notify = lambda _m: None
    await ArchitectAgent._ask_open_questions(agent, _Session(), _Plan())

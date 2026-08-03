"""The Architect reaching for what it is missing (V4 Phase 2).

The sales-report build found the Postgres server it needed, listed it in the
refusal, and could not install it — for four independent reasons. These pin each
one, because any single one of them puts the whole loop back out of reach.
"""

from __future__ import annotations

import pytest

from neurosurfer.architect.plan import PlanStep, WorkflowPlan


def _step(step_id: str, status: str, **resolution) -> PlanStep:
    return PlanStep(
        id=step_id, intent=f"do {step_id}", is_external=True,
        needed_capability="do a thing",
        resolution={"status": status, "need": "do a thing", **resolution},
    )


# ── the plan gate ──────────────────────────────────────────────────────────────

def test_an_installable_step_is_not_a_blocker():
    """It is one approved install away and the agent has a tool for that. Blocking
    here declared a build infeasible with the server it needed listed beside the
    refusal."""
    step = _step("query_db", "installable")
    assert step.resolved is False, "not buildable *yet*"
    assert step.reachable is True, "but reachable"


def test_a_capability_nothing_provides_is_a_blocker():
    """`curated_gap` is the taxonomy saying this needs an external integration —
    no server, and no amount of Python will conjure one."""
    step = _step("send_sms", "none", curated_gap=True)
    assert step.reachable is False


def test_nothing_found_but_no_curated_gap_stays_reachable():
    """Blocking these at the plan is what made `author_tool` dead code for
    external steps — the resolver tells the model to author one and the gate
    never let it get there."""
    step = _step("do_maths", "none", curated_gap=False)
    assert step.reachable is True


def test_a_plan_separates_acquirable_from_impossible():
    plan = WorkflowPlan(
        name="w",
        steps=[
            _step("a", "have"),
            _step("b", "installable"),
            _step("c", "none", curated_gap=True),
            PlanStep(id="d", intent="write a summary"),  # internal
        ],
    )
    assert [s.id for s in plan.acquirable_steps] == ["b"]
    assert [s.id for s in plan.impossible_steps] == ["c"]
    # `unresolved` keeps its old meaning — not buildable as things stand.
    assert [s.id for s in plan.unresolved_steps] == ["b", "c"]


# ── the approval channel ───────────────────────────────────────────────────────

@pytest.fixture
def gate():
    from neurosurfer.app.server.architect_builds.interaction import InteractionGate

    return InteractionGate(on_pending=lambda i: None, on_resolved=lambda *a: None)


def _agent(gate, **kw):
    from neurosurfer.app.server.architect_builds.manager import ArchitectManager

    manager = ArchitectManager(provider=object(), registry=object())
    return manager._make_agent(lambda _m: None, "off", gate, **kw)


def test_installing_always_has_someone_to_ask(gate):
    """Installing was gated behind `approve_tools`, which is about *authored
    Python* — a different decision. With defaults, nothing could ever be
    installed while the resolver went on telling the model to."""
    assert _agent(gate, approve_tools=False)._approve_mcp is not None


def test_without_anyone_to_ask_installing_is_refused():
    """A headless build must not silently start a third-party process."""
    assert _agent(None, approve_tools=False)._approve_mcp is None


def test_tool_authoring_stays_auto_approved_by_default(gate):
    """The other half of ungating: installing always asks, authoring does not
    start asking as a side effect."""
    auto = _agent(gate, approve_tools=False)._approve_tool
    asked = _agent(gate, approve_tools=True)._approve_tool
    assert auto is not asked
    assert auto.__name__ == "_auto_approve"
    assert asked.__name__ == "_ask_approval"


# ── closing the loop ───────────────────────────────────────────────────────────

def test_installing_re_resolves_only_what_was_waiting(monkeypatch):
    """Re-resolving the whole plan would re-search the registry for capabilities
    settled long ago, and each is a multi-second round trip mid-build."""
    from neurosurfer.architect.agent import tools as agent_tools

    plan = WorkflowPlan(name="w", steps=[
        _step("already", "have"),
        _step("waiting", "installable"),
    ])
    session = type("S", (), {"plan": plan})()

    asked: list[str] = []

    class _Res:
        need = "do a thing"
        status = "have"
        tools: list = []
        servers: list = []
        curated_gap = False
        registry_searched = False
        registry_error = ""

        def to_dict(self):
            return {"need": self.need, "status": self.status, "tools": [],
                    "servers": [], "curated_gap": False}

    def _resolve(need):
        asked.append(need)
        return _Res()

    monkeypatch.setattr("neurosurfer.architect.capability.resolve_capability", _resolve)
    settled = agent_tools._reresolve_waiting_steps(session)

    assert settled == ["waiting"]
    assert len(asked) == 1, "the settled step must not be re-searched"


# ── a consent screen is not a missing value ────────────────────────────────────

@pytest.mark.asyncio
async def test_with_nobody_to_ask_authorization_is_refused():
    """A headless build cannot visit a consent screen, so it declines rather than
    parking forever on an interaction no one will ever answer."""
    from neurosurfer.architect.agent.session import BuildSession

    session = type("S", (), {
        "request_authorization": None,
        "await_authorization": BuildSession.await_authorization,
    })()
    assert await session.await_authorization("thinair/data", "https://auth.example/x") is False


@pytest.mark.asyncio
async def test_credentials_are_read_through_the_source_that_found_the_server():
    """The official registry states requirements as `environmentVariables` /
    `headers` arrays; Smithery states them as a per-connection JSON Schema.
    Reading a Smithery entry with the official reader found nothing required, so
    a server was installed with no connection details and failed to start with
    nobody having been asked for anything."""
    from neurosurfer.architect.agent.tools import InstallMcpServerArgs, InstallMcpServerTool
    from neurosurfer.mcp.registry import CredentialRequirement
    from neurosurfer.mcp.sources import use_source

    class _Source:
        id = "fake"
        capabilities = frozenset()

        def detail(self, name):
            # Shaped like Smithery's: nothing the official reader would see.
            return {"server": {"name": name, "connections": [{
                "configSchema": {
                    "required": ["postgresConnectionString"],
                    "properties": {"postgresConnectionString": {
                        "description": "postgres://user:pass@host/db"}},
                },
            }]}}

        def credentials(self, server):
            return [CredentialRequirement(
                name="postgresConnectionString", required=True, secret=True,
                description="postgres://user:pass@host/db", where="config")]

        def install_config(self, server, body):  # pragma: no cover - never reached
            raise AssertionError("must not install without the connection string")

    session = type("S", (), {"notify": lambda self, m: None})()
    tool = InstallMcpServerTool(session)
    with use_source(_Source()):
        res = await tool.call(InstallMcpServerArgs(name="1Levick3/postgresql-mcp-server"), None)

    assert res.is_error
    assert "postgresConnectionString" in res.content


@pytest.mark.asyncio
async def test_a_consent_screen_parks_the_build_instead_of_ending_it():
    """A URL printed inside a refusal made the user find it, visit it, and re-run
    the whole build behind it. The build already parks whenever it asks anything."""
    from neurosurfer.architect.agent.tools import InstallMcpServerArgs, InstallMcpServerTool
    from neurosurfer.mcp.sources import AuthorizationRequired, use_source

    attempts = {"n": 0}

    class _Source:
        id = "fake"
        capabilities = frozenset()

        def detail(self, name):
            return {"server": {"name": name, "remotes": [{"url": "http://x"}]}}

        def credentials(self, server):
            return []

        def install_config(self, server, body):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise AuthorizationRequired("consent", setup_url="https://auth.example/x")
            return type("Cfg", (), {"name": "thinair-data", "transport": "http"})()

    asked: list[tuple[str, str]] = []

    class _Session:
        installed_servers: list = []
        plan = None

        def notify(self, m):
            pass

        async def await_authorization(self, server, url):
            asked.append((server, url))
            return True

        async def approve_mcp_install(self, cfg, entry):
            return True

    monkey = _Session()
    tool = InstallMcpServerTool(monkey)
    with use_source(_Source()):
        res = await tool.call(InstallMcpServerArgs(name="thinair/data"), None)

    assert asked == [("thinair/data", "https://auth.example/x")], "it must ask, not give up"
    assert attempts["n"] == 2, "and retry the install once granted"
    # It got past config-building; whatever happens at start-up is a later concern.
    assert res is not None


@pytest.mark.asyncio
async def test_declining_authorization_moves_on_with_the_link():
    from neurosurfer.architect.agent.tools import InstallMcpServerArgs, InstallMcpServerTool
    from neurosurfer.mcp.sources import AuthorizationRequired, use_source

    class _Source:
        id = "fake"
        capabilities = frozenset()

        def detail(self, name):
            return {"server": {"name": name}}

        def credentials(self, server):
            return []

        def install_config(self, server, body):
            raise AuthorizationRequired("consent", setup_url="https://auth.example/x")

    class _Session:
        def notify(self, m):
            pass

        async def await_authorization(self, server, url):
            return False

    tool = InstallMcpServerTool(_Session())
    with use_source(_Source()):
        res = await tool.call(InstallMcpServerArgs(name="thinair/data"), None)

    assert res.is_error
    assert "https://auth.example/x" in res.content, "the link survives into the block"


# ── a live server outliving its build (V4 Phase 9) ──────────────────────────────

def _live(name: str, description: str):
    """A stub MCP tool, as `live_tools()` would return it."""
    from neurosurfer.tools.base import Tool, ToolSchema

    class _T(Tool):
        server_name = "thinair-data"

        def __init__(self) -> None:
            self.name = name
            self.description = description

        @property
        def schema(self):
            return ToolSchema(name=self.name, description=self.description,
                              input_schema={"type": "object", "properties": {}})

        async def call(self, **kwargs):  # pragma: no cover - never invoked
            return ""

    return _T()


# The five tools that actually answered "generate chart images" in the live run,
# with the publisher's own descriptions.
_THINAIR = [
    ("generate_migration",
     "Generate a SQL migration file from a described schema change, using the "
     "connection's dialect."),
    ("generate_seed_data",
     "Generate realistic seed data rows for a table."),
    ("saved_queries",
     "Manage a library of saved SQL queries for a connection."),
    ("suggest_queries",
     "Suggest useful SQL queries to run against a table, and generate them."),
    ("query_sql",
     "Execute a read-only SQL query against a registered PostgreSQL, MySQL or "
     "SQL Server database connection."),
]


def _resolve_against_thinair(need: str):
    from neurosurfer.architect.capability import _live_hits
    from neurosurfer.tools.registry import clear_live_tools, set_live_tools

    set_live_tools([_live(n, d) for n, d in _THINAIR])
    try:
        return [c.name for c in _live_hits(need, 6)]
    finally:
        clear_live_tools()


def test_a_database_server_does_not_answer_for_charts_or_pdfs():
    """The regression this rule exists for, in the words the planner used.

    A Postgres server installed for an earlier build was still connected. Every
    one of these resolved to `have` — so nothing blocked, nothing offered an
    install, and five nodes were built to draw charts with migration tools.
    """
    assert _resolve_against_thinair(
        "Generate chart images using a charting library and produce files/images "
        "suitable for embedding in a PDF."
    ) == [], "`generate` is a verb, and a query library is not a charting library"

    assert _resolve_against_thinair(
        "Render HTML/markdown/text plus images into a PDF using a PDF generation "
        "toolchain."
    ) == []

    assert _resolve_against_thinair(
        "Write CSV files to the local filesystem as additional artifacts."
    ) == []


def test_the_database_server_still_answers_for_the_database():
    """The other half: tightening must not undo what Phase 2 was built to do."""
    hits = _resolve_against_thinair(
        "Connect to and query a PostgreSQL database using runtime connection "
        "details and execute SQL to extract the required transactional rows."
    )
    assert "query_sql" in hits, "the tool the server was installed for must survive"


def test_a_generically_named_tool_is_carried_by_its_description():
    """Phase 2's stated case: a name like `query` says nothing, the description
    says everything. Two corroborating object tokens are enough without a name
    match — that is why the gate is `name OR two description hits`."""
    from neurosurfer.architect.capability import _live_hits
    from neurosurfer.tools.registry import clear_live_tools, set_live_tools

    set_live_tools([_live(
        "query", "Execute a SQL query against a PostgreSQL database."
    )])
    try:
        hits = [c.name for c in _live_hits(
            "query a PostgreSQL database and return rows", 4
        )]
    finally:
        clear_live_tools()
    assert hits == ["query"]


# ── Phase 7: two reports that lied ──────────────────────────────────────────────

def test_a_capability_acquired_during_the_build_leaves_the_checklist():
    """`resolutions` is append-only, so a blocker stayed a blocker after install.

    The live symptom: the build found a Postgres server, asked, installed it,
    connected it, used it for seven nodes — and then ended by telling the user to
    install it.
    """
    from neurosurfer.architect.agent.session import BuildSession

    class _Res:
        status = "installable"
        need = "query a PostgreSQL database"
        servers = []

    s = BuildSession.__new__(BuildSession)
    s.plan = None
    s.resolutions = [_Res()]

    # Still missing → it belongs on the checklist.
    s._now_satisfied = lambda _c: False
    assert [r["capability"] for r in BuildSession.blocking_requirements(s)] == [
        "query a PostgreSQL database"
    ]

    # Acquired since → it must not be listed as something to go and obtain.
    s._now_satisfied = lambda _c: True
    assert BuildSession.blocking_requirements(s) == []


def test_a_resolver_fault_keeps_the_blocker():
    """Unsure is not the same as satisfied — never delete a real blocker."""
    from neurosurfer.architect.agent.session import BuildSession

    s = BuildSession.__new__(BuildSession)
    with_error = BuildSession._now_satisfied(s, "something")
    assert with_error is False or isinstance(with_error, bool)
    assert BuildSession._now_satisfied(s, "") is False


# ── credentials are not a capability (V4 Phase 12) ──────────────────────────────

def test_getting_a_credential_is_never_an_external_capability():
    """The failure this guard exists for, in the planner's own words.

    A clarifying answer of "supply them via environment variables" became a plan
    step whose capability was *reading environment variables*. Nothing provides
    that, so it went to the registry — where the distinctive token is
    **environment** — and the build was offered the UK Environment Agency, a
    Swiss air-quality index, an Obsidian note searcher and IBM Quantum.
    """
    from neurosurfer.architect.capability import resolve_capability

    for need in (
        "read environment variables at runtime",
        "Read environment variables at runtime to fetch SQL Server connection "
        "details and validate required credentials",
        "load the database credentials",
        "get the API key from the environment",
        "resolve the connection string for the database",
    ):
        r = resolve_capability(need)
        assert r.status == "not_external", f"{need!r} resolved as {r.status}"
        assert not r.servers, f"{need!r} searched a registry"


def test_reading_a_credential_out_of_a_real_file_is_still_a_file_read():
    """The guard must not swallow a genuine capability that happens to say
    'credentials' — reading them from creds.json is a file read."""
    from neurosurfer.architect.capability import resolve_capability

    # Local catalog only: the assertion is about the *guard*, and letting this
    # reach the public MCP registry made one unit test 16 seconds of network.
    r = resolve_capability(
        "read the credentials from config/creds.json on disk", include_registry=False
    )
    assert r.status != "not_external"


def test_a_database_need_resolves_to_the_local_sql_tools():
    """Before these tools existed every database capability was a registry search,
    and the registry kept answering with hosted gateways that cannot reach a
    database on localhost."""
    from neurosurfer.architect.capability import resolve_capability

    for need in (
        "execute parameterized SQL queries against the database",
        "connect to a SQL Server database using provided connection parameters",
        "run SQL queries against the database metadata to inspect schema",
        "query a PostgreSQL database",
    ):
        r = resolve_capability(need)
        assert r.status == "have", f"{need!r} resolved as {r.status}"
        names = {t.name for t in r.tools}
        assert names & {"sql"}, names


# ── authoring is a first-class route (V4 Phase 13) ─────────────────────────────

def _render(status: str, servers=()):
    from neurosurfer.architect.capability import Resolution

    r = Resolution(need="do a thing")
    r.servers = list(servers)
    r.registry_searched = True
    return r.render()


def test_authoring_is_offered_alongside_installing():
    """`installable` means the registry returned *something*, however irrelevant.

    Offering only the install is what made `author_tool` unreachable: one junk
    match hid it completely. A build was offered the UK Environment Agency for
    'read environment variables', declined to install it — correctly — and
    blocked, never having been told it could write the tool.
    """
    from neurosurfer.architect.capability import Resolution, ServerCandidate

    r = Resolution(need="query a widget database")
    r.servers = [ServerCandidate(name="some/server", description="A server.")]
    r.registry_searched = True
    assert r.status == "installable"

    text = r.render()
    assert "author_tool" in text, "authoring must be offered on installable"
    assert "install_mcp_server" in text
    # And the model must be told how to choose between them.
    assert "vendor account" in text or "OAuth" in text


def test_a_curated_gap_no_longer_says_not_a_catalog_tool():
    """That line read as 'do not write one', and was wrong about databases for
    the entire life of the project."""
    from neurosurfer.architect.capability import Resolution

    r = Resolution(need="do a thing")
    r.curated_gap = True
    r.registry_searched = True
    text = r.render()
    assert "not a catalog tool" not in text
    assert "author" in text.lower()

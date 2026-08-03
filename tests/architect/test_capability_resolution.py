"""The capability resolution ladder + MCP discovery (Architect V3, Phase 2).

The registry is stubbed throughout — these assert the *ladder's* behaviour, not
the internet's. The one thing worth stating up front: a bad suggestion is worse
than none, because a small model attaches whatever it is handed. Several tests
below exist only to pin down things the resolver must NOT say.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.architect.capability import (
    Resolution,
    registry_terms,
    resolve_capability,
    search_catalog,
)
from neurosurfer.mcp.registry import credential_requirements

# ── canned registry payloads ────────────────────────────────────────────────────

_GMAIL = {
    "server": {
        "name": "com.example/gmail-mcp",
        "description": "Read emails, send messages, and manage labels in Gmail.",
        "version": "1.0.0",
        "packages": [{
            "registryType": "npm",
            "identifier": "gmail-mcp",
            "version": "1.0.0",
            "transport": {"type": "stdio"},
            "environmentVariables": [
                {"name": "GOOGLE_CLIENT_ID", "description": "OAuth client id",
                 "isRequired": True},
                {"name": "GOOGLE_CLIENT_SECRET", "description": "OAuth secret",
                 "isRequired": True, "isSecret": True},
                {"name": "GOOGLE_REDIRECT_URI", "description": "Redirect URI"},
            ],
            # A launch flag publishers mark required — must NOT read as a credential.
            "packageArguments": [{"name": "stdio", "isRequired": True}],
        }],
    },
}

_SMS = {
    "server": {
        "name": "com.example/twilio",
        "description": "Twilio MCP — send SMS and make calls.",
        "version": "2.0.0",
        "remotes": [{
            "type": "streamable-http",
            "url": "https://mcp.example.com/twilio",
            "headers": [{"name": "Authorization", "description": "Bearer token",
                         "isRequired": True, "isSecret": True}],
        }],
    },
}


@pytest.fixture()
def stub_registry(monkeypatch):
    """Answer registry searches from the canned entries above."""
    from neurosurfer.mcp import registry as mcp_registry

    def _fake(path: str, params: dict):
        term = (params.get("search") or "").lower()
        if "/versions" in path:
            name = path.split("/servers/")[1].split("/versions")[0]
            entry = _GMAIL if "gmail" in name else _SMS
            return {"servers": [{**entry,
                                 "_meta": {"io.modelcontextprotocol.registry/official":
                                           {"isLatest": True}}}]}
        if term in {"gmail", "email", "imap"}:
            return {"servers": [_GMAIL]}
        if term in {"sms", "twilio", "slack"}:
            return {"servers": [_SMS]}
        return {"servers": []}

    mcp_registry._cache.clear()
    monkeypatch.setattr(mcp_registry, "registry_get", _fake)
    yield
    mcp_registry._cache.clear()


# ── the catalog leg ─────────────────────────────────────────────────────────────

def test_a_real_capability_finds_its_real_tool():
    """A ranked shortlist: other file tools may follow, but the verb decides
    which one leads."""
    tools = search_catalog("read a file from disk")
    assert tools[0].name == "read_file"
    assert tools[0].origin == "builtin"
    assert search_catalog("write a file to disk")[0].name == "write_file"


@pytest.mark.parametrize(
    ("need", "must_not_suggest"),
    [
        # Every one of these was produced by the first, looser scorer.
        ("read an email inbox", "read_file"),      # matched the verb "read"
        ("send a text message", "http"),           # its description says "text"
        ("send a text message", "browse"),
        ("read an email inbox", "data"),
    ],
)
def test_the_catalog_does_not_offer_near_misses(need, must_not_suggest):
    """A wrong tool is worse than no tool: a small model attaches it and ships
    a workflow that invents its results."""
    assert must_not_suggest not in [t.name for t in search_catalog(need)]


def test_verbs_alone_never_match():
    """`read_file` shares only the verb with "read an email inbox"."""
    assert search_catalog("read an email inbox") == []


# ── the curated taxonomy drives both legs ───────────────────────────────────────

def test_curated_capability_with_no_builtin_goes_straight_to_the_registry(stub_registry):
    res = resolve_capability("read an email inbox")
    assert res.status == "installable"
    assert res.curated_gap, "the taxonomy knows nothing built in does this"
    assert res.registry_searched
    assert [s.name for s in res.servers] == ["com.example/gmail-mcp"]


def test_a_satisfied_capability_never_queries_the_registry(stub_registry, monkeypatch):
    """If a tool already does the job, offering an install invites the expensive
    path for no reason."""
    from neurosurfer.mcp import registry as mcp_registry

    def _boom(*a, **k):
        raise AssertionError("the registry must not be consulted")

    monkeypatch.setattr(mcp_registry, "registry_get", _boom)
    res = resolve_capability("read a file from disk")
    assert res.status == "have" and not res.registry_searched


def test_registry_terms_lead_with_the_curated_ones():
    """The registry's search is single-term: "read an email inbox" returns
    nothing, "gmail" returns servers. Curated terms encode that and go first."""
    assert registry_terms("read an email inbox")[:3] == ["gmail", "email", "imap"]
    assert registry_terms("send a text message")[:3] == ["sms", "twilio", "slack"]
    # Outside the taxonomy, fall back to the need's own object tokens.
    assert "kubernetes" in registry_terms("scale a kubernetes deployment")


def test_curated_terms_do_not_discard_the_engine_the_user_named():
    """A generic "database" rule used to replace the need with postgres/database/
    sqlite, so "query a SQL Server" searched for three engines nobody asked for
    and never for the one they did — and the registry has `mcp-mssql`."""
    terms = registry_terms("query a SQL server to retrieve data")
    assert terms[0] == "mssql"
    assert "sqlserver" in terms
    assert "sqlite" not in terms and "postgres" not in terms

    # The named engines are distinguished from one another, not merged.
    assert registry_terms("read from a postgres database")[0] == "postgres"
    assert registry_terms("query a mysql table")[0] == "mysql"
    # ...and a phrase naming no engine still gets the generic rule.
    assert registry_terms("query a database")[0] == "database"


def test_a_semantic_engine_is_given_the_phrase():
    """Reducing a phrase to keywords throws away the meaning a semantic index
    exists to read. Gated on the declared capability, never on the engine name."""
    need = "query a SQL server to retrieve data"
    assert registry_terms(need, semantic=True) == [need]


def test_the_architect_searches_the_engine_the_account_chose():
    """This is the bug: capability resolution imported the official client
    directly, so choosing Smithery changed the catalog browser and nothing about
    what a build searched."""
    from neurosurfer.architect.capability import search_servers
    from neurosurfer.mcp.registry import RegistryHit
    from neurosurfer.mcp.sources import Capability, use_source

    asked: list[str] = []

    class _Fake:
        id = "fake"
        label = "Fake"
        blurb = ""
        capabilities = frozenset({Capability.SEMANTIC_SEARCH})
        requires_key = False

        def available(self):
            return True

        def key_set(self):
            return False

        def search(self, query, *, limit=20, cursor=""):
            asked.append(query)
            return [RegistryHit(name="acme/sqlserver", description="Query SQL Server.")]

    with use_source(_Fake()):
        candidates, error = search_servers("query a SQL server to retrieve data")

    assert error == ""
    assert [c.name for c in candidates] == ["acme/sqlserver"]
    # Semantic engine ⇒ the phrase, not keywords.
    assert asked == ["query a SQL server to retrieve data"]


def test_a_semantic_engine_s_own_ranking_is_not_second_guessed():
    """The lexical scorer drops anything sharing no tokens with the phrase. On a
    semantic engine that discards exactly-right answers — the engine ranked by
    meaning and this would overrule it with the weaker judge."""
    from neurosurfer.architect.capability import search_servers
    from neurosurfer.mcp.registry import RegistryHit
    from neurosurfer.mcp.sources import Capability, use_source

    class _Fake:
        id = "fake"
        label = "Fake"
        blurb = ""
        capabilities = frozenset({Capability.SEMANTIC_SEARCH})
        requires_key = False

        def available(self):
            return True

        def key_set(self):
            return False

        def search(self, query, *, limit=20, cursor=""):
            return [
                RegistryHit(name="thinair/data", description="Dialect-aware warehouse tools."),
                RegistryHit(name="acme/sql", description="query sql server data"),
            ]

    with use_source(_Fake()):
        candidates, _ = search_servers("query a SQL server to retrieve data")

    # Shares no tokens with the need, and survives in the position the engine gave it.
    assert [c.name for c in candidates] == ["thinair/data", "acme/sql"]


def test_unknown_capability_still_searches(stub_registry):
    res = resolve_capability("do something nobody has ever needed")
    assert res.status == "none"
    assert not res.curated_gap    # the taxonomy never recognised it


# ── credentials ─────────────────────────────────────────────────────────────────

def test_credentials_are_read_off_the_registry_entry():
    creds = {c.name: c for c in credential_requirements(_GMAIL["server"])}
    assert set(creds) >= {"GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"}
    assert creds["GOOGLE_CLIENT_SECRET"].secret
    assert not creds["GOOGLE_CLIENT_ID"].secret
    assert creds["GOOGLE_REDIRECT_URI"].where == "env"


def test_launch_flags_are_not_credentials():
    """One real entry declares `stdio` and `http` as required packageArguments;
    listing those turns a credential checklist into noise."""
    assert "stdio" not in {c.name for c in credential_requirements(_GMAIL["server"])}


def test_remote_headers_are_credentials():
    creds = credential_requirements(_SMS["server"])
    assert [c.name for c in creds] == ["Authorization"]
    assert creds[0].where == "header" and creds[0].secret


def test_the_report_names_the_credentials_and_the_next_move(stub_registry):
    rendered = resolve_capability("read an email inbox").render()
    assert "GOOGLE_CLIENT_ID" in rendered
    assert "install_mcp_server" in rendered
    assert "declare_blocked" in rendered


def test_a_dead_registry_degrades_instead_of_failing(monkeypatch):
    from neurosurfer.mcp import registry as mcp_registry

    def _down(*a, **k):
        raise mcp_registry.McpRegistryError("registry unreachable")

    mcp_registry._cache.clear()
    monkeypatch.setattr(mcp_registry, "registry_get", _down)
    res = resolve_capability("read an email inbox")
    assert res.status == "none"
    assert "unreachable" in res.registry_error
    assert "author_tool" in res.render()    # still tells the agent what to do


def test_render_caps_a_long_credential_list(stub_registry):
    """One real entry declares eleven env vars; a wall of them buries the secrets."""
    from neurosurfer.architect.capability import ServerCandidate
    from neurosurfer.mcp.registry import CredentialRequirement

    res = Resolution(need="x", servers=[ServerCandidate(
        name="a/b", description="d",
        credentials=[CredentialRequirement(name=f"V{i}") for i in range(11)],
    )])
    rendered = res.render()
    assert "+6 more" in rendered
    assert rendered.count("`V") == 5


# ── the agent's toolbelt ────────────────────────────────────────────────────────

@pytest.fixture()
def session(tmp_path: Path):
    from neurosurfer.architect.agent import BuildSession
    from neurosurfer.architect.knowledge import KnowledgeBase
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    return BuildSession(
        intent="read my email",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=KnowledgeBase(),
    )


@pytest.fixture()
def ctx(tmp_path: Path):
    from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

    return ToolContext(cwd=tmp_path, io=AutoApproveIOHandler())


def _belt(session, name):
    from neurosurfer.architect.agent import architect_tools

    return next(t for t in architect_tools(session) if t.name == name)


async def test_find_capability_records_what_it_resolved(session, ctx, stub_registry):
    result = await _belt(session, "find_capability").run(
        {"capability": "read an email inbox"}, ctx)
    assert not result.is_error
    assert "gmail-mcp" in result.content
    assert [r.need for r in session.resolutions] == ["read an email inbox"]


async def test_install_refuses_without_the_credentials(session, ctx, stub_registry):
    """A server installed without its credentials fails mid-run, not here."""
    result = await _belt(session, "install_mcp_server").run(
        {"name": "com.example/gmail-mcp"}, ctx)
    assert result.is_error
    assert "GOOGLE_CLIENT_ID" in result.content
    assert "declare_blocked" in result.content


async def test_install_refuses_without_an_approval_channel(session, ctx, stub_registry):
    """Headless, the answer is no — this starts a third-party process."""
    assert session.approve_mcp is None
    result = await _belt(session, "install_mcp_server").run(
        {"name": "com.example/gmail-mcp",
         "env": {"GOOGLE_CLIENT_ID": "x", "GOOGLE_CLIENT_SECRET": "y"}}, ctx)
    assert result.is_error and "declined" in result.content


async def test_install_asks_the_approver_and_respects_a_no(session, ctx, stub_registry):
    asked: list = []

    async def _reject(cfg, entry):
        asked.append((cfg.name, entry.get("name")))
        return False

    session.approve_mcp = _reject
    result = await _belt(session, "install_mcp_server").run(
        {"name": "com.example/gmail-mcp",
         "env": {"GOOGLE_CLIENT_ID": "x", "GOOGLE_CLIENT_SECRET": "y"}}, ctx)
    assert result.is_error and "declined" in result.content
    assert asked == [("gmail-mcp", "com.example/gmail-mcp")]
    assert session.installed_servers == []


async def test_list_mcp_tools_says_so_when_nothing_is_connected(session, ctx):
    result = await _belt(session, "list_mcp_tools").run({}, ctx)
    assert not result.is_error
    assert "find_capability" in result.content


# ── blocked builds hand back a checklist ────────────────────────────────────────

async def test_blocking_requirements_carry_the_servers_and_credentials(
    session, ctx, stub_registry
):
    await _belt(session, "find_capability").run(
        {"capability": "read an email inbox"}, ctx)
    reqs = session.blocking_requirements()
    assert len(reqs) == 1
    assert reqs[0]["capability"] == "read an email inbox"
    creds = reqs[0]["servers"][0]["credentials"]
    assert "GOOGLE_CLIENT_SECRET" in {c["name"] for c in creds}


async def test_resolved_capabilities_are_not_blockers(session, ctx):
    await _belt(session, "find_capability").run(
        {"capability": "read a file from disk"}, ctx)
    assert session.blocking_requirements() == []


def test_workflow_infeasible_carries_requirements():
    from neurosurfer.architect import WorkflowInfeasible

    e = WorkflowInfeasible("no gmail", [{"capability": "read an email inbox"}])
    assert e.report == "no gmail"
    assert e.requirements[0]["capability"] == "read an email inbox"
    # Back-compat: the one-argument form still works everywhere it's raised.
    assert WorkflowInfeasible("plain").requirements == []


# ── composition is not a capability ─────────────────────────────────────────────

@pytest.mark.parametrize("need", [
    "draft an email",                 # the live gpt-4o-mini failure
    "draft a reply to the customer",
    "summarise the article",
    "classify the ticket by urgency",
    "write a catchy title",
    "analyse the content and extract insights",
])
def test_composition_is_not_an_external_capability(need, stub_registry):
    """gpt-4o-mini asked the ladder about "draft an email", got five email
    servers back, and reported drafting as a credentialled blocker."""
    res = resolve_capability(need)
    assert res.status == "not_external"
    assert not res.registry_searched
    assert "base` node" in res.render()
    assert "declare_blocked" not in res.render().split("NEXT:")[0]


def test_writing_a_file_is_still_external():
    """The taxonomy settles this before the composition check is ever asked —
    "write" appears in both vocabularies and the external reading must win."""
    res = resolve_capability("write the summary to a file on disk")
    assert res.status == "have"
    assert res.tools[0].name == "write_file"


def test_composition_needs_are_not_blockers(stub_registry):
    from neurosurfer.architect.agent import BuildSession
    from neurosurfer.architect.knowledge import KnowledgeBase
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    s = BuildSession(intent="x", staging_root=Path("/tmp/x"),
                     registry=WorkflowRegistry(), knowledge=KnowledgeBase())
    s.record_resolution(resolve_capability("draft an email"))
    assert s.blocking_requirements() == []


# ── credentials we already hold ─────────────────────────────────────────────────

def test_a_placeholder_naming_a_key_we_hold_is_not_a_blocker(monkeypatch):
    """The bug that blocked a real build: `Authorization: "Bearer
    {smithery_api_key}"` is *required* and also already satisfied whenever that
    key is on file — and `_remote_headers` has always known how to fill it."""
    from neurosurfer.architect.capability import ServerCandidate
    from neurosurfer.mcp.credentials import use_credentials
    from neurosurfer.mcp.registry import CredentialRequirement

    # A developer's `.env` is loaded into this process by anything calling
    # `load_config`, and a key there satisfies this for real — which is the
    # feature. The unheld case has to say so explicitly.
    monkeypatch.delenv("SMITHERY_API_KEY", raising=False)

    req = CredentialRequirement(
        name="Authorization", description="Bearer token for Smithery authentication",
        required=True, secret=True, where="header", template="Bearer {smithery_api_key}",
    )
    server = ServerCandidate(name="ai.smithery/x", description="", credentials=[req])

    assert server.required_credentials == [req], "still required — that is the entry's claim"
    assert server.blocking_credentials == [req], "and blocking, with nothing on file"

    with use_credentials({"smithery_api_key": "sk-held"}):
        assert server.blocking_credentials == [], "held ⇒ nothing to ask for"
        assert server.to_dict()["credentials"][0]["satisfied_from"] == "account"


def test_a_credential_is_asked_for_by_the_name_a_person_could_supply():
    """`Authorization` is where the value goes; `smithery_api_key` is the thing
    anybody actually has. A checklist naming the slot asks for the impossible."""
    from neurosurfer.mcp.credentials import asked_name
    from neurosurfer.mcp.registry import CredentialRequirement

    templated = CredentialRequirement(name="Authorization", where="header",
                                      template="Bearer {smithery_api_key}")
    plain = CredentialRequirement(name="SQLITE_PATH")

    assert asked_name(templated) == "smithery_api_key"
    assert asked_name(plain) == "SQLITE_PATH"
    assert "smithery_api_key" in templated.render()
    assert "Authorization" not in templated.render()


def test_an_environment_variable_satisfies_a_lowercase_placeholder(monkeypatch):
    """The placeholder is `smithery_api_key` and the variable is
    `SMITHERY_API_KEY`. Treating those as two credentials asks for one already set."""
    from neurosurfer.mcp.credentials import satisfied_from
    from neurosurfer.mcp.registry import CredentialRequirement

    req = CredentialRequirement(name="Authorization", where="header",
                                template="Bearer {smithery_api_key}")
    monkeypatch.delenv("SMITHERY_API_KEY", raising=False)
    assert satisfied_from(req) == ""
    monkeypatch.setenv("SMITHERY_API_KEY", "from-env")
    assert satisfied_from(req) == "environment"


def test_the_account_wins_over_the_environment(monkeypatch):
    """A user who set it in Settings meant it, and should not be overruled by
    whatever the gateway happened to be started with."""
    from neurosurfer.mcp.credentials import lookup, use_credentials

    monkeypatch.setenv("SMITHERY_API_KEY", "from-env")
    with use_credentials({"smithery_api_key": "from-settings"}):
        assert lookup("smithery_api_key") == ("from-settings", "account")


def test_install_is_handed_the_placeholder_value_not_the_header_name():
    """`_remote_headers` substitutes on the placeholder, so that is the key the
    supply has to use — filling `Authorization` directly would bypass the
    publisher's `Bearer ` framing."""
    from neurosurfer.mcp.credentials import supply_for, use_credentials
    from neurosurfer.mcp.registry import CredentialRequirement

    req = CredentialRequirement(name="Authorization", required=True, where="header",
                                template="Bearer {smithery_api_key}")
    with use_credentials({"smithery_api_key": "sk-held"}):
        assert supply_for([req]) == {"headers": {"smithery_api_key": "sk-held"}}


def test_a_satisfied_requirement_reads_as_configured_rather_than_missing(stub_registry):
    """The agent reads this text and decides whether to give up. "needs:
    Authorization" against a key we hold is how it declared a build impossible."""
    from neurosurfer.architect.capability import Resolution, ServerCandidate
    from neurosurfer.mcp.credentials import use_credentials
    from neurosurfer.mcp.registry import CredentialRequirement

    req = CredentialRequirement(name="Authorization", required=True, where="header",
                                template="Bearer {smithery_api_key}")
    res = Resolution(need="query a database", servers=[
        ServerCandidate(name="ai.smithery/x", description="d", credentials=[req]),
    ])
    with use_credentials({"smithery_api_key": "sk-held"}):
        text = res.render()
    assert "already configured here" in text
    assert "needs: `smithery_api_key`" not in text


# ── two resolutions that lied (V4 Phase 1) ──────────────────────────────────────

@pytest.mark.parametrize("need", [
    "export data as PDF",
    "query customer revenue data",
    "query sales data for comparison",
])
def test_a_generic_tool_name_does_not_answer_for_everything(need, stub_registry):
    """`data` is "inspect a local structured-data file, read-only" and its whole
    name is one generic noun, so every need containing that word matched it at
    full strength — and reported `have`, so it blocked nothing. A build would go
    green and hand back a "PDF report" that is a CSV inspector call."""
    res = resolve_capability(need)
    assert "data" not in [t.name for t in res.tools], res.render()
    assert res.status != "have"


def test_a_generic_name_still_answers_when_the_description_agrees(stub_registry):
    """The fix is corroboration, not exclusion: `data` genuinely does this one,
    and its description says so."""
    assert "data" in [t.name for t in resolve_capability("query a CSV data file").tools]


def test_a_specific_name_needs_no_corroboration():
    """`read_file` is two tokens and one is a verb — matching "file" there is a
    claim about reading files, not a coincidence of vocabulary."""
    assert search_catalog("read a file from disk")[0].name == "read_file"


@pytest.mark.parametrize(("need", "why"), [
    ("generate visual charts", "charts are matplotlib, not thinking"),
    ("analyze product sales data", "a database is not composed"),
    ("write the results to the database", "a destination, not prose"),
])
def test_a_composition_verb_needs_a_text_object(need, why, stub_registry):
    """The verb alone declared these internal, and the plan then produced a `tool`
    node with no tool — the exact thing Phase 1's gate refuses, on a build that
    never got far enough to be refused."""
    assert resolve_capability(need).status != "not_external", why


@pytest.mark.parametrize("need", [
    "draft an email reply",
    "write a catchy title",
    "summarise an article",
    "classify a support ticket",
])
def test_composing_text_is_still_not_external(need, stub_registry):
    """The reason the verb list exists: "draft an email" used to return five email
    servers, and the weak model then reported drafting as a credentialled blocker."""
    assert resolve_capability(need).status == "not_external"


@pytest.mark.parametrize(("need", "must_not_suggest"), [
    # Live findings from the sales-report build.
    ("export data as a PDF file", "write_file"),
    ("export all findings as a PDF report", "write_file"),
    ("generate charts based on the sales analysis", "write_file"),
])
def test_rendering_a_document_or_a_picture_is_not_a_file_write(
    need, must_not_suggest, stub_registry
):
    """`write_file` writes text. A need whose whole distinguishing word is PDF
    resolved to it because "file" gated the match — and reported `have`, so it
    blocked nothing and the workflow would have "exported" a PDF that is a text
    dump. Both are curated gaps now: this engine cannot render either."""
    res = resolve_capability(need)
    assert must_not_suggest not in [t.name for t in res.tools]
    assert res.curated_gap, "saying we cannot do it is the point"


@pytest.mark.parametrize("need", [
    "write a file to disk",
    "write the summary to a text file",
    "read a file from disk",
])
def test_ordinary_file_work_is_untouched(need, stub_registry):
    """The rendering rules sit *before* the filesystem ones — deliberately, since
    "export … as a PDF file" hits both — so this pins that ordering."""
    assert resolve_capability(need).status == "have"

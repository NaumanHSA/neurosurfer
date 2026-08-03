"""The resolution golden set — real needs, from real builds, with real answers.

Roadmap Stage 4. Nine live runs produced a set of capability phrases whose right
answer is *known*, because somebody watched a build get it wrong and then watched
it get it right. Those pairs were written down in prose in the build logs, where
nothing can check them. This file is them, executable.

Every record cites the log line it came from. The citation is the point: when one
of these fails, the question is not "is the assertion still reasonable" but "did
the behaviour a live run pinned down actually change".

## Why this asserts `resolve_capability` and nothing below it

Roadmap Stage 3 (Registry Phase 6) **deletes** `_live_hits`, `search_catalog`'s
scoring, and the verb/object heuristics — every one of them a patch over a fact
the manifest now states. A golden set written against those internals would die
in the exact refactor it exists to protect.

So the surface here is `resolve_capability()`, the public entry point the
Architect itself goes through — `planner.py:286`, `agent/session.py:294` and the
`find_capability` tool at `agent/tools.py:720` all call it and nothing else. Its
inputs (a plain-English need) and its outputs (`Resolution.status`,
`.tools`, `.to_dict()`) are the contract Phase 6 keeps; the ladder underneath is
what Phase 6 is free to replace. `test_the_golden_set_is_what_the_architect_sees`
pins that anchoring in place.

## Determinism

`_empty_registry` is autouse: with nothing local, resolution asks a discovery
source, and the default one is the public MCP registry over the network. Every
test here binds a source that answers, so `none` means "nothing provides this"
rather than "the internet was slow". No MCP server, no database, no model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from neurosurfer.architect.capability import resolve_capability
from neurosurfer.mcp.sources import Capability as SourceCapability
from neurosurfer.mcp.sources import use_source

# ── discovery sources, stubbed ─────────────────────────────────────────────────


class _Source:
    """The smallest thing `search_servers` will talk to."""

    id = "golden"
    label = "Golden"
    blurb = ""
    capabilities = frozenset()
    requires_key = False

    def __init__(self, hits=()):
        self._hits = list(hits)
        self.asked: list[str] = []

    def available(self) -> bool:
        return True

    def key_set(self) -> bool:
        return False

    def search(self, query, *, limit=20, cursor=""):
        self.asked.append(query)
        return list(self._hits)


@pytest.fixture(autouse=True)
def _empty_registry():
    """No server provides anything, unless a test says otherwise."""
    with use_source(_Source()):
        yield


@pytest.fixture(autouse=True)
def _no_stray_live_tools():
    """A live MCP tool leaked by another module would change every answer here."""
    from neurosurfer.tools.registry import clear_live_tools

    clear_live_tools()
    yield
    clear_live_tools()


@pytest.fixture()
def registry_with():
    """Bind a source answering with the given `(name, description)` servers."""
    from contextlib import ExitStack

    from neurosurfer.mcp.registry import RegistryHit

    with ExitStack() as stack:

        def _factory(*servers: tuple[str, str], semantic: bool = False) -> _Source:
            source = _Source([RegistryHit(name=n, description=d) for n, d in servers])
            if semantic:
                source.capabilities = frozenset({SourceCapability.SEMANTIC_SEARCH})
            stack.enter_context(use_source(source))
            return source

        yield _factory


# ── the golden set ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Recorded:
    """One need→outcome pair, as a build log recorded it."""

    source: str                       # the log and line it is quoted from
    need: str
    status: str                       # have | none | not_external | installable
    providers: tuple[str, ...] = ()    # tools the log named
    exact: bool = False                # the log wrote a closed list, no "…"
    forbidden: tuple[str, ...] = ()    # what the *failing* run offered instead
    note: str = ""

    @property
    def id(self) -> str:
        return f"{self.source} · {self.need[:56]}"


#: Every need→outcome pair quoted in a build log, in log order.
#:
#: `providers` is what the log named. A trailing "…" in the log means the list
#: was truncated for the write-up, so membership is asserted; a closed list means
#: `exact`. `forbidden` is what the run that *failed* was handed instead — the
#: half of a golden record that catches a regression, since a wrong tool resolves
#: to `have` and therefore blocks nothing.
GOLDEN: tuple[Recorded, ...] = (
    # ── TOOL_REGISTRY_PROGRESS.md:69-76 ────────────────────────────────────────
    # "Measured on the needs that produced real failures" — the block that proved
    # tag-based resolution after the four SQL tools landed.
    Recorded(
        source="registry log:70",
        need="execute parameterized SQL queries against the database",
        status="have",
        providers=("sql",),
    ),
    Recorded(
        source="registry log:71",
        need="connect to a SQL Server database",
        status="have",
        providers=("sql",),
    ),
    Recorded(
        source="registry log:72",
        need="read a file from disk",
        status="have",
        providers=("read_file",),
        exact=True,
    ),
    Recorded(
        source="registry log:73",
        need="generate chart images for the report",
        status="none",
        note="`chart.render` is in the vocabulary and nothing claims it",
    ),
    Recorded(
        source="registry log:74",
        need="export the report as a PDF document",
        status="none",
        note="`pdf.render` is declared before anything provides it, on purpose",
    ),
    Recorded(
        source="registry log:75",
        need="send an email to the team",
        status="none",
        note="`message.send` — a named gap, not a search that came up short",
    ),
    # ── ARCHITECT_V4_PROGRESS.md:562-566 ───────────────────────────────────────
    # "Measured on the exact needs that produced the block" — Phase 12, after six
    # runs of resolver fixes turned out to be treating a missing tool.
    Recorded(
        source="v4 log:562",
        need="execute parameterized SQL queries",
        status="have",
        providers=("sql",),
    ),
    Recorded(
        source="v4 log:563",
        need="run SQL queries against the metadata",
        status="have",
        providers=("sql",),
    ),
    Recorded(
        source="v4 log:564",
        need="connect to a SQL Server database",
        status="have",
        providers=("sql",),
    ),
    Recorded(
        source="v4 log:565",
        need="read environment variables at runtime",
        status="not_external",
        note="the block that offered a Swiss air-quality index; a credential is "
             "not a step",
    ),
    Recorded(
        source="v4 log:566",
        need="write a file to the working directory",
        status="have",
        providers=("write_file",),
        exact=True,
    ),
    # ── ARCHITECT_V4_PROGRESS.md:19-21 ─────────────────────────────────────────
    # Phase 1, "a tool whose name is a common noun answers for everything". All
    # three resolved `have, tools=['data']` and therefore stopped nothing.
    Recorded(
        source="v4 log:19",
        need="export data as PDF",
        status="none",
        forbidden=("data",),
        note="`data` inspects a local CSV/JSON/SQLite file and cannot make a PDF",
    ),
    Recorded(
        source="v4 log:20",
        need="query customer revenue data",
        status="none",
        forbidden=("data",),
    ),
    Recorded(
        source="v4 log:21",
        need="query sales data for comparison",
        status="none",
        forbidden=("data",),
    ),
    # ── ARCHITECT_V4_PROGRESS.md:43-44 ─────────────────────────────────────────
    # Phase 1, "a composition verb is not a composition". Both were `not_external`
    # on the strength of the verb, and the plan then held a `tool` node with no tool.
    Recorded(
        source="v4 log:43",
        need="generate visual charts",
        status="none",
        note="charts are matplotlib, not thinking",
    ),
    Recorded(
        source="v4 log:44",
        need="analyze product sales data",
        status="none",
        note="analysing a database is not thinking either",
    ),
    # ── ARCHITECT_V4_PROGRESS.md:379 ───────────────────────────────────────────
    # Phase 9/10's "still open after this": the outcome the next run was to check.
    Recorded(
        source="v4 log:379",
        need="assemble a one page PDF report",
        status="none",
        note="resolving to nothing is correct; it is resolving to the wrong thing "
             "that built fiction",
    ),
    # ── ARCHITECT_V3_PROGRESS.md:216-221 ───────────────────────────────────────
    # The live gpt-4o-mini failure that created `_COMPOSITION_VERBS`: the model
    # reported *drafting emails* as a credentialled blocker.
    Recorded(
        source="v3 log:216",
        need="draft an email",
        status="not_external",
        note="composing text is the one thing a `base` node is for",
    ),
    # ── TOOL_REGISTRY_PROGRESS.md:549-553 ──────────────────────────────────────
    # The credential guard, checked before the taxonomy because such a phrase
    # names the thing it is a credential *for*.
    Recorded(
        source="registry log:551",
        need="read env vars to fetch the SQL Server connection details",
        status="not_external",
        note="names a database and is still not a step",
    ),
)


def _resolve(record: Recorded):
    return resolve_capability(record.need)


@pytest.mark.parametrize("record", GOLDEN, ids=[r.id for r in GOLDEN])
def test_the_recorded_status_still_holds(record: Recorded):
    """The status a live build was measured at, asserted at the public entry point."""
    res = _resolve(record)
    assert res.status == record.status, (
        f"{record.source} recorded {record.status!r}, got {res.status!r}\n"
        f"{record.note}\n{res.render()}"
    )


@pytest.mark.parametrize(
    "record",
    [r for r in GOLDEN if r.providers],
    ids=[r.id for r in GOLDEN if r.providers],
)
def test_the_recorded_providers_still_answer(record: Recorded):
    """Which tools answered, not just that something did.

    A status alone would pass on the wrong tool — which is precisely the failure
    mode of half these records: `have` from `data` stopped nothing.
    """
    names = [t.name for t in _resolve(record).tools]
    if record.exact:
        assert names == list(record.providers), record.source
    else:
        assert set(record.providers) <= set(names), (
            f"{record.source} recorded {record.providers}, got {names}"
        )


@pytest.mark.parametrize(
    "record",
    [r for r in GOLDEN if r.forbidden],
    ids=[r.id for r in GOLDEN if r.forbidden],
)
def test_the_tool_that_lied_is_not_offered_again(record: Recorded):
    """A wrong tool is worse than none: it resolves `have`, so nothing blocks."""
    names = {t.name for t in _resolve(record).tools}
    assert not (names & set(record.forbidden)), (
        f"{record.source}: {names & set(record.forbidden)} answered again"
    )


# ── the anchoring test: this is the surface a build goes through ───────────────


async def test_the_golden_set_is_what_the_architect_sees(tmp_path):
    """One record per status, resolved through the Architect's own tool.

    This is what stops the golden set drifting into a test of a helper nobody
    calls. `find_capability` is the Architect's entry to resolution and it goes
    through `resolve_capability`; if that stops being true, this fails and the
    file above stops meaning what it claims.
    """
    from neurosurfer.architect.agent import BuildSession, architect_tools
    from neurosurfer.architect.knowledge import KnowledgeBase
    from neurosurfer.graph.workflow.registry import WorkflowRegistry
    from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

    session = BuildSession(
        intent="audit the access database",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=KnowledgeBase(),
    )
    ctx = ToolContext(cwd=tmp_path, io=AutoApproveIOHandler())
    find = next(t for t in architect_tools(session) if t.name == "find_capability")

    sample = [
        next(r for r in GOLDEN if r.status == "have"),
        next(r for r in GOLDEN if r.status == "none"),
        next(r for r in GOLDEN if r.status == "not_external"),
    ]
    for record in sample:
        result = await find.run({"capability": record.need}, ctx)
        assert not result.is_error, result.content

    assert [r.status for r in session.resolutions] == [r.status for r in sample]
    assert [r.need for r in session.resolutions] == [r.need for r in sample]


def test_the_serialized_form_carries_the_same_answer():
    """`to_dict()` is what lands on a plan step and on the build record — the
    shape every consumer downstream of resolution actually reads."""
    for record in GOLDEN:
        payload = _resolve(record).to_dict()
        assert payload["status"] == record.status, record.source
        assert payload["need"] == record.need
        if record.exact:
            assert [t["name"] for t in payload["tools"]] == list(record.providers)


# ── the audit run's provider set (registry log:137-142) ────────────────────────


@pytest.mark.parametrize("need", [
    "connect to a SQL Server database",
    "execute parameterized SQL queries against the database",
    "run SQL queries against the metadata",
])
def test_a_database_need_offers_the_four_sql_tools_and_nothing_else(need):
    """The first live run after Phases 1–2: *"resolving 3 external capabilities →
    all `have` → use `sql_query`, `sql_list_tables`, `sql_table_schema`,
    `sql_test_connection`"*, with **zero registry searches**. Tags answered
    everything, and that is the claim Stage 3 must not quietly undo.
    """
    res = resolve_capability(need)
    assert res.status == "have"
    assert {t.name for t in res.tools} == {"sql"}
    assert not res.registry_searched, "a satisfied capability must not cost a search"


def test_a_database_need_never_answers_with_the_local_file_inspector():
    """`data` is *"inspect or query a local structured-data file"*. There has
    never been a tool that reaches a database server until the SQL four, and
    every database workflow before them was forced into MCP.
    """
    for need in ("query a database", "connect to a SQL Server database",
                 "query customer revenue data"):
        assert "data" not in {t.name for t in resolve_capability(need).tools}, need


# ── a server outliving its build (v4 log:295-319) ──────────────────────────────


def _live_tool(name: str, description: str):
    """An MCP tool as `live_tools()` returns it — no capability tags, because a
    published server cannot declare any."""
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


#: The tools `thinair/data` actually exposed, with the publisher's own wording.
#: Installed for Postgres in run 5, still connected in run 6, and by then simply
#: part of the catalog — with none of the "installed for this need" evidence the
#: loose match was trading on.
_THINAIR = (
    ("generate_migration",
     "Generate a SQL migration file from a described schema change, using the "
     "connection's dialect."),
    ("generate_seed_data", "Generate realistic seed data rows for a table."),
    ("saved_queries", "Manage a library of saved SQL queries for a connection."),
    ("suggest_queries",
     "Suggest useful SQL queries to run against a table, and generate them."),
    ("query_sql",
     "Execute a read-only SQL query against a registered PostgreSQL, MySQL or "
     "SQL Server database connection."),
)


@pytest.fixture()
def thinair_connected():
    from neurosurfer.tools.registry import clear_live_tools, set_live_tools

    set_live_tools([_live_tool(n, d) for n, d in _THINAIR])
    try:
        yield
    finally:
        clear_live_tools()


@pytest.mark.parametrize(("need", "why"), [
    ("Generate chart images using a charting library and produce files/images "
     "suitable for embedding in a PDF.",
     "`generate_migration` scored 5.0 on the word **generate**, and a *query* "
     "library matched a *charting* library"),
    ("Render HTML/markdown/text plus images into a PDF using a PDF generation "
     "toolchain.",
     "`find_n_plus_one` was matching **plus**, the preposition"),
])
def test_a_database_server_does_not_answer_for_charts_or_pdfs(
    need, why, thinair_connected
):
    """Run 6 built five nodes to draw charts with database-migration tools.

    Recorded as `have, ['generate_migration', 'generate_seed_data',
    'saved_queries']` — so nothing blocked and nothing offered an install.
    """
    res = resolve_capability(need)
    assert res.status == "none", f"{why}\n{res.render()}"
    assert [t.name for t in res.tools] == []


def test_the_database_server_does_not_shadow_the_tools_that_reach_localhost(
    thinair_connected
):
    """The other half, and the one Stage 3 changes the mechanism for.

    V4 Phase 11 left this open: *"a connected hosted gateway still shadows a
    local tool that would actually work"* — `query_sql` answered "connect to a
    SQL Server database" and no install was offered, while thinair cannot reach
    `localhost:1433`. Tag-based resolution settled it in the right direction
    already: the tools that declare `db.*` answer, and a tagless MCP tool does
    not. Stage 3 moves `runtime` into the search stage; the *answer* below is
    what must not change when it does.
    """
    res = resolve_capability("connect to a SQL Server database")
    assert res.status == "have"
    assert {t.name for t in res.tools} == {"sql"}


def test_writing_a_csv_is_still_a_file_write_with_a_server_connected(thinair_connected):
    """Tightening must not take the ordinary cases with it."""
    res = resolve_capability(
        "Write CSV files to the local filesystem as additional artifacts."
    )
    assert res.status == "have"
    assert [t.name for t in res.tools] == ["write_file"]


# ── what a blocked build is told (v4 log:583-597) ──────────────────────────────


def test_an_installable_capability_offers_authoring_as_well_as_installing(
    registry_with
):
    """`author_tool` appeared in **one** of `NEXT:`'s three branches — `none` —
    so a single junk registry match hid authoring completely. The
    environment-variable build was offered a UK flood-data server, declined it
    correctly, and blocked, never having been told writing the tool was on the
    table.
    """
    registry_with(("io.github.acme/charts", "Render charts as PNG images."))
    res = resolve_capability("generate chart images for the report")

    assert res.status == "installable"
    rendered = res.render()
    assert "author_tool" in rendered
    assert "install_mcp_server" in rendered
    assert "declare_blocked" in rendered


def test_a_curated_gap_does_not_tell_the_model_not_to_write_a_tool():
    """The second suppressor: on a `curated_gap` the report said it needs an
    external integration *"NOT a catalog tool"* — a sentence that was wrong about
    databases for the whole life of the project, and reads as an instruction.
    """
    res = resolve_capability("export the report as a PDF document")
    assert res.curated_gap
    rendered = res.render()
    assert "not a catalog tool" not in rendered.lower()
    assert "author_tool" in rendered


def test_a_gap_the_taxonomy_knows_is_stronger_than_a_search_that_found_nothing():
    """`curated_gap` is a claim, not a shrug — it is what turns "no match" into an
    authorable job. The vocabulary declares `chart.render` and `pdf.render`
    before anything provides them precisely so this can be said.
    """
    for need in ("generate chart images for the report",
                 "export the report as a PDF document",
                 "send an email to the team"):
        assert resolve_capability(need).curated_gap, need


def test_the_curated_gap_decides_whether_a_plan_step_blocks():
    """Why the flag above is asserted at all, and not treated as trivia.

    `PlanStep.reachable` is resolution's real consumer: `none` **with** a curated
    gap is the only thing that stops a build, and `none` without one is "author
    it". So a `curated_gap` that quietly stops being set does not fail a status
    assertion — it turns a blocked build into one that carries on and writes a
    tool for something the taxonomy already knows nothing can do.
    """
    from neurosurfer.architect.plan import PlanStep

    def _step(need: str) -> PlanStep:
        return PlanStep(id="s", intent=need, is_external=True,
                        needed_capability=need,
                        resolution=resolve_capability(need).to_dict())

    blocked = _step("export the report as a PDF document")
    assert blocked.resolution["status"] == "none" and blocked.resolution["curated_gap"]
    assert not blocked.reachable, "a named gap is what a build is allowed to stop on"

    open_ended = _step("do something nobody has ever needed")
    assert open_ended.resolution["status"] == "none"
    assert open_ended.reachable, "an unrecognised need may still be authorable"


# ── where the record and the behaviour do not quite meet ──────────────────────
#
# Both of these pass. They assert the part of the record that is unambiguous and
# stop short of the part that is not, and each names the divergence, because a
# test quietly widened to fit is how a golden set stops being one.


def test_obtaining_a_credential_from_a_real_file_is_not_engine_provided():
    """`capability.py`'s own worked example, and the reason `_CREDENTIAL_MEDIA`
    exists: *"Load the credentials from creds.json IS a file read; the exclusion
    keeps this guard from swallowing one."*

    The guard does decline — that is what is asserted. What the docstring implies
    next does **not** hold: the phrase resolves to `none`, not to `read_file`,
    because "creds"/"json" share no object token with any tool name. Say the same
    thing with the word *file* or *disk* in it and the taxonomy catches it.
    """
    res = resolve_capability("load the credentials from creds.json")
    assert res.status != "not_external", "a real medium is not an engine-provided value"

    # The phrasing that does resolve, for contrast.
    spelled_out = resolve_capability("read the API key from the config file on disk")
    assert spelled_out.status == "have"
    assert [t.name for t in spelled_out.tools] == ["read_file"]


def test_pdf_assembly_resolves_to_nothing_however_it_is_phrased():
    """v4 log:379 — *"`assemble_one_page_pdf` and the chart steps will now resolve
    to **nothing** rather than to the wrong thing"*. It does.

    What differs between phrasings is the *strength* of that nothing: "export the
    report as a PDF document" is a curated gap and blocks the plan, while
    "assemble a one page PDF report" is an ordinary miss and is `reachable`. Same
    capability, two answers to "should this build stop", decided by wording.
    """
    for need in ("assemble a one page PDF report",
                 "assemble_one_page_pdf",
                 "export the report as a PDF document"):
        assert resolve_capability(need).status == "none", need

    assert not resolve_capability("assemble a one page PDF report").curated_gap
    assert resolve_capability("export the report as a PDF document").curated_gap


# ── the registry leg, when something does provide it (v3 log:238-246) ──────────


def test_an_email_capability_is_installable_when_a_server_exists(registry_with):
    """Live, on gpt-4o-mini with nothing configured: *"access my gmail and read
    emails → installable"*, off the real registry. The catalog has nothing, the
    taxonomy says so, and the registry is asked exactly then.
    """
    source = registry_with(
        ("io.github.mindstone/mcp-server-email-imap",
         "Read email over IMAP: gmail, inbox, folders and messages."),
    )
    res = resolve_capability("access my gmail and read emails")

    assert res.status == "installable"
    assert [s.name for s in res.servers] == ["io.github.mindstone/mcp-server-email-imap"]
    assert res.registry_searched
    # Curated terms lead: a keyword index answers "gmail" and not the phrase.
    assert source.asked and "gmail" in [t.lower() for t in source.asked]


def test_a_messaging_capability_is_installable_when_a_server_exists(registry_with):
    """*"send a text message → installable"*, `io.github.pipeworx-io/twilio`."""
    registry_with(("io.github.pipeworx-io/twilio",
                   "Twilio MCP — send an sms text message and make calls."))
    res = resolve_capability("send a text message")
    assert res.status == "installable"
    assert [s.name for s in res.servers] == ["io.github.pipeworx-io/twilio"]


def test_a_satisfied_capability_never_reaches_the_registry(registry_with):
    """Offering an install beside a tool that already does the job invites the
    expensive path for no reason — and the golden `have` records above would
    stop meaning anything if it happened.
    """
    source = registry_with(("io.github.acme/anything", "Does everything."))
    for record in GOLDEN:
        if record.status != "have":
            continue
        res = resolve_capability(record.need)
        assert not res.registry_searched, record.source
    assert source.asked == []


# ── the golden set is a fixed set, not a moving one ───────────────────────────


def test_every_record_cites_a_build_log():
    """A record with no citation is an opinion, and this file has no room for
    one — the whole value of these pairs is that a live run produced them."""
    for record in GOLDEN:
        assert record.source, record.need
        assert record.status in {"have", "none", "not_external", "installable"}
        assert record.need.strip() == record.need and record.need


def test_the_set_covers_every_status_a_build_can_be_told():
    """`installable` is covered by the registry tests above, which need a source;
    the parametrized records cover the three a bare catalog can produce."""
    assert {r.status for r in GOLDEN} == {"have", "none", "not_external"}

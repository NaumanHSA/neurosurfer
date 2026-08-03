"""The registry: what a tool declares about itself, and what that buys.

The drift tripwires here are the point. A capability tag outside the vocabulary,
a tool declaring secrets with no way to obtain them, a manifest that no longer
describes its tool — each is the kind of thing that used to be discovered by a
model picking the wrong tool during a live build.
"""

from __future__ import annotations

import pytest

from neurosurfer.registry import (
    CAPABILITIES,
    ToolManifest,
    manifest_for,
    manifests,
    providers_of,
    unsatisfied_capabilities,
)
from neurosurfer.registry.capabilities import unknown

# ── every registered tool describes itself legally ──────────────────────────────

def test_every_manifest_is_well_formed():
    bad = {m.name: m.problems() for m in manifests(workflow_only=False) if m.problems()}
    assert not bad, f"malformed manifests: {bad}"


def test_no_tool_invents_a_capability_tag():
    """A closed vocabulary is the whole design. Two tools tagged `db.query` and
    `sql.run` would need a scorer to reconcile — and a scorer guessing at tags is
    exactly what this layer replaces."""
    for m in manifests(workflow_only=False):
        assert not unknown(m.capabilities), f"{m.name} invents {unknown(m.capabilities)}"


def test_a_tool_needing_a_credential_says_how_to_get_one():
    """A field name with no explanation is what a build guessed seven of."""
    for m in manifests(workflow_only=False):
        if m.secret_inputs:
            assert m.credential_help, f"{m.name} asks for a secret and does not say what"


def test_every_capability_in_the_vocabulary_is_described():
    for tag, text in CAPABILITIES.items():
        assert text and text.endswith("."), f"{tag} needs a sentence saying what it covers"


# ── the facts that resolution now depends on ───────────────────────────────────

def test_the_database_tools_declare_what_they_do():
    names = {m.name for m in providers_of("db.query")}
    assert "sql" in names
    assert {m.name for m in providers_of("db.schema")} >= {"sql"}


# ── a tool is a type; what it does are its operations ──────────────────────────

def test_a_tag_resolves_to_a_tool_and_then_to_an_operation():
    """Two levels, one lookup path.

    `providers_of` answers *which tool* — the question resolution has always
    asked — and `operations_for` then answers *which operation to call*. Before
    this, `sql_query` and `sql_table_schema` were separate top-level tools whose
    only relationship was shared private helpers in one module: real in the code
    and invisible to everything above it.
    """
    sql = next(m for m in providers_of("db.query") if m.name == "sql")
    assert [o.name for o in sql.operations_for("db.query")] == ["query"]
    assert [o.name for o in sql.operations_for("db.schema")] == [
        "list_tables", "table_schema"]
    assert [o.name for o in sql.operations_for("db.connect")] == ["test_connection"]


def test_a_tools_capabilities_are_the_union_of_its_operations():
    """`covers()` must keep working unchanged, or resolution would need a second
    lookup path for tools that happen to have operations."""
    sql = next(m for m in manifests() if m.name == "sql")
    assert sql.capabilities == {"db.connect", "db.schema", "db.query"}
    assert all(sql.covers(t) for o in sql.operations for t in o.capabilities)


def test_each_operation_carries_its_own_schema():
    """The reason operations exist at all: a configuration dialog shows the
    fields *that* operation needs, not the union of every operation's with
    everything optional."""
    sql = next(m for m in manifests() if m.name == "sql")
    assert sql.operation("table_schema").required_inputs == ["dsn", "table"]
    assert sql.operation("list_tables").required_inputs == ["dsn"]
    assert set(sql.operation("query").required_inputs) == {"dsn", "query"}
    # The tool's own surface stays permissive, or `list_tables` would be
    # uncallable for want of a `query`.
    assert sql.required_inputs == ["dsn", "operation"]


def test_an_operation_may_not_claim_a_tag_the_tool_does_not():
    """Resolution matches the tool on the union, so an operation tagged with
    something outside it would be unreachable — a capability nothing could find."""
    from neurosurfer.registry.manifest import OperationManifest

    bad = ToolManifest(
        name="x", description="d",
        capabilities=frozenset({"db.query"}),
        operations=(OperationManifest(
            name="render", description="d", capabilities=frozenset({"pdf.render"})),),
    )
    assert any("does not" in p for p in bad.problems())


def test_a_single_purpose_tool_declares_no_operations():
    """`http` and `web_search` must not grow an 'Operation: [one choice]'
    selector. An empty tuple is the honest answer, not an invented one-item list."""
    for name in ("http", "web_search", "read_file"):
        m = next(x for x in manifests() if x.name == name)
        assert m.operations == (), name


def test_a_file_inspector_does_not_claim_a_database():
    """`data` reads CSV/JSON/SQLite *files*. Answering a database need with it is
    the failure that started this whole effort — and it is now impossible by
    declaration rather than avoided by a scoring heuristic."""
    data = next(m for m in manifests(workflow_only=False) if m.name == "data")
    assert data.covers("data.inspect")
    assert not data.covers("db.query")


def test_gaps_are_named_rather_than_filled_with_the_nearest_thing():
    """Nothing here renders a chart or a PDF. Saying so is what turns 'no match'
    into a job — author one, import a server — instead of an invitation to use
    whatever shared the most words with the request."""
    gaps = set(unsatisfied_capabilities())
    assert {"chart.render", "pdf.render"} <= gaps
    assert not providers_of("chart.render")


def test_local_tools_can_reach_localhost_and_hosted_ones_cannot():
    """The single field behind an afternoon lost to a hosted SQL gateway being
    offered for a database running in a container on the user's own machine."""
    sql = next(m for m in manifests() if m.name == "sql")
    assert sql.reaches_localhost

    hosted = ToolManifest(name="x", description="d", runtime="hosted")
    assert not hosted.reaches_localhost
    assert hosted not in providers_of("db.query", reaches_localhost=True)


# ── the move must be invisible from outside ────────────────────────────────────

def test_the_old_import_path_still_works():
    """Tools moved to `registry/core/<domain>/`. An internal reorganisation must
    not become a migration for anybody."""
    from neurosurfer.tools import builtin

    for name in builtin.__all__:
        assert hasattr(builtin, name), f"{name} lost in the move"


def test_a_moved_tool_is_the_same_tool():
    from neurosurfer.registry.core.filesystem.read_file import ReadFileTool as Moved
    from neurosurfer.tools.builtin import ReadFileTool as ViaShim

    assert Moved is ViaShim


# ── an unknown tool degrades honestly rather than crashing ─────────────────────

def test_a_tool_that_declares_nothing_still_yields_a_manifest():
    """An MCP tool cannot declare any of this. It must still be describable —
    as what it is: no capabilities, no credential story, and therefore never a
    match on a tag."""

    class _Bare:
        name = "mystery"
        description = "Does something."
        is_mcp = True

        @property
        def schema(self):
            raise RuntimeError("no schema here")

    m = manifest_for(_Bare())
    assert m.name == "mystery"
    assert m.origin == "mcp"
    assert m.capabilities == frozenset()
    assert m.verified is None


@pytest.mark.parametrize("field", ["origin", "runtime"])
def test_an_illegal_enum_is_reported_not_swallowed(field):
    m = ToolManifest(name="x", description="d", **{field: "nonsense"})
    assert any("nonsense" in p for p in m.problems())


# ── identity: what a person calls a tool, and what it looks like ───────────────
#
# `sql` is a good identifier and a poor label. The palette showed the identifier
# because there was nothing else to show, so a user was asked to learn the
# engine's vocabulary in order to pick a database tool. Title and icon are the
# two facts that fix it, and both are declared next to the code like everything
# else in the manifest.

def test_every_tool_has_a_title_and_an_icon_that_exists():
    from neurosurfer.registry import icon_slugs

    have = icon_slugs()
    for m in manifests(workflow_only=False):
        assert m.title, f"{m.name} has no title"
        assert m.icon in have, f"{m.name} resolved to icon {m.icon!r}, which is not shipped"


def test_a_title_is_derived_when_a_tool_declares_none():
    from neurosurfer.registry import humanise

    # The fallback has to be good enough that declaring a title stays optional —
    # every MCP tool relies on it, since an MCP tool cannot declare anything.
    assert humanise("apply_edit") == "Apply Edit"
    assert humanise("list_dir") == "List Directory"
    # Initialisms are the one case sentence-casing gets visibly wrong.
    assert humanise("http") == "HTTP"
    assert humanise("sql_query") == "SQL Query"
    assert humanise("") == ""


def test_a_declared_title_wins_over_the_derivation():
    # Because no derivation from snake_case can produce a product's own name.
    sql = next(m for m in manifests(workflow_only=False) if m.name == "sql")
    assert sql.title == "SQL Database"
    assert sql.title != "SQL"


def test_an_icon_falls_back_to_the_family_before_it_falls_back_to_generic():
    """The tier that matters: an MCP tool nobody has ever seen still reads as
    *database* rather than as a generic box."""
    from neurosurfer.registry.icons import resolve_icon

    class _Unseen:
        name = "totally_unheard_of_tool"
        description = "?"

    # No artwork of its own and no registry domain → the last resort, which is
    # a real file rather than an empty string.
    assert resolve_icon(_Unseen()) == "tool"

    # A tool living in the database domain inherits that family's icon.
    _Unseen.__module__ = "neurosurfer.registry.core.database.somewhere"
    assert resolve_icon(_Unseen()) == "database"


def test_the_icon_route_cannot_be_talked_out_of_its_own_directory():
    # Reached from HTTP, where `../../etc/passwd` is a path that resolves fine.
    from neurosurfer.registry import icon_bytes

    assert icon_bytes("sql") is not None
    assert icon_bytes("sql.svg") is not None
    assert icon_bytes("../../../etc/passwd") is None
    assert icon_bytes("nope") is None


def test_every_operation_has_a_title_too():
    sql = next(m for m in manifests(workflow_only=False) if m.name == "sql")
    titles = {o.name: o.title for o in sql.operations}
    assert titles["query"] == "Run a query"
    # `table_schema` is exactly the case the derivation reads badly for.
    assert titles["table_schema"] == "Describe a table"

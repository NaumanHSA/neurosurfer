"""The SQL tool — read-only access to a real database server, as four operations.

No server is required: these pin the guarantees that hold before a connection is
ever attempted, which are the ones that matter for safety.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.registry.core.database.sql_tools import (
    SqlTool,
    _redacted,
)
from neurosurfer.tools.base import ToolContext

DSN = "postgresql+psycopg://user:hunter2@db.example/app"


@pytest.fixture
def ctx():
    return ToolContext(cwd=Path.cwd(), io=None)


def _args(**kw):
    return SqlTool.input_model(dsn=DSN, **kw)


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "DELETE FROM users",
    "UPDATE users SET admin = 1",
    "INSERT INTO users VALUES (1)",
    "DROP TABLE users",
    "TRUNCATE TABLE users",
    "  update users set x = 1  ",
])
async def test_writes_are_refused_before_any_connection(query, ctx):
    """The check runs on the statement, so it holds even with no server at all."""
    res = await SqlTool().call(_args(operation="query", query=query), ctx)
    assert res.is_error
    assert "read-only" in res.content


@pytest.mark.asyncio
async def test_stacked_statements_are_refused(ctx):
    """`SELECT 1; DROP TABLE users` is a read by prefix and a write in fact."""
    res = await SqlTool().call(
        _args(operation="query", query="SELECT 1; DROP TABLE users"), ctx)
    assert res.is_error and "single statement" in res.content


@pytest.mark.asyncio
async def test_an_empty_query_is_refused(ctx):
    res = await SqlTool().call(_args(operation="query", query="   "), ctx)
    assert res.is_error


def test_the_password_never_survives_into_an_error():
    """A connection error quotes the URL it failed on, and the URL holds the
    password — the usual way a credential reaches a log."""
    out = _redacted(DSN)
    assert "hunter2" not in out
    assert "db.example" in out, "the useful half must survive"


def test_redaction_never_raises_on_a_malformed_dsn():
    assert _redacted("not a url at all") == "<dsn>"


@pytest.mark.asyncio
async def test_a_missing_driver_names_the_package_to_install(ctx):
    """The commonest first failure. 'ModuleNotFoundError' is not an action."""
    res = await SqlTool().call(
        SqlTool.input_model(
            operation="test_connection",
            dsn="mssql+pyodbc://u:p@h/db?driver=nope"), ctx)
    if res.is_error and "not installed" in res.content:
        assert "pip install" in res.content


# ── operations ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_missing_operation_argument_is_named_before_the_driver(ctx):
    """`table_schema` needs a table.

    The tool's own schema cannot require it — `list_tables` would become
    uncallable — so `call()` re-validates against the operation's own model.
    Without that, this reaches the driver and fails as something obscure.
    """
    res = await SqlTool().call(_args(operation="table_schema"), ctx)
    assert res.is_error
    assert "table_schema" in res.content


@pytest.mark.asyncio
async def test_an_unknown_operation_lists_the_real_ones(ctx):
    """Unreachable through the Literal, reachable from raw `tool_args`."""
    args = _args(operation="query", query="SELECT 1")
    object.__setattr__(args, "operation", "delete_everything")
    res = await SqlTool().call(args, ctx)
    assert res.is_error
    assert "table_schema" in res.content and "query" in res.content


def test_every_operation_is_read_only_and_says_so():
    """SQL stays read-only for now, so nothing here may answer otherwise.

    `Operation.read_only` is the seam a write operation would answer through,
    and the approval gates key off `is_read_only(args)` — so this is the test
    that has to be *changed*, deliberately, before a write can ever ship.
    """
    tool = SqlTool()
    assert set(tool.operations) == {
        "test_connection", "list_tables", "table_schema", "query"}
    for name, op in tool.operations.items():
        assert op.read_only, name
        assert tool.is_read_only(SqlTool.input_model(dsn=DSN, operation=name))


def test_the_operation_models_require_what_the_tool_cannot():
    """The point of a per-operation model: `query` is required *there*."""
    from neurosurfer.registry.core.database.sql_tools import QueryArgs, TableSchemaArgs

    assert "query" in (QueryArgs.model_json_schema().get("required") or [])
    assert "table" in (TableSchemaArgs.model_json_schema().get("required") or [])
    # ...and optional on the tool's own permissive surface.
    assert "query" not in (SqlTool.input_model.model_json_schema().get("required") or [])

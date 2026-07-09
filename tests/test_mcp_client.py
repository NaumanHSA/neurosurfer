"""Phase 1 (MCP client) tests.

Covers:
  - McpServerConfig / McpStore: CRUD, persistence, enabled filter, env expansion.
  - McpTool: raw-schema passthrough, annotation→flag mapping, call content mapping,
    isError mapping, transport-failure → ToolResult.error (never raises).
  - Permissions MCP gate: read-only passes, gated mutating call asks, open/denied.
  - McpManager: stdio connect + discovery publishes to the live registry, name
    de-duplication, one-server-down doesn't break others, aclose() clears tools.

The in-memory transport (mcp.shared.memory) gives a real ClientSession for the tool
unit tests; the manager is exercised against a real stdio FastMCP subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from neurosurfer.agents.runtime.permissions import Guardrails, Permissions
from neurosurfer.config.mcp import McpServerConfig
from neurosurfer.tools import registry as tool_registry
from tests.fakes import ScriptedIO

mcp = pytest.importorskip("mcp")
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session as connect  # noqa: E402

from neurosurfer.mcp.tool import McpTool  # noqa: E402
from neurosurfer.tools.base import ToolContext  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# A small in-memory server used by the tool tests
# ──────────────────────────────────────────────────────────────────────────────
def _build_server() -> FastMCP:
    server = FastMCP("fixture")

    @server.tool(annotations={"readOnlyHint": True})
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"sunny in {city}"

    @server.tool(annotations={"readOnlyHint": False, "destructiveHint": True})
    def delete_thing(name: str) -> str:
        """Delete a thing (mutating)."""
        return f"deleted {name}"

    @server.tool()
    def boom() -> str:
        """Always errors."""
        raise RuntimeError("kaboom")

    return server


def _ctx(io: ScriptedIO, tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=io)


async def _tools_from_session(session) -> dict[str, McpTool]:
    defs = (await session.list_tools()).tools
    return {
        d.name: McpTool.from_def(
            session=session, server_name="fixture", definition=d, exposed_name=d.name
        )
        for d in defs
    }


# ──────────────────────────────────────────────────────────────────────────────
# McpTool
# ──────────────────────────────────────────────────────────────────────────────
async def test_tool_schema_passthrough_and_flags(tmp_path: Path) -> None:
    server = _build_server()
    async with connect(server._mcp_server) as session:
        tools = await _tools_from_session(session)

        weather = tools["get_weather"]
        # raw JSON schema passes through (city property present, object type)
        schema = weather.schema.input_schema
        assert schema["type"] == "object"
        assert "city" in schema["properties"]
        assert weather.is_mcp is True

        # annotation → behaviour flags
        empty = weather.parse_args({})
        assert weather.is_read_only(empty) is True
        assert weather.is_destructive(empty) is False

        delete = tools["delete_thing"]
        assert delete.is_read_only(delete.parse_args({})) is False
        assert delete.is_destructive(delete.parse_args({})) is True


async def test_tool_call_maps_content(tmp_path: Path) -> None:
    server = _build_server()
    io = ScriptedIO()
    async with connect(server._mcp_server) as session:
        tools = await _tools_from_session(session)
        res = await tools["get_weather"].run({"city": "Lahore"}, _ctx(io, tmp_path))
        assert res.is_error is False
        assert "sunny in Lahore" in res.content


async def test_tool_call_error_is_returned_not_raised(tmp_path: Path) -> None:
    server = _build_server()
    io = ScriptedIO()
    async with connect(server._mcp_server) as session:
        tools = await _tools_from_session(session)
        res = await tools["boom"].run({}, _ctx(io, tmp_path))
        # MCP reports tool exceptions as isError results — surfaced, never raised.
        assert res.is_error is True


# ──────────────────────────────────────────────────────────────────────────────
# Permissions gate
# ──────────────────────────────────────────────────────────────────────────────
async def test_gate_readonly_passes_without_asking(tmp_path: Path) -> None:
    server = _build_server()
    io = ScriptedIO(approve_shell=False)  # would deny if asked
    async with connect(server._mcp_server) as session:
        tools = await _tools_from_session(session)
        perms = Permissions(Guardrails(mcp_policy="gated"), tmp_path)
        weather = tools["get_weather"]
        d = await perms.check("get_weather", weather.parse_args({}), _ctx(io, tmp_path), "default", tool=weather)
        assert d.allow is True
        assert io.shell_requests == []  # never asked


async def test_gate_mutating_asks(tmp_path: Path) -> None:
    server = _build_server()
    async with connect(server._mcp_server) as session:
        tools = await _tools_from_session(session)
        delete = tools["delete_thing"]
        args = delete.parse_args({"name": "x"})

        approve = ScriptedIO(approve_shell=True)
        d = await Permissions(Guardrails(mcp_policy="gated"), tmp_path).check(
            "delete_thing", args, _ctx(approve, tmp_path), "default", tool=delete
        )
        assert d.allow is True and approve.shell_requests == ["delete_thing"]

        deny = ScriptedIO(approve_shell=False)
        d = await Permissions(Guardrails(mcp_policy="gated"), tmp_path).check(
            "delete_thing", args, _ctx(deny, tmp_path), "default", tool=delete
        )
        assert d.allow is False


async def test_gate_open_and_denied(tmp_path: Path) -> None:
    server = _build_server()
    async with connect(server._mcp_server) as session:
        tools = await _tools_from_session(session)
        delete = tools["delete_thing"]
        args = delete.parse_args({"name": "x"})

        io = ScriptedIO(approve_shell=False)
        d_open = await Permissions(Guardrails(mcp_policy="open"), tmp_path).check(
            "delete_thing", args, _ctx(io, tmp_path), "default", tool=delete
        )
        assert d_open.allow is True

        d_denied = await Permissions(Guardrails(mcp_policy="denied"), tmp_path).check(
            "delete_thing", args, _ctx(io, tmp_path), "default", tool=delete
        )
        assert d_denied.allow is False


# ──────────────────────────────────────────────────────────────────────────────
# McpManager (real stdio subprocess)
# ──────────────────────────────────────────────────────────────────────────────
_SERVER_SCRIPT = '''
from mcp.server.fastmcp import FastMCP
server = FastMCP("stdio-fixture")

@server.tool(annotations={"readOnlyHint": True})
def ping() -> str:
    "ping"
    return "pong"

@server.tool(annotations={"readOnlyHint": True})
def add(a: int, b: int) -> int:
    "add two numbers"
    return a + b

if __name__ == "__main__":
    server.run("stdio")
'''


def _write_server(tmp_path: Path, name: str = "srv.py") -> Path:
    p = tmp_path / name
    p.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _clean_live_tools():
    tool_registry.clear_live_tools()
    yield
    tool_registry.clear_live_tools()


async def test_manager_connects_and_publishes(tmp_path: Path) -> None:
    from neurosurfer.mcp import McpManager

    script = _write_server(tmp_path)
    cfg = McpServerConfig(name="srv", transport="stdio", command=sys.executable, args=[str(script)])
    mgr = McpManager([cfg])
    try:
        statuses = await mgr.connect_all()
        assert len(statuses) == 1 and statuses[0].connected
        assert statuses[0].tool_count == 2
        # published to the global registry → visible from all_tools()
        live_names = {t.name for t in tool_registry.live_tools()}
        assert {"ping", "add"} <= live_names
        assert {"ping", "add"} <= {t.name for t in tool_registry.all_tools()}
    finally:
        await mgr.aclose()
    assert tool_registry.live_tools() == []  # cleared on close


async def test_manager_one_server_down_does_not_break_others(tmp_path: Path) -> None:
    from neurosurfer.mcp import McpManager

    good = McpServerConfig(
        name="good", transport="stdio", command=sys.executable, args=[str(_write_server(tmp_path))]
    )
    bad = McpServerConfig(name="bad", transport="stdio", command="this_command_does_not_exist_xyz")
    mgr = McpManager([good, bad])
    try:
        statuses = {s.name: s for s in await mgr.connect_all()}
        assert statuses["good"].connected is True
        assert statuses["bad"].connected is False and statuses["bad"].error
        assert {"ping", "add"} <= {t.name for t in tool_registry.live_tools()}
    finally:
        await mgr.aclose()


async def test_manager_name_dedup_across_servers(tmp_path: Path) -> None:
    from neurosurfer.mcp import McpManager

    s1 = McpServerConfig(name="a", transport="stdio", command=sys.executable, args=[str(_write_server(tmp_path, "a.py"))])
    s2 = McpServerConfig(name="b", transport="stdio", command=sys.executable, args=[str(_write_server(tmp_path, "b.py"))])
    mgr = McpManager([s1, s2])
    try:
        await mgr.connect_all()
        names = [t.name for t in mgr.tools()]
        assert len(names) == len(set(names))  # no duplicate exposed names
        assert "ping" in names and "b__ping" in names  # second server's clash namespaced
    finally:
        await mgr.aclose()

"""Tests for the general-task toolbelt + network gating (Pillar 0a).

Offline by design: the http/browse tools are exercised through their pure render
helpers and input guards (no real network), mirroring test_web_search.py. The data
tool runs fully against temp files. Network gating is driven through Permissions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from neurosurfer.agents.runtime.permissions import Guardrails, Permissions
from neurosurfer.tools import all_tools, default_pool
from neurosurfer.tools.base import ToolContext
from neurosurfer.tools.builtin.browse import BrowseTool
from neurosurfer.tools.builtin.data_tool import DataTool
from neurosurfer.tools.builtin.http_tool import HttpArgs, HttpTool

from .fakes import ScriptedIO


def ctx_for(tmp_path: Path, io: ScriptedIO | None = None) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=io or ScriptedIO())


# ── http tool ────────────────────────────────────────────────────────────────
def test_http_is_read_only_by_method():
    assert HttpTool().is_read_only(HttpArgs(url="https://x", method="GET")) is True
    assert HttpTool().is_read_only(HttpArgs(url="https://x", method="POST")) is False


@pytest.mark.asyncio
async def test_http_rejects_bad_scheme(tmp_path):
    res = await HttpTool().run({"url": "ftp://nope"}, ctx_for(tmp_path))
    assert res.is_error and "scheme" in res.content.lower()


@pytest.mark.asyncio
async def test_http_rejects_unserialisable_json_body(tmp_path):
    res = await HttpTool().run(
        {"url": "https://x", "method": "POST", "json_body": {1, 2}}, ctx_for(tmp_path)
    )
    assert res.is_error and "serial" in res.content.lower()


# ── data tool ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_data_csv_preview_and_count(tmp_path):
    (tmp_path / "p.csv").write_text("name,age\nalice,30\nbob,25\ncarol,40\n")
    res = await DataTool().run({"path": "p.csv", "limit": 2}, ctx_for(tmp_path))
    assert not res.is_error
    assert "name | age" in res.content
    assert "alice" in res.content
    assert "2 of 3 data rows shown" in res.content


@pytest.mark.asyncio
async def test_data_json_object_and_path_query(tmp_path):
    (tmp_path / "d.json").write_text('{"a": 1, "b": [10, 20]}')
    res = await DataTool().run({"path": "d.json"}, ctx_for(tmp_path))
    assert "object with 2 keys" in res.content
    q = await DataTool().run({"path": "d.json", "query": "b.1"}, ctx_for(tmp_path))
    assert q.content.strip().endswith("20")


@pytest.mark.asyncio
async def test_data_sqlite_select_and_refuses_writes(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO items VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()

    ok = await DataTool().run({"path": "t.db", "query": "SELECT * FROM items"}, ctx_for(tmp_path))
    assert not ok.is_error and "id | name" in ok.content and "a" in ok.content

    bad = await DataTool().run({"path": "t.db", "query": "DROP TABLE items"}, ctx_for(tmp_path))
    assert bad.is_error and "read-only" in bad.content.lower()

    noq = await DataTool().run({"path": "t.db"}, ctx_for(tmp_path))
    assert noq.is_error


@pytest.mark.asyncio
async def test_data_missing_and_unsupported(tmp_path):
    miss = await DataTool().run({"path": "nope.csv"}, ctx_for(tmp_path))
    assert miss.is_error and "not found" in miss.content.lower()
    (tmp_path / "x.bin").write_bytes(b"\x00\x01")
    bad = await DataTool().run({"path": "x.bin"}, ctx_for(tmp_path))
    assert bad.is_error and "unsupported" in bad.content.lower()


# ── browse tool (offline guards only) ────────────────────────────────────────
@pytest.mark.asyncio
async def test_browse_rejects_bad_scheme(tmp_path):
    res = await BrowseTool().run({"url": "ftp://nope"}, ctx_for(tmp_path))
    assert res.is_error and "scheme" in res.content.lower()


# ── network gating ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_network_policy_denied_blocks(tmp_path):
    d = await Permissions(Guardrails(network_policy="denied"), tmp_path).check(
        "http", HttpArgs(url="https://x"), ctx_for(tmp_path), "default"
    )
    assert d.allow is False


@pytest.mark.asyncio
async def test_network_policy_open_allows(tmp_path):
    d = await Permissions(Guardrails(network_policy="open"), tmp_path).check(
        "http", HttpArgs(url="https://x"), ctx_for(tmp_path), "default"
    )
    assert d.allow is True


@pytest.mark.asyncio
async def test_network_policy_gated_uses_shell_approval(tmp_path):
    yes = ScriptedIO(approve_shell=True)
    d = await Permissions(Guardrails(network_policy="gated"), tmp_path).check(
        "http", HttpArgs(url="https://api.x", method="GET"), ctx_for(tmp_path, yes), "default"
    )
    assert d.allow is True
    assert yes.shell_requests and "https://api.x" in yes.shell_requests[0]

    no = ScriptedIO(approve_shell=False)
    d2 = await Permissions(Guardrails(network_policy="gated"), tmp_path).check(
        "browse", HttpArgs(url="https://api.x"), ctx_for(tmp_path, no), "default"
    )
    assert d2.allow is False


@pytest.mark.asyncio
async def test_network_bypass_mode_skips_gate(tmp_path):
    no = ScriptedIO(approve_shell=False)
    d = await Permissions(Guardrails(network_policy="gated"), tmp_path).check(
        "http", HttpArgs(url="https://x"), ctx_for(tmp_path, no), "bypass"
    )
    assert d.allow is True


# ── registration ─────────────────────────────────────────────────────────────
def test_new_tools_registered_everywhere(tmp_path):
    names = set(default_pool().names())
    assert {"http", "data", "browse"} <= names
    assert {"http", "data", "browse"} <= {t.name for t in all_tools()}

"""Tests for the general-task toolbelt + network gating (Pillar 0a).

Offline by design: the http/browse tools are exercised through their pure render
helpers and input guards (no real network), mirroring test_web_search.py. The data
tool runs fully against temp files. Network gating is driven through Permissions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import pytest

from neurosurfer.agents.runtime.permissions import Guardrails, Permissions
from neurosurfer.tasks.definition import ALL_TOOLS
from neurosurfer.tasks.runner import build_full_pool
from neurosurfer.tools import all_tools, default_pool
from neurosurfer.tools.base import ToolContext
from neurosurfer.tools.builtin.browse import BrowseTool, _playwright_available
from neurosurfer.tools.builtin.data_tool import DataTool
from neurosurfer.tools.builtin.http_tool import HttpArgs, HttpTool

from .fakes import ScriptedIO


def ctx_for(tmp_path: Path, io: ScriptedIO | None = None) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=io or ScriptedIO())


def _resp(status: int, ctype: str, content: bytes) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": ctype}, content=content)


# ── http tool ────────────────────────────────────────────────────────────────
def test_http_render_pretty_prints_json():
    out = HttpTool._render("GET", "https://api.x/y", _resp(200, "application/json", b'{"a": 1, "b": 2}'))
    assert "200 OK" in out
    assert '"a": 1' in out  # pretty-printed, not the compact source


def test_http_render_truncates_long_text():
    out = HttpTool._render("GET", "https://x", _resp(200, "text/plain", b"x" * 40_000))
    assert "[body truncated]" in out


def test_http_render_skips_binary():
    out = HttpTool._render("GET", "https://x/img", _resp(200, "image/png", b"\x89PNG\r\n"))
    assert "non-text content not shown" in out


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
async def test_data_jsonl(tmp_path):
    (tmp_path / "e.jsonl").write_text('{"x": 1}\n{"x": 2}\n{"x": 3}\n')
    res = await DataTool().run({"path": "e.jsonl"}, ctx_for(tmp_path))
    assert "3 records" in res.content


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


def test_data_is_read_only():
    assert DataTool().is_read_only(object()) is True  # arg ignored; data never writes


# ── browse tool (offline guards only) ────────────────────────────────────────
@pytest.mark.asyncio
async def test_browse_rejects_bad_scheme(tmp_path):
    res = await BrowseTool().run({"url": "ftp://nope"}, ctx_for(tmp_path))
    assert res.is_error and "scheme" in res.content.lower()


def test_browse_flags_and_enabled_consistency():
    b = BrowseTool()
    assert b.name == "browse"
    assert b.is_read_only(object()) is True
    assert b.is_concurrency_safe(object()) is False
    assert b.is_enabled() == _playwright_available()


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
    assert {"http", "data", "browse"} <= set(ALL_TOOLS)
    pool = build_full_pool(tmp_path)
    assert {"http", "data", "browse", "web_search"} <= set(pool.names())

"""Tests for the tool-author engine: generation, sandbox validation, approval (E4+E5)."""

from __future__ import annotations

import pytest

from neurosurfer.tools.generated import GeneratedToolsConfig
from neurosurfer.graph.builder.tool_author import (
    ToolAuthor,
    ToolDraft,
    ToolGapSpec,
)

# ── candidate sources ────────────────────────────────────────────────────────────

_RAW_GOOD = '''\
from pydantic import BaseModel, Field
from neurosurfer.tools.base import Tool, ToolContext, ToolResult


class CountLinesArgs(BaseModel):
    path: str = Field(description="file to count")


class CountLinesTool(Tool):
    name = "count_lines"
    description = "Count the lines in a text file."
    input_model = CountLinesArgs

    def is_read_only(self, args):
        return True

    async def call(self, args, ctx):
        return ToolResult.ok("counted")
'''

_RAW_SYNTAX_ERROR = "def (this is not python"

_RAW_TWO_CLASSES = _RAW_GOOD + '''

class SecondTool(Tool):
    name = "second"
    description = "another"
    input_model = CountLinesArgs

    async def call(self, args, ctx):
        return ToolResult.ok("x")
'''

_RAW_RISKY = '''\
import os
from pydantic import BaseModel, Field
from neurosurfer.tools.base import Tool, ToolContext, ToolResult


class WipeArgs(BaseModel):
    path: str = Field(description="path")


class WipeTool(Tool):
    name = "count_lines"
    description = "does something."
    input_model = WipeArgs

    async def call(self, args, ctx):
        os.system("echo hi")
        return ToolResult.ok("ok")
'''


def _fence(code: str) -> str:
    return f"```python\n{code}\n```"


class _Resp:
    def __init__(self, text: str) -> None:
        self._t = text

    def text(self) -> str:
        return self._t


class _FakeProvider:
    """Returns canned replies in order; ToolAuthor only calls `.text()`."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0

    async def complete(self, messages, system, tools, config):  # noqa: ANN001
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return _Resp(reply)


def _author(provider, tmp_path) -> ToolAuthor:
    return ToolAuthor(provider, cfg=GeneratedToolsConfig(dir=tmp_path), max_attempts=3)


async def _yes(draft, result):  # noqa: ANN001
    return True


async def _no(draft, result):  # noqa: ANN001
    return False


# ── static + sandbox validation ──────────────────────────────────────────────────

def test_valid_draft_passes(tmp_path):
    author = _author(_FakeProvider([]), tmp_path)
    result = author.validate_draft(ToolDraft("count_lines", _RAW_GOOD, ToolGapSpec("count_lines", "count")))
    assert result.ok, result.render()
    assert result.checks["call_is_async"]
    assert result.checks["name_matches"]


def test_syntax_error_fails(tmp_path):
    author = _author(_FakeProvider([]), tmp_path)
    result = author.validate_draft(ToolDraft("x", _RAW_SYNTAX_ERROR, ToolGapSpec("x", "x")))
    assert not result.ok
    assert result.checks.get("parses") is False


def test_name_mismatch_fails(tmp_path):
    author = _author(_FakeProvider([]), tmp_path)
    # spec/expected name 'wrong' but code defines name 'count_lines'
    result = author.validate_draft(ToolDraft("wrong", _RAW_GOOD, ToolGapSpec("wrong", "x")))
    assert not result.ok
    assert result.checks.get("name_matches") is False


def test_two_tool_classes_fails(tmp_path):
    author = _author(_FakeProvider([]), tmp_path)
    result = author.validate_draft(ToolDraft("count_lines", _RAW_TWO_CLASSES, ToolGapSpec("count_lines", "x")))
    assert not result.ok
    assert result.checks.get("single_tool_class") is False


def test_risky_tokens_flagged_as_warnings(tmp_path):
    author = _author(_FakeProvider([]), tmp_path)
    result = author.validate_draft(ToolDraft("count_lines", _RAW_RISKY, ToolGapSpec("count_lines", "x")))
    # still structurally valid, but os.system is surfaced as a warning
    assert any("os.system" in w for w in result.warnings)


# ── end-to-end authoring with approval ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_author_registers_on_approval(tmp_path):
    provider = _FakeProvider([_fence(_RAW_GOOD)])
    author = _author(provider, tmp_path)
    meta = await author.author(ToolGapSpec("count_lines", "count lines"), approve=_yes)
    assert meta is not None
    assert meta.name == "count_lines"
    assert author.cfg.tool_path("count_lines").exists()
    assert author.cfg.meta_path("count_lines").exists()


@pytest.mark.asyncio
async def test_author_does_not_register_on_rejection(tmp_path):
    provider = _FakeProvider([_fence(_RAW_GOOD)])
    author = _author(provider, tmp_path)
    meta = await author.author(ToolGapSpec("count_lines", "count lines"), approve=_no)
    assert meta is None
    assert not author.cfg.tool_path("count_lines").exists()


@pytest.mark.asyncio
async def test_author_retries_then_succeeds(tmp_path):
    # first reply is broken, second is valid → should retry and register
    provider = _FakeProvider([_fence(_RAW_SYNTAX_ERROR), _fence(_RAW_GOOD)])
    author = _author(provider, tmp_path)
    meta = await author.author(ToolGapSpec("count_lines", "count lines"), approve=_yes)
    assert meta is not None
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_author_gives_up_after_max_attempts(tmp_path):
    provider = _FakeProvider([_fence(_RAW_SYNTAX_ERROR)])  # always broken
    author = _author(provider, tmp_path)
    meta = await author.author(ToolGapSpec("count_lines", "x"), approve=_yes)
    assert meta is None
    assert provider.calls == 3  # max_attempts
    assert not author.cfg.tool_path("count_lines").exists()


@pytest.mark.asyncio
async def test_approval_never_reached_for_invalid_code(tmp_path):
    provider = _FakeProvider([_fence(_RAW_SYNTAX_ERROR)])
    author = _author(provider, tmp_path)
    seen = []

    async def approve(draft, result):
        seen.append(draft)
        return True

    meta = await author.author(ToolGapSpec("count_lines", "x"), approve=approve)
    assert meta is None
    assert seen == []  # approval gate is only reached by code that passed validation

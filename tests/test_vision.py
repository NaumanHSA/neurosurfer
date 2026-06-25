"""Phase 3 (vision / multimodal) tests.

Covers:
  - ImageBlock projection through the Anthropic adapter (base64 + url).
  - ImageBlock projection through the OpenAI adapter (multimodal user turn).
  - Non-vision capability drops images with a text note (both adapters).
  - ToolResult.images flow into history via the agent loop.
  - read_file returns an ImageBlock for image files.
  - Token heuristic counts images.
"""

from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel

from neurosurfer.agents.agentic_loop import AgenticLoop
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.llm.providers.anthropic import to_anthropic_messages
from neurosurfer.llm.providers.openai import to_openai_messages
from neurosurfer.llm.tokens import IMAGE_TOKENS, estimate_messages_tokens
from neurosurfer.llm.types import ImageBlock, Message, ToolResultBlock
from neurosurfer.tools.base import Tool, ToolPool, ToolResult
from tests.fakes import ScriptedIO, ScriptedProvider

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic projection
# ──────────────────────────────────────────────────────────────────────────────
def test_anthropic_base64_image() -> None:
    msg = Message.user_with_images("look", [ImageBlock.from_base64(_PNG_B64, "image/png")])
    out = to_anthropic_messages([msg], supports_vision=True)
    parts = out[0]["content"]
    img = next(p for p in parts if p["type"] == "image")
    assert img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/png"
    assert img["source"]["data"] == _PNG_B64


def test_anthropic_url_image() -> None:
    msg = Message.user_with_images("look", [ImageBlock.from_url("https://h/a.png")])
    out = to_anthropic_messages([msg], supports_vision=True)
    img = next(p for p in out[0]["content"] if p["type"] == "image")
    assert img["source"] == {"type": "url", "url": "https://h/a.png"}


def test_anthropic_drops_image_when_no_vision() -> None:
    msg = Message.user_with_images("look", [ImageBlock.from_base64(_PNG_B64)])
    out = to_anthropic_messages([msg], supports_vision=False)
    types = [p["type"] for p in out[0]["content"]]
    assert "image" not in types
    assert any("omitted" in p.get("text", "") for p in out[0]["content"])


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI projection
# ──────────────────────────────────────────────────────────────────────────────
def test_openai_image_multimodal_turn() -> None:
    msg = Message.user_with_images("look", [ImageBlock.from_base64(_PNG_B64, "image/png")])
    out = to_openai_messages([msg], None, supports_vision=True)
    user = out[-1]
    assert user["role"] == "user"
    assert isinstance(user["content"], list)
    img = next(p for p in user["content"] if p["type"] == "image_url")
    assert img["image_url"]["url"] == f"data:image/png;base64,{_PNG_B64}"


def test_openai_url_image() -> None:
    msg = Message.user_with_images("", [ImageBlock.from_url("https://h/a.png")])
    out = to_openai_messages([msg], None, supports_vision=True)
    img = next(p for p in out[-1]["content"] if p["type"] == "image_url")
    assert img["image_url"]["url"] == "https://h/a.png"


def test_openai_drops_image_when_no_vision() -> None:
    msg = Message.user_with_images("look", [ImageBlock.from_base64(_PNG_B64)])
    out = to_openai_messages([msg], None, supports_vision=False)
    user = out[-1]
    assert isinstance(user["content"], str)  # plain string, not multimodal parts
    assert "omitted" in user["content"]


def test_openai_image_alongside_tool_result() -> None:
    # tool_result blocks become tool messages; the image rides a separate user turn.
    msg = Message(
        role="user",
        content=[
            ToolResultBlock(tool_use_id="t1", content="done"),
            ImageBlock.from_base64(_PNG_B64),
        ],
    )
    out = to_openai_messages([msg], None, supports_vision=True)
    assert out[0]["role"] == "tool" and out[0]["tool_call_id"] == "t1"
    assert out[1]["role"] == "user" and isinstance(out[1]["content"], list)


# ──────────────────────────────────────────────────────────────────────────────
# Tokens
# ──────────────────────────────────────────────────────────────────────────────
def test_image_token_estimate() -> None:
    text_only = [Message.user_text("hi")]
    with_img = [Message.user_with_images("hi", [ImageBlock.from_base64(_PNG_B64)])]
    assert estimate_messages_tokens(with_img) - estimate_messages_tokens(text_only) >= IMAGE_TOKENS


# ──────────────────────────────────────────────────────────────────────────────
# ToolResult.images → history (agent loop)
# ──────────────────────────────────────────────────────────────────────────────
class _SnapArgs(BaseModel):
    pass


class _ScreenshotTool(Tool):
    name = "snap"
    description = "returns an image"
    input_model = _SnapArgs

    async def call(self, args, ctx):  # type: ignore[override]
        return ToolResult.with_images("snapped", [ImageBlock.from_base64(_PNG_B64)])


async def test_tool_images_land_in_history(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            ("Snapping.", [("snap", {})]),
            ("Done.", [("finish", {"summary": "ok", "status": "success"})]),
        ]
    )
    from neurosurfer.tools.builtin import FinishTool

    pool = ToolPool([_ScreenshotTool(), FinishTool()])
    agent = AgenticLoop(
        provider=provider,
        tools=pool,
        system_prompt="x",
        guardrails=Guardrails(),
        io=ScriptedIO(),
        cwd=tmp_path,
    )
    async for _ in agent.run("go"):
        pass

    # The tool-results user turn should carry the ImageBlock after the tool_result.
    image_msgs = [
        m
        for m in agent.history.messages
        if m.role == "user" and any(isinstance(b, ImageBlock) for b in m.content)
    ]
    assert image_msgs, "expected an ImageBlock in history"
    blocks = image_msgs[0].content
    assert isinstance(blocks[0], ToolResultBlock)  # tool_result first
    assert any(isinstance(b, ImageBlock) for b in blocks)


# ──────────────────────────────────────────────────────────────────────────────
# read_file image support
# ──────────────────────────────────────────────────────────────────────────────
async def test_read_file_returns_image(tmp_path: Path) -> None:
    from neurosurfer.tools.base import ToolContext
    from neurosurfer.tools.builtin import ReadFileTool

    img_path = tmp_path / "pic.png"
    img_path.write_bytes(_PNG_BYTES)

    tool = ReadFileTool()
    res = await tool.run({"path": "pic.png"}, ToolContext(cwd=tmp_path, io=ScriptedIO()))
    assert res.is_error is False
    assert len(res.images) == 1
    assert res.images[0].media_type == "image/png"
    assert res.images[0].data == _PNG_B64

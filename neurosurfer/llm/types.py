"""Canonical, provider-neutral message / content / event model.

The engine depends only on these types and on the ``Provider`` protocol — never
on ``anthropic`` or ``openai`` directly. Each provider adapter translates between
these canonical types and its own wire format.

Canonical content blocks mirror Anthropic's block model (text / thinking /
tool_use / tool_result) because it is the richer of the two; the OpenAI adapter
projects them onto chat-completion's flatter shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────────────────────
# Content blocks
# ──────────────────────────────────────────────────────────────────────────────


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """Extended-thinking block. Anthropic-only; carries a signature that must be
    preserved verbatim when appended back into history."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str = ""
    is_error: bool = False


class ImageBlock(BaseModel):
    """An image provided to a vision-capable model.

    Two sources: an inline base64 payload (``source="base64"`` + ``media_type`` +
    ``data``) or a remote URL (``source="url"`` + ``url``). Providers project this
    onto their own wire format; non-vision models drop it (with a text note) so a
    run never hard-fails just because an image slipped into history.
    """

    type: Literal["image"] = "image"
    source: Literal["base64", "url"] = "base64"
    media_type: str = "image/png"  # used when source == "base64"
    data: str | None = None  # base64-encoded bytes when source == "base64"
    url: str | None = None  # when source == "url"

    @classmethod
    def from_base64(cls, data: str, media_type: str = "image/png") -> ImageBlock:
        return cls(source="base64", data=data, media_type=media_type)

    @classmethod
    def from_url(cls, url: str) -> ImageBlock:
        return cls(source="url", url=url)


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock | ImageBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """A single conversation turn. ``system`` is passed out-of-band to the
    provider, so a message role is only ever ``user`` or ``assistant``."""

    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_string_content(cls, v: object) -> object:
        if isinstance(v, str):
            return [{"type": "text", "text": v}]
        return v

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @classmethod
    def user_text(cls, text: str) -> Message:
        return cls(role="user", content=[TextBlock(text=text)])

    @classmethod
    def user_with_images(cls, text: str, images: list[ImageBlock]) -> Message:
        """A user turn carrying text plus one or more images."""
        content: list[ContentBlock] = []
        if text:
            content.append(TextBlock(text=text))
        content.extend(images)
        return cls(role="user", content=content)

    @classmethod
    def assistant_text(cls, text: str) -> Message:
        return cls(role="assistant", content=[TextBlock(text=text)])


# ──────────────────────────────────────────────────────────────────────────────
# Tools, usage, config
# ──────────────────────────────────────────────────────────────────────────────


class ToolSchema(BaseModel):
    """Provider-neutral tool description. Rendered to Anthropic's tool format or
    OpenAI's ``function`` format by the respective adapter."""

    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    def add(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
        )


@dataclass
class GenerationConfig:
    # Sampling / budget knobs are PROVIDER-OWNED: leave them None and the provider
    # fills them from its own capabilities at request time (token budget from the
    # model's max output, temperature/effort per the model's rules). Agents should
    # not set these — they don't know a given model's constraints (e.g. gpt-5 and
    # the o-series reject any non-default temperature). Set one explicitly only to
    # deliberately override the provider default for a specific call.
    max_tokens: int | None = None
    temperature: float | None = None
    effort: str | None = None
    # Call/task semantics (safe for agents to set — honored uniformly):
    # whether to allow thinking, hard stop sequences, and streaming vs. one-shot.
    enable_thinking: bool = True
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = True


@dataclass
class CanonicalResponse:
    """The assembled assistant turn — what gets appended back to history."""

    content: list[ContentBlock]
    stop_reason: str
    usage: Usage
    model: str = ""

    def text(self) -> str:
        parts = [b.text for b in self.content if isinstance(b, TextBlock)]
        if parts:
            return "".join(parts)
        # Thinking-only models (DeepSeek-R1, Gemma QAT, …) put everything in
        # reasoning_content; fall back so callers always get a non-empty string.
        return "".join(b.thinking for b in self.content if isinstance(b, ThinkingBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def as_message(self) -> Message:
        return Message(role="assistant", content=list(self.content))


# ──────────────────────────────────────────────────────────────────────────────
# Streaming events (provider-normalized)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolUseStart:
    index: int
    id: str
    name: str


@dataclass
class ToolUseArgsDelta:
    index: int
    partial_json: str


@dataclass
class Done:
    """Terminal event carrying the fully assembled response."""

    response: CanonicalResponse


StreamEvent = TextDelta | ThinkingDelta | ToolUseStart | ToolUseArgsDelta | Done

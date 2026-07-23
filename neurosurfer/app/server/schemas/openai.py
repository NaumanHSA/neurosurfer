from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAIBase(BaseModel):
    model_config = ConfigDict(extra="allow")


class ModelPermission(OpenAIBase):
    id: str
    object: Literal["model_permission"] = "model_permission"
    created: int
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelCard(OpenAIBase):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "neurosurfer"
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None
    permission: list[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBase):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ChatMessage(OpenAIBase):
    role: str
    content: Any = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None


class ChatCompletionRequest(OpenAIBase):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    user: str | None = None
    response_format: dict | None = None
    stream_options: dict | None = None
    metadata: dict | None = None


class ChatCompletionChoice(OpenAIBase):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class CompletionUsage(OpenAIBase):
    """OpenAI-compatible token accounting. Tokens only — cost is a monitoring
    backend's concern, never the gateway's. Real counts when the provider reports
    them; estimated (tiktoken/char heuristic) for servers that don't."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(OpenAIBase):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage | None = None


class ChatCompletionChunkDelta(OpenAIBase):
    role: str | None = None
    content: str | None = None
    tool_calls: list[dict] | None = None


class ChatCompletionChunkChoice(OpenAIBase):
    index: int = 0
    delta: ChatCompletionChunkDelta = Field(default_factory=ChatCompletionChunkDelta)
    finish_reason: str | None = None


class ChatCompletionChunk(OpenAIBase):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    # Present only on the final usage chunk (OpenAI sends it when the request set
    # `stream_options.include_usage`); that chunk carries an empty `choices` list.
    usage: CompletionUsage | None = None
